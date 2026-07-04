from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from . import max_cut_common as common

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

APPROACH: str = CONFIG["approach"]

_encoding_cfg = CONFIG.get("encoding", {})
ENABLE_INPUT_PARAMS: bool = _encoding_cfg.get("enable_input_params", False)
PARAM_BLOCK_PROB: float = _encoding_cfg.get("param_block_prob", 0.15)

_scale_cfg = CONFIG.get("circuit_scale", {})
MAX_QUBITS = _scale_cfg.get("max_qubits")
if not MAX_QUBITS:
    raise ValueError("circuit_scale.max_qubits must be set for the general approach")
INSTANCE_QUBITS_FILTER = _scale_cfg.get("instance_qubits_filter")

_split_cfg = CONFIG.get("instance_split", {})
VALIDATION_FRACTION: float = _split_cfg.get("validation_fraction", 0.2)
SPLIT_SEED: int = _split_cfg.get("seed", 42)

q_strategy = importlib.import_module(f".max_cut_{APPROACH}", package=__package__)

EvolutionaryIndividual = q_strategy.EvolutionaryIndividual

build_quantum_circuit = q_strategy.build_quantum_circuit
get_param_indices = q_strategy.get_param_indices
generate_guided_individual = q_strategy.generate_guided_individual
mut_quantum_circuit = q_strategy.mut_quantum_circuit

generate_heuristic_individual = common.generate_heuristic_individual
load_external_maxcut_instance = common.load_external_maxcut_instance
cx_quantum_circuit = common.cx_quantum_circuit
max_cut_fitness = common.max_cut_fitness
build_universal_input_values = common.build_universal_input_values

InstanceData = Tuple[nx.Graph, float, List[float]]

_TRAINING_SIMULATOR = None


def get_training_simulator() -> AerSimulator:
    global _TRAINING_SIMULATOR
    if _TRAINING_SIMULATOR is None:
        method = "stabilizer" if APPROACH == "clifford" else "statevector"
        _TRAINING_SIMULATOR = AerSimulator(method=method, max_parallel_threads=1)
    return _TRAINING_SIMULATOR


def update_hof(block_gates: list) -> None:
    if APPROACH != "clifford":
        return

    if block_gates not in q_strategy.BLOCK_HOF:
        q_strategy.BLOCK_HOF.append(block_gates)

    if len(q_strategy.BLOCK_HOF) > 50:
        q_strategy.BLOCK_HOF.pop(0)


def describe_architecture(individual: EvolutionaryIndividual) -> dict:
    if APPROACH == "clifford":
        return {"blocks": q_strategy.describe_blocks(individual)}
    return {"param_genes": q_strategy.describe_param_genes(individual, MAX_QUBITS)}


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    instances: List[InstanceData],
    shots: int,
    gamma: float,
) -> Tuple[Tuple[float, float], Dict[int, float], List[list]]:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    simulator = get_training_simulator()

    def objective(weight_vector) -> float:
        weight_map = dict(zip(sorted_weight_indices, weight_vector))
        circuits = [
            build_quantum_circuit(
                individual, num_qubits, inst_input, weight_map, measure=True
            )
            for _, _, inst_input in instances
        ]
        results = simulator.run(circuits, shots=shots).result()

        ratios = []
        for idx, (graph_instance, optimal_cut, _) in enumerate(instances):
            counts = results.get_counts(idx)
            cvar_cut = max_cut_fitness(counts, graph_instance, alpha=gamma)
            ratios.append(cvar_cut / optimal_cut if optimal_cut > 0 else 0.0)

        return -sum(ratios) / len(ratios)

    if num_weights > 0:
        maxiter = max(20, num_weights * 3)
        result = minimize(
            objective,
            x0=np.random.uniform(0, 2 * np.pi, size=num_weights),
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        best_avg_ratio = -result.fun
        best_weights = dict(zip(sorted_weight_indices, (float(w) for w in result.x)))
    else:
        best_avg_ratio = -objective([])
        best_weights = {}

    depths = [
        build_quantum_circuit(
            individual, num_qubits, inst_input, best_weights, measure=False
        ).depth()
        for _, _, inst_input in instances
    ]
    avg_depth = sum(depths) / len(depths)

    hof_candidates: List[list] = []
    if best_avg_ratio > 0.8 and APPROACH == "clifford":
        hof_candidates = [gen[2] for gen in individual if gen[0] == "PARAM_BLOCK"]

    return (-best_avg_ratio, avg_depth), best_weights, hof_candidates
