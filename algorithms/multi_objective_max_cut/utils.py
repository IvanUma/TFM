from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

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


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    graph_instance: nx.Graph,
    optimal_classical_cut: float,
    input_values: List[float],
    shots: int,
    gamma: float,
) -> Tuple[Tuple[float, float], List[float], List[list]]:
    _, weight_indices = get_param_indices(individual)
    num_weights = max(weight_indices) + 1 if weight_indices else 0

    simulator = get_training_simulator()

    def objective(weight_values) -> float:
        qc = build_quantum_circuit(
            individual, num_qubits, input_values, list(weight_values), measure=True
        )
        counts = simulator.run(qc, shots=shots).result().get_counts()
        cvar_cut = max_cut_fitness(counts, graph_instance, alpha=gamma)
        return -cvar_cut

    if num_weights > 0:
        maxiter = max(20, num_weights * 3)
        result = minimize(
            objective,
            x0=np.random.uniform(0, 2 * np.pi, size=num_weights),
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        best_cut = -result.fun
        best_weights = [float(w) for w in result.x]
    else:
        best_cut = -objective([])
        best_weights = []

    approx_ratio = (
        best_cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0
    )

    depth = build_quantum_circuit(
        individual, num_qubits, input_values, best_weights, measure=False
    ).depth()

    hof_candidates: List[list] = []
    if approx_ratio > 0.8 and APPROACH == "clifford":
        hof_candidates = [gen[2] for gen in individual if gen[0] == "PARAM_BLOCK"]

    return (-approx_ratio, depth), best_weights, hof_candidates
