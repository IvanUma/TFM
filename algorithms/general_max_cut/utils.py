from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from . import max_cut_common as common

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

APPROACH: str = CONFIG["approach"]

PROBLEM_TYPE: str = CONFIG.get("problem", {}).get("type", "maxcut")

_encoding_cfg = CONFIG.get("encoding", {})
ENABLE_INPUT_PARAMS: bool = _encoding_cfg.get("enable_input_params", False)
PARAM_BLOCK_PROB: float = _encoding_cfg.get("param_block_prob", 0.15)

_manual_input_values = _encoding_cfg.get("manual_input_values", [1.0])
if not isinstance(_manual_input_values, list) or len(_manual_input_values) == 0:
    raise ValueError("encoding.manual_input_values must be a non-empty list of numbers")
MANUAL_INPUT_VALUES: List[float] = [float(v) for v in _manual_input_values]
NUM_PARAMS: int = len(MANUAL_INPUT_VALUES)

_scale_cfg = CONFIG.get("circuit_scale", {})
MAX_QUBITS = _scale_cfg.get("max_qubits")
if not MAX_QUBITS:
    raise ValueError("circuit_scale.max_qubits must be set for the general approach")
INSTANCE_QUBITS_FILTER = _scale_cfg.get("instance_qubits_filter")
MAX_INSTANCES_PER_SIZE: int | None = _scale_cfg.get("max_instances_per_size")

_split_cfg = CONFIG.get("instance_split", {})
VALIDATION_FRACTION: float = _split_cfg.get("validation_fraction", 0.2)
SPLIT_SEED: int = _split_cfg.get("seed", 42)

q_strategy = importlib.import_module(f".max_cut_{APPROACH}", package=__package__)

EvolutionaryIndividual = q_strategy.EvolutionaryIndividual

build_quantum_circuit = q_strategy.build_quantum_circuit
get_param_indices = q_strategy.get_param_indices
generate_guided_individual = q_strategy.generate_guided_individual
mut_quantum_circuit = q_strategy.mut_quantum_circuit
serialize_individual = q_strategy.serialize_individual
deserialize_individual = q_strategy.deserialize_individual

generate_heuristic_individual = common.generate_heuristic_individual
load_external_maxcut_instance = common.load_external_maxcut_instance
cx_quantum_circuit = common.cx_quantum_circuit
max_cut_fitness = common.max_cut_fitness

InstanceData = Tuple[nx.Graph, float]

_execution_cfg = CONFIG.get("execution", {})
_STABILIZER_THREADS: int = _execution_cfg.get("stabilizer_max_parallel_threads", 1)
_STATEVECTOR_THREADS: int = _execution_cfg.get("statevector_max_parallel_threads", 0)
_REQUESTED_DEVICE: str = _execution_cfg.get("device", "auto")

SIMULATOR_METHOD: str = "stabilizer" if APPROACH == "clifford" else "statevector"
SIMULATOR_THREADS: int = (
    _STABILIZER_THREADS if APPROACH == "clifford" else _STATEVECTOR_THREADS
)


def _resolve_device(requested_device: str, method: str) -> str:
    if method != "statevector":
        return "CPU"

    try:
        available = AerSimulator().available_devices()
    except Exception as exc:
        logger.warning("Could not query available simulator devices (%s)", exc)
        available = ("CPU",)

    if requested_device == "CPU":
        return "CPU"

    if requested_device == "GPU":
        if "GPU" in available:
            return "GPU"
        logger.warning("GPU requested but not available; falling back to CPU")
        return "CPU"

    return "GPU" if "GPU" in available else "CPU"


SIMULATOR_DEVICE: str = _resolve_device(_REQUESTED_DEVICE, SIMULATOR_METHOD)

_TRAINING_SIMULATOR = None


def get_training_simulator() -> AerSimulator:
    global _TRAINING_SIMULATOR
    if _TRAINING_SIMULATOR is None:
        try:
            _TRAINING_SIMULATOR = AerSimulator(
                method=SIMULATOR_METHOD,
                device=SIMULATOR_DEVICE,
                max_parallel_threads=SIMULATOR_THREADS,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize simulator with device=%s (%s); falling back to CPU",
                SIMULATOR_DEVICE,
                exc,
            )
            _TRAINING_SIMULATOR = AerSimulator(
                method=SIMULATOR_METHOD,
                device="CPU",
                max_parallel_threads=SIMULATOR_THREADS,
            )
    return _TRAINING_SIMULATOR


def update_hof(block_gates: list) -> None:
    if APPROACH != "clifford":
        return

    if block_gates not in q_strategy.BLOCK_HOF:
        q_strategy.BLOCK_HOF.append(block_gates)

    if len(q_strategy.BLOCK_HOF) > 100:
        q_strategy.BLOCK_HOF.pop(0)


def describe_architecture(individual: EvolutionaryIndividual) -> dict:
    if APPROACH == "clifford":
        return {"blocks": q_strategy.describe_blocks(individual)}
    return {"param_genes": q_strategy.describe_param_genes(individual)}


def serialize_architecture(individual: EvolutionaryIndividual) -> list:
    return serialize_individual(individual)


def deserialize_architecture(data: list) -> EvolutionaryIndividual:
    return deserialize_individual(data)


_QNN_TRAIN_DATA: Tuple[np.ndarray, np.ndarray] | None = None
_QNN_VAL_DATA: Tuple[np.ndarray, np.ndarray] | None = None


def set_qnn_data(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
) -> None:
    global _QNN_TRAIN_DATA, _QNN_VAL_DATA
    _QNN_TRAIN_DATA = train_data
    _QNN_VAL_DATA = val_data


def _qnn_classify(
    counts: Dict[str, float],
    n_classes: int,
) -> int:
    total = sum(counts.values())
    if total == 0:
        return 0
    probs = {}
    for bitstring, count in counts.items():
        probs[int(bitstring.replace(" ", ""), 2)] = (
            probs.get(int(bitstring.replace(" ", ""), 2), 0) + count
        )
    best_class = max(range(2**n_classes), key=lambda c: probs.get(c, 0))
    return min(best_class, n_classes - 1)


def _qnn_accuracy(
    individual,
    num_qubits: int,
    X_data: np.ndarray,
    y_data: np.ndarray,
    weight_map: Dict[int, float],
    shots: int,
    n_classes: int,
    simulator: AerSimulator,
) -> float:
    correct = 0
    total = len(y_data)
    for i in range(total):
        state_vector = X_data[i]
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.initialize(state_vector, range(num_qubits))
        circuit = build_quantum_circuit(
            individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=False
        )
        qc.compose(circuit, inplace=True)
        qc.measure_all()
        counts = simulator.run(qc, shots=shots).result().get_counts()
        pred = _qnn_classify(counts, n_classes)
        if pred == y_data[i]:
            correct += 1
    return correct / total if total > 0 else 0.0


_WEIGHT_CACHE: Dict[str, np.ndarray] = {}
_CIRCUIT_CACHE: Dict[str, QuantumCircuit] = {}


def _individual_cache_key(individual: EvolutionaryIndividual) -> str:
    parts = []
    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            parts.append(f"P{gen[1]}")
        else:
            parts.append(str(gen[0]))
    return "_".join(parts)


def _build_param_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    sorted_weight_indices: List[int],
    weight_map: Dict[int, float],
) -> QuantumCircuit:
    if APPROACH == "rotation":
        from qiskit.circuit import Parameter
        from qiskit import QuantumCircuit as QCircuit

        key = _individual_cache_key(individual) + f"_{num_qubits}"
        cached = _CIRCUIT_CACHE.get(key)
        if cached is not None:
            param_circuit, param_indices, param_objs = cached
            assignments = {param_objs[i]: weight_map.get(i, 0.0) for i in param_indices}
            return param_circuit.assign_parameters(assignments, inplace=False)

        param_indices = sorted(set(sorted_weight_indices))
        param_objs = {i: Parameter(f"w_{i}") for i in param_indices}
        qc = QCircuit(num_qubits)

        for gen in individual:
            if gen[0] == "PARAM_BLOCK":
                _, p_type, p_idx, rot_gate, qubit = gen
                if p_type == "INPUT":
                    theta = (
                        MANUAL_INPUT_VALUES[p_idx % len(MANUAL_INPUT_VALUES)]
                        if MANUAL_INPUT_VALUES
                        else 0.0
                    )
                else:
                    theta = param_objs.get(p_idx, 0.0)
                getattr(qc, rot_gate.lower())(theta, qubit)
            else:
                if gen[0] == "H":
                    qc.h(gen[1])
                elif gen[0] == "S":
                    qc.s(gen[1])
                elif gen[0] == "CX":
                    qc.cx(gen[1], gen[2])

        qc.measure_all()
        _CIRCUIT_CACHE[key] = (qc, param_indices, param_objs)
        if len(_CIRCUIT_CACHE) > 200:
            _CIRCUIT_CACHE.pop(next(iter(_CIRCUIT_CACHE)))

        assignments = {param_objs[i]: weight_map.get(i, 0.0) for i in param_indices}
        return qc.assign_parameters(assignments, inplace=False)

    from qiskit import QuantumCircuit as QCircuit

    quantized = tuple(round(weight_map.get(i, 0.0), 2) for i in sorted_weight_indices)
    key = _individual_cache_key(individual) + f"_{num_qubits}_" + str(quantized)
    cached = _CIRCUIT_CACHE.get(key)
    if cached is not None:
        return cached

    qc = build_quantum_circuit(
        individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=True
    )
    _CIRCUIT_CACHE[key] = qc
    if len(_CIRCUIT_CACHE) > 500:
        _CIRCUIT_CACHE.pop(next(iter(_CIRCUIT_CACHE)))
    return qc


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    instances: List[InstanceData],
    shots: int,
    gamma: float,
) -> Tuple[Tuple[float, float], Dict[int, float], List[list], float]:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    simulator = get_training_simulator()
    simulation_seconds = 0.0

    def objective(weight_vector) -> float:
        nonlocal simulation_seconds
        weight_map = dict(zip(sorted_weight_indices, weight_vector))
        qc = _build_param_circuit(
            individual, num_qubits, sorted_weight_indices, weight_map
        )
        sim_start = time.perf_counter()
        counts = simulator.run(qc, shots=shots).result().get_counts()
        simulation_seconds += time.perf_counter() - sim_start

        ratios = []
        for graph_instance, optimal_cut in instances:
            cvar_cut = max_cut_fitness(counts, graph_instance, alpha=gamma)
            ratios.append(cvar_cut / optimal_cut if optimal_cut > 0 else 0.0)

        mean_ar = sum(ratios) / len(ratios)
        if len(ratios) > 1:
            std_ar = (sum((r - mean_ar) ** 2 for r in ratios) / len(ratios)) ** 0.5
            return -(mean_ar - 0.2 * std_ar)
        return -mean_ar

    if num_weights > 0:
        maxiter = max(50, num_weights * 15)
        cache_key = _individual_cache_key(individual) + f"_{shots}_{gamma}"
        inherited = getattr(individual, "_seed_weights", None)
        if inherited is not None:
            x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
        else:
            seed = _WEIGHT_CACHE.get(cache_key)
            x0 = (
                seed
                if seed is not None
                else np.random.uniform(0, 2 * np.pi, size=num_weights)
            )
        result = minimize(
            objective,
            x0=x0,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5},
        )
        best_avg_ratio = -result.fun
        best_weights = dict(zip(sorted_weight_indices, (float(w) for w in result.x)))
        _WEIGHT_CACHE[cache_key] = result.x.copy()
        if len(_WEIGHT_CACHE) > 2000:
            _WEIGHT_CACHE.pop(next(iter(_WEIGHT_CACHE)))
    else:
        best_avg_ratio = -objective([])
        best_weights = {}

    depth = build_quantum_circuit(
        individual, num_qubits, MANUAL_INPUT_VALUES, best_weights, measure=False
    ).depth()

    hof_candidates: List[list] = []
    if best_avg_ratio > 0.8 and APPROACH == "clifford":
        hof_candidates = [gen[2] for gen in individual if gen[0] == "PARAM_BLOCK"]

    return (
        (-best_avg_ratio, float(depth)),
        best_weights,
        hof_candidates,
        simulation_seconds,
    )


def validate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    validation_instances,
    shots: int,
    seed_weights: dict | None = None,
) -> float:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    simulator = get_training_simulator()

    def objective(weight_vector) -> float:
        weight_map = dict(zip(sorted_weight_indices, weight_vector))
        qc = build_quantum_circuit(
            individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=True
        )
        counts = simulator.run(qc, shots=shots).result().get_counts()

        ratios = []
        for _, graph, optimal_cut in validation_instances:
            cut = max_cut_fitness(counts, graph, alpha=1.0)
            ratios.append(cut / optimal_cut if optimal_cut > 0 else 0.0)

        return -sum(ratios) / len(ratios)

    if num_weights > 0:
        if seed_weights is not None:
            x0 = np.array([seed_weights.get(i, 0.0) for i in sorted_weight_indices])
        else:
            rng = np.random.RandomState(42)
            x0 = rng.uniform(0, 2 * np.pi, size=num_weights)

        maxiter = max(50, num_weights * 10)
        result = minimize(
            objective,
            x0=x0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        return -result.fun
    else:
        return -objective([])
