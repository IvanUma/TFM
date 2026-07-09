from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from . import qnn_common as common

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

APPROACH: str = CONFIG["approach"]

_qnn_cfg = CONFIG.get("qnn", {})
DATASET_NAME: str = _qnn_cfg.get("dataset", "iris")
TEST_SPLIT: float = _qnn_cfg.get("test_split", 0.2)

_encoding_cfg = CONFIG.get("encoding", {})
ENABLE_INPUT_PARAMS: bool = _encoding_cfg.get("enable_input_params", False)
PARAM_BLOCK_PROB: float = _encoding_cfg.get("param_block_prob", 0.15)

_manual_input_values = _encoding_cfg.get("manual_input_values", [1.0])
if not isinstance(_manual_input_values, list) or len(_manual_input_values) == 0:
    raise ValueError("encoding.manual_input_values must be a non-empty list of numbers")
MANUAL_INPUT_VALUES: List[float] = [float(v) for v in _manual_input_values]
NUM_PARAMS: int = len(MANUAL_INPUT_VALUES)

q_strategy = importlib.import_module(f".qnn_{APPROACH}", package=__package__)

EvolutionaryIndividual = q_strategy.EvolutionaryIndividual
build_quantum_circuit = q_strategy.build_quantum_circuit
get_param_indices = q_strategy.get_param_indices
generate_guided_individual = q_strategy.generate_guided_individual
mut_quantum_circuit = q_strategy.mut_quantum_circuit
serialize_individual = q_strategy.serialize_individual
deserialize_individual = q_strategy.deserialize_individual

cx_quantum_circuit = common.cx_quantum_circuit

_execution_cfg = CONFIG.get("execution", {})
_STATEVECTOR_THREADS: int = _execution_cfg.get("statevector_max_parallel_threads", 0)
_REQUESTED_DEVICE: str = _execution_cfg.get("device", "auto")


def _resolve_device(requested_device: str) -> str:
    if requested_device == "CPU":
        return "CPU"
    try:
        available = AerSimulator().available_devices()
    except Exception:
        available = ("CPU",)
    if requested_device == "GPU" and "GPU" in available:
        return "GPU"
    if "GPU" in available:
        return "GPU"
    return "CPU"


SIMULATOR_DEVICE: str = _resolve_device(_REQUESTED_DEVICE)

_TRAINING_SIMULATOR = None


def get_training_simulator() -> AerSimulator:
    global _TRAINING_SIMULATOR
    if _TRAINING_SIMULATOR is None:
        try:
            _TRAINING_SIMULATOR = AerSimulator(
                method="statevector",
                device=SIMULATOR_DEVICE,
                max_parallel_threads=_STATEVECTOR_THREADS,
            )
        except Exception as exc:
            logger.warning("Failed to init statevector simulator (%s); fallback CPU", exc)
            _TRAINING_SIMULATOR = AerSimulator(method="statevector", device="CPU")
    return _TRAINING_SIMULATOR


def describe_architecture(individual: EvolutionaryIndividual) -> dict:
    if APPROACH == "clifford":
        return {"blocks": q_strategy.describe_blocks(individual)}
    return {"param_genes": q_strategy.describe_param_genes(individual)}


def serialize_architecture(individual: EvolutionaryIndividual) -> list:
    return serialize_individual(individual)


def deserialize_architecture(data: list) -> EvolutionaryIndividual:
    return deserialize_individual(data)


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


def _classify_from_counts(counts: Dict[str, float], n_classes: int) -> int:
    total = sum(counts.values())
    if total == 0:
        return 0
    class_probs = {}
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        class_probs[idx] = class_probs.get(idx, 0) + count
    best = max(range(n_classes), key=lambda c: class_probs.get(c, 0))
    return min(best, n_classes - 1)


def _qnn_accuracy(
    individual: EvolutionaryIndividual,
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
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.initialize(X_data[i], range(num_qubits))
        circuit = build_quantum_circuit(individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=False)
        qc.compose(circuit, inplace=True)
        qc.measure_all()
        counts = simulator.run(qc, shots=shots).result().get_counts()
        pred = _classify_from_counts(counts, n_classes)
        if pred == y_data[i]:
            correct += 1
    return correct / total if total > 0 else 0.0


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    instances: List[Tuple[np.ndarray, int]],
    shots: int,
    n_classes: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Tuple[float, float], Dict[int, float], List[list], float]:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    simulator = get_training_simulator()
    simulation_seconds = 0.0

    def objective(weight_vector) -> float:
        nonlocal simulation_seconds
        weight_map = dict(zip(sorted_weight_indices, weight_vector))
        loss = 0.0
        sim_start = time.perf_counter()
        for features, label in instances:
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.initialize(features, range(num_qubits))
            circuit = build_quantum_circuit(individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=False)
            qc.compose(circuit, inplace=True)
            qc.measure_all()
            counts = simulator.run(qc, shots=shots).result().get_counts()
            class_probs = {}
            for bitstring, count in counts.items():
                idx = int(bitstring.replace(" ", ""), 2)
                class_probs[idx] = class_probs.get(idx, 0)
            for c in range(n_classes):
                if c not in class_probs:
                    class_probs[c] = 0.0
            total = sum(class_probs.values())
            if total > 0:
                for c in class_probs:
                    class_probs[c] /= total
            eps = 1e-12
            loss -= np.log(class_probs.get(label, eps) + eps)
        simulation_seconds += time.perf_counter() - sim_start
        return loss / len(instances) if instances else 0.0

    if num_weights > 0:
        maxiter = max(50, num_weights * CONFIG.get("evaluation", {}).get("cobyla_maxiter_factor", 15))
        inherited = getattr(individual, "_seed_weights", None)
        if inherited is not None:
            x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
        else:
            x0 = np.random.uniform(0, 2 * np.pi, size=num_weights)
        result = minimize(objective, x0=x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        best_weights = dict(zip(sorted_weight_indices, (float(w) for w in result.x)))
    else:
        objective([])
        best_weights = {}

    val_acc = _qnn_accuracy(individual, num_qubits, X_val, y_val, best_weights, shots, n_classes, simulator)
    depth = build_quantum_circuit(individual, num_qubits, MANUAL_INPUT_VALUES, best_weights, measure=False).depth()

    return ((-val_acc, float(depth)), best_weights, [], simulation_seconds)


def validate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    shots: int,
    n_classes: int,
    seed_weights: dict | None = None,
) -> float:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)
    simulator = get_training_simulator()

    def objective(weight_vector) -> float:
        weight_map = dict(zip(sorted_weight_indices, weight_vector))
        loss = 0.0
        for i in range(len(X_val)):
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.initialize(X_val[i], range(num_qubits))
            circuit = build_quantum_circuit(individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=False)
            qc.compose(circuit, inplace=True)
            qc.measure_all()
            counts = simulator.run(qc, shots=shots).result().get_counts()
            class_probs = {}
            for bitstring, count in counts.items():
                idx = int(bitstring.replace(" ", ""), 2)
                class_probs[idx] = class_probs.get(idx, 0)
            for c in range(n_classes):
                if c not in class_probs:
                    class_probs[c] = 0.0
            total = sum(class_probs.values())
            if total > 0:
                for c in class_probs:
                    class_probs[c] /= total
            eps = 1e-12
            loss -= np.log(class_probs.get(y_val[i], eps) + eps)
        return loss / len(X_val) if len(X_val) > 0 else 0.0

    if num_weights > 0:
        if seed_weights is not None:
            x0 = np.array([seed_weights.get(i, 0.0) for i in sorted_weight_indices])
        else:
            x0 = np.random.RandomState(42).uniform(0, 2 * np.pi, size=num_weights)
        maxiter = max(50, num_weights * 10)
        result = minimize(objective, x0=x0, method="COBYLA", options={"maxiter": maxiter})
        best_weights = dict(zip(sorted_weight_indices, (float(w) for w in result.x)))
    else:
        best_weights = {}

    val_acc = _qnn_accuracy(individual, num_qubits, X_val, y_val, best_weights, shots, n_classes, simulator)
    return val_acc