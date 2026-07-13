from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    Optimize1qGatesDecomposition,
)
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from . import qnn_common as common

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

APPROACH: str = CONFIG["approach"]
RANDOM_SEED: int = int(CONFIG.get("random_seed", 42))

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
RANDOM_GENERATOR = np.random.default_rng(RANDOM_SEED)

_ENCODING_MODE: str = "clifford_angle" if APPROACH == "clifford" else "amplitude"


def get_training_simulator() -> AerSimulator:
    global _TRAINING_SIMULATOR
    if _TRAINING_SIMULATOR is None:
        method = "stabilizer" if APPROACH == "clifford" else "statevector"
        try:
            _TRAINING_SIMULATOR = AerSimulator(
                method=method,
                device=SIMULATOR_DEVICE if method == "statevector" else "CPU",
                max_parallel_threads=_STATEVECTOR_THREADS,
                seed_simulator=RANDOM_SEED,
            )
        except Exception as exc:
            logger.warning(
                "Failed to init %s simulator (%s); fallback CPU statevector",
                method,
                exc,
            )
            _TRAINING_SIMULATOR = AerSimulator(
                method="statevector", device="CPU", seed_simulator=RANDOM_SEED
            )
    return _TRAINING_SIMULATOR


def _prepare_input_circuit(
    num_qubits: int, encoded_sample: np.ndarray
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    if _ENCODING_MODE == "clifford_angle":
        buckets = encoded_sample.astype(int)
        for q in range(num_qubits):
            level = buckets[q]
            if level == 1:
                qc.h(q)
            elif level == 2:
                qc.x(q)
            elif level == 3:
                qc.x(q)
                qc.h(q)
        for q in range(num_qubits - 1):
            if buckets[q] >= 2 and buckets[q + 1] >= 2:
                qc.cx(q, q + 1)
    else:
        qc.initialize(encoded_sample, range(num_qubits))
    return qc


def describe_architecture(individual: EvolutionaryIndividual) -> dict:
    if APPROACH == "clifford":
        return {"blocks": q_strategy.describe_blocks(individual)}
    return {"param_genes": q_strategy.describe_param_genes(individual)}


def serialize_architecture(individual: EvolutionaryIndividual) -> list:
    return serialize_individual(individual)


def deserialize_architecture(data: list) -> EvolutionaryIndividual:
    return deserialize_individual(data)


_WEIGHT_CACHE: Dict[str, np.ndarray] = {}
_CIRCUIT_CACHE: Dict[str, object] = {}


def _individual_cache_key(individual: EvolutionaryIndividual) -> str:
    parts = []
    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            parts.append(f"P{gen[1]}")
        else:
            parts.append(str(gen[0]))
    return "_".join(parts)


_ROTATION_SIMPLIFY_PM = PassManager(
    [
        CommutativeCancellation(),
        Optimize1qGatesDecomposition(basis=["rx", "ry", "rz", "h", "s", "cx"]),
    ]
)


def _effective_depth(qc: QuantumCircuit) -> int:
    if APPROACH != "rotation":
        return qc.depth()
    try:
        simplified = _ROTATION_SIMPLIFY_PM.run(qc)
        return simplified.depth()
    except Exception:
        return qc.depth()


def _bitstring_class(bitstring: str, n_classes: int, num_qubits: int) -> int:
    idx = int(bitstring.replace(" ", ""), 2)
    total_states = 1 << num_qubits
    return idx * n_classes // total_states


def _classify_from_counts(
    counts: Dict[str, float], n_classes: int, num_qubits: int
) -> int:
    total = sum(counts.values())
    if total == 0:
        return int(RANDOM_GENERATOR.integers(0, n_classes))
    class_probs = {c: 0.0 for c in range(n_classes)}
    for bitstring, count in counts.items():
        c = _bitstring_class(bitstring, n_classes, num_qubits)
        class_probs[c] += count
    best_val = max(class_probs.values())
    candidates = [c for c, v in class_probs.items() if v == best_val]
    return int(RANDOM_GENERATOR.choice(candidates))


def _build_ansatz_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    sorted_weight_indices: List[int],
    weight_map: Dict[int, float],
) -> QuantumCircuit:
    if APPROACH == "rotation":
        key = _individual_cache_key(individual) + f"_{num_qubits}"
        cached = _CIRCUIT_CACHE.get(key)
        if cached is None:
            param_indices = sorted(set(sorted_weight_indices))
            param_objs = {i: Parameter(f"w_{i}") for i in param_indices}
            qc = QuantumCircuit(num_qubits)
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
            cached = (qc, param_indices, param_objs)
            _CIRCUIT_CACHE[key] = cached
            if len(_CIRCUIT_CACHE) > 200:
                _CIRCUIT_CACHE.pop(next(iter(_CIRCUIT_CACHE)))
        qc, param_indices, param_objs = cached
        assignments = {param_objs[i]: weight_map.get(i, 0.0) for i in param_indices}
        return qc.assign_parameters(assignments, inplace=False)

    quantized = tuple(round(weight_map.get(i, 0.0), 2) for i in sorted_weight_indices)
    key = _individual_cache_key(individual) + f"_{num_qubits}_" + str(quantized)
    cached = _CIRCUIT_CACHE.get(key)
    if cached is not None:
        return cached
    qc = build_quantum_circuit(
        individual, num_qubits, MANUAL_INPUT_VALUES, weight_map, measure=False
    )
    _CIRCUIT_CACHE[key] = qc
    if len(_CIRCUIT_CACHE) > 500:
        _CIRCUIT_CACHE.pop(next(iter(_CIRCUIT_CACHE)))
    return qc


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
    total = len(y_data)
    if total == 0:
        return 0.0
    sorted_weight_indices = sorted(weight_map.keys())
    ansatz = _build_ansatz_circuit(
        individual, num_qubits, sorted_weight_indices, weight_map
    )

    circuits = []
    for i in range(total):
        qc = _prepare_input_circuit(num_qubits, X_data[i])
        qc.compose(ansatz, inplace=True)
        qc.measure_all()
        circuits.append(qc)

    results = simulator.run(circuits, shots=shots).result()
    correct = 0
    for i in range(total):
        counts = results.get_counts(i)
        pred = _classify_from_counts(counts, n_classes, num_qubits)
        if pred == y_data[i]:
            correct += 1
    return correct / total


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
        ansatz = _build_ansatz_circuit(
            individual, num_qubits, sorted_weight_indices, weight_map
        )

        circuits = []
        for features, label in instances:
            qc = _prepare_input_circuit(num_qubits, features)
            qc.compose(ansatz, inplace=True)
            qc.measure_all()
            circuits.append(qc)

        sim_start = time.perf_counter()
        results = simulator.run(circuits, shots=shots).result()
        simulation_seconds += time.perf_counter() - sim_start

        loss = 0.0
        eps = 1e-12
        for idx, (_, label) in enumerate(instances):
            counts = results.get_counts(idx)
            class_probs = {c: 0.0 for c in range(n_classes)}
            for bitstring, count in counts.items():
                c = _bitstring_class(bitstring, n_classes, num_qubits)
                class_probs[c] += count
            total = sum(class_probs.values())
            if total > 0:
                for c in class_probs:
                    class_probs[c] /= total
            loss -= np.log(class_probs.get(label, eps) + eps)
        return loss / len(instances) if instances else 0.0

    if num_weights > 0:
        maxiter = max(
            50,
            num_weights * CONFIG.get("evaluation", {}).get("cobyla_maxiter_factor", 15),
        )
        inherited = getattr(individual, "_seed_weights", None)

        if APPROACH == "rotation":
            if inherited is not None:
                x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
            else:
                x0 = RANDOM_GENERATOR.uniform(0, 2 * np.pi, size=num_weights)
            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 1.5},
            )
            best_weights = dict(
                zip(sorted_weight_indices, (float(w) for w in result.x))
            )
        else:
            if inherited is not None:
                x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
            else:
                x0 = RANDOM_GENERATOR.uniform(0, 5, size=num_weights)

            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 1.0},
            )
            best_weights = dict(
                zip(sorted_weight_indices, (float(w) for w in result.x))
            )
    else:
        objective([])
        best_weights = {}

    val_acc = _qnn_accuracy(
        individual, num_qubits, X_val, y_val, best_weights, shots, n_classes, simulator
    )
    depth = _effective_depth(
        build_quantum_circuit(
            individual, num_qubits, MANUAL_INPUT_VALUES, best_weights, measure=False
        )
    )

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
        ansatz = _build_ansatz_circuit(
            individual, num_qubits, sorted_weight_indices, weight_map
        )

        circuits = []
        for i in range(len(X_val)):
            qc = _prepare_input_circuit(num_qubits, X_val[i])
            qc.compose(ansatz, inplace=True)
            qc.measure_all()
            circuits.append(qc)

        results = simulator.run(circuits, shots=shots).result() if circuits else None

        loss = 0.0
        eps = 1e-12
        for i in range(len(X_val)):
            counts = results.get_counts(i)
            class_probs = {c: 0.0 for c in range(n_classes)}
            for bitstring, count in counts.items():
                c = _bitstring_class(bitstring, n_classes, num_qubits)
                class_probs[c] += count
            total = sum(class_probs.values())
            if total > 0:
                for c in class_probs:
                    class_probs[c] /= total
            loss -= np.log(class_probs.get(y_val[i], eps) + eps)
        return loss / len(X_val) if len(X_val) > 0 else 0.0

    if num_weights > 0:
        maxiter = max(50, num_weights * 10)
        if APPROACH == "rotation":
            if seed_weights is not None:
                x0 = np.array([seed_weights.get(i, 0.0) for i in sorted_weight_indices])
            else:
                x0 = RANDOM_GENERATOR.uniform(0, 2 * np.pi, size=num_weights)

            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 1.5},
            )
            best_weights = dict(
                zip(sorted_weight_indices, (float(w) for w in result.x))
            )
        else:
            if seed_weights is not None:
                x0 = np.array([seed_weights.get(i, 0.0) for i in sorted_weight_indices])
            else:
                x0 = RANDOM_GENERATOR.uniform(0, 5, size=num_weights)

            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 1.0},
            )
            best_weights = dict(
                zip(sorted_weight_indices, (float(w) for w in result.x))
            )
    else:
        best_weights = {}

    val_acc = _qnn_accuracy(
        individual, num_qubits, X_val, y_val, best_weights, shots, n_classes, simulator
    )
    return val_acc


def update_hof(block_gates: list) -> None:
    if APPROACH != "clifford":
        return
    if block_gates not in q_strategy.BLOCK_HOF:
        q_strategy.BLOCK_HOF.append(block_gates)
    if len(q_strategy.BLOCK_HOF) > 100:
        q_strategy.BLOCK_HOF.pop(0)
