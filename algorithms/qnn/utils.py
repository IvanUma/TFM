from __future__ import annotations

import functools
import importlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    Optimize1qGatesDecomposition,
)
from qiskit.quantum_info import Statevector, StabilizerState
from scipy.optimize import minimize

from . import qnn_common as common

logger = logging.getLogger(__name__)

_CONFIG = None
CONFIG = None
CONFIG_PATH = None
APPROACH = None
RANDOM_SEED = None
DATASET_NAME = None
TEST_SPLIT = None
VAL_SPLIT = None
ENABLE_INPUT_PARAMS = None
PARAM_BLOCK_PROB = None
MANUAL_INPUT_VALUES = None
NUM_PARAMS = None
SIMULATOR_DEVICE = None
_ENCODING_MODE = None
_RANDOM_GENERATOR = None
_STATEVECTOR_THREADS = None

q_strategy = None
EvolutionaryIndividual = None
build_quantum_circuit = None
get_param_indices = None
generate_guided_individual = None
mut_quantum_circuit = None
serialize_individual = None
deserialize_individual = None

cx_quantum_circuit = common.cx_quantum_circuit


def _deep_tuple(obj):
    if isinstance(obj, list):
        return tuple(_deep_tuple(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(_deep_tuple(item) for item in obj)
    return obj


_ROTATION_SIMPLIFY_PM = PassManager(
    [
        CommutativeCancellation(),
        Optimize1qGatesDecomposition(basis=["rx", "ry", "rz", "h", "s", "cx"]),
    ]
)

_CLIFFORD_SIMPLIFY_PM = PassManager(
    [
        CommutativeCancellation(basis_gates=["h", "s", "cx"]),
    ]
)


def _resolve_device(requested_device: str) -> str:
    if requested_device == "GPU":
        logger.warning(
            "El cálculo ahora es exacto vía Statevector/StabilizerState (numpy, solo CPU); "
            "se ignora la solicitud de GPU."
        )
    return "CPU"


def init_config(dataset_name: str, approach: str | None = None) -> None:
    global _CONFIG, CONFIG, CONFIG_PATH, APPROACH, RANDOM_SEED, DATASET_NAME
    global TEST_SPLIT, VAL_SPLIT, ENABLE_INPUT_PARAMS, PARAM_BLOCK_PROB
    global MANUAL_INPUT_VALUES, NUM_PARAMS, SIMULATOR_DEVICE
    global _ENCODING_MODE, _RANDOM_GENERATOR, _STATEVECTOR_THREADS
    global q_strategy, EvolutionaryIndividual, build_quantum_circuit
    global get_param_indices, generate_guided_individual, mut_quantum_circuit
    global serialize_individual, deserialize_individual

    config_dir = Path(__file__).parent / "configs"
    config_path = config_dir / f"{dataset_name}.json"
    if not config_path.exists():
        config_path = Path(__file__).with_name("config.json")
    CONFIG_PATH = config_path

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
    CONFIG = _CONFIG

    APPROACH = _CONFIG["approach"]
    if approach is not None:
        APPROACH = approach
    RANDOM_SEED = int(_CONFIG.get("random_seed", 42))
    _RANDOM_GENERATOR = np.random.default_rng(RANDOM_SEED)

    _qnn_cfg = _CONFIG.get("qnn", {})
    DATASET_NAME = _qnn_cfg.get("dataset", dataset_name)
    TEST_SPLIT = float(_qnn_cfg.get("test_split", 0.2))
    VAL_SPLIT = float(_qnn_cfg.get("val_split", 0.2))

    _encoding_cfg = _CONFIG.get("encoding", {})
    ENABLE_INPUT_PARAMS = _encoding_cfg.get("enable_input_params", False)
    PARAM_BLOCK_PROB = _encoding_cfg.get("param_block_prob", 0.15)

    _manual_input_values = _encoding_cfg.get("manual_input_values", [1.0])
    if not isinstance(_manual_input_values, list) or len(_manual_input_values) == 0:
        raise ValueError(
            "encoding.manual_input_values must be a non-empty list of numbers"
        )
    MANUAL_INPUT_VALUES = [float(v) for v in _manual_input_values]
    NUM_PARAMS = len(MANUAL_INPUT_VALUES)

    q_strategy = importlib.import_module(f".qnn_{APPROACH}", package=__package__)
    EvolutionaryIndividual = q_strategy.EvolutionaryIndividual
    build_quantum_circuit = q_strategy.build_quantum_circuit
    get_param_indices = q_strategy.get_param_indices
    generate_guided_individual = q_strategy.generate_guided_individual
    mut_quantum_circuit = q_strategy.mut_quantum_circuit
    serialize_individual = q_strategy.serialize_individual
    deserialize_individual = q_strategy.deserialize_individual

    _execution_cfg = _CONFIG.get("execution", {})
    _STATEVECTOR_THREADS = _execution_cfg.get("statevector_max_parallel_threads", 0)
    _REQUESTED_DEVICE = _execution_cfg.get("device", "auto")
    SIMULATOR_DEVICE = _resolve_device(_REQUESTED_DEVICE)

    _ENCODING_MODE = "clifford_angle" if APPROACH == "clifford" else "amplitude"


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


def _effective_depth(qc: QuantumCircuit) -> int:
    if APPROACH == "rotation":
        pm = _ROTATION_SIMPLIFY_PM
    elif APPROACH == "clifford":
        pm = _CLIFFORD_SIMPLIFY_PM
    else:
        return qc.depth()
    try:
        simplified = pm.run(qc)
        return simplified.depth()
    except Exception:
        return qc.depth()


def _expectation_z_from_probs(probs: np.ndarray, num_qubits: int) -> np.ndarray:
    exp_z = np.zeros(num_qubits)
    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        bits = format(idx, f"0{num_qubits}b")
        for q in range(num_qubits):
            exp_z[q] += p * (1 if bits[q] == "0" else -1)
    return exp_z


def _exact_probabilities(qc: QuantumCircuit) -> np.ndarray:
    if APPROACH == "clifford":
        return StabilizerState(qc).probabilities()
    return Statevector.from_instruction(qc).probabilities()


def _exact_expectation_z(qc: QuantumCircuit, num_qubits: int) -> np.ndarray:
    probs = _exact_probabilities(qc)
    return _expectation_z_from_probs(probs, num_qubits)


def _exp_z_to_class_probs(
    exp_z: np.ndarray, n_classes: int, beta: float = 5.0
) -> np.ndarray:
    if len(exp_z) == 0:
        return np.ones(n_classes) / n_classes

    scaled_z = exp_z * beta

    if n_classes == 2:
        prob_1 = 1.0 / (1.0 + np.exp(-scaled_z[0]))
        return np.array([1.0 - prob_1, prob_1])

    k = min(n_classes, len(scaled_z))
    scores = scaled_z[:k]
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probs = np.zeros(n_classes)
    probs[:k] = exp_scores / (np.sum(exp_scores) + 1e-12)
    return probs


@functools.lru_cache(maxsize=500)
def _get_rotation_ansatz(
    individual_tuple: tuple, num_qubits: int, sorted_indices_tuple: tuple
):
    param_indices = list(sorted_indices_tuple)
    param_objs = {i: Parameter(f"w_{i}") for i in param_indices}
    qc = QuantumCircuit(num_qubits)
    for gen in individual_tuple:
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
    return qc, param_objs


@functools.lru_cache(maxsize=500)
def _get_clifford_ansatz(
    individual_tuple: tuple, num_qubits: int, quantized_weights: tuple
):
    weight_map = dict(quantized_weights)
    return build_quantum_circuit(
        list(individual_tuple),
        num_qubits,
        MANUAL_INPUT_VALUES,
        weight_map,
        measure=False,
    )


def _build_ansatz_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    sorted_weight_indices: List[int],
    weight_map: Dict[int, float],
) -> QuantumCircuit:
    individual_tuple = _deep_tuple(individual)

    if APPROACH == "rotation":
        indices_tuple = tuple(sorted_weight_indices)
        qc, param_objs = _get_rotation_ansatz(
            individual_tuple, num_qubits, indices_tuple
        )
        assignments = {
            param_objs[i]: weight_map.get(i, 0.0) for i in sorted_weight_indices
        }
        return qc.assign_parameters(assignments, inplace=False)

    quantized_weights = tuple(
        (i, round(weight_map.get(i, 0.0), 2)) for i in sorted_weight_indices
    )
    return _get_clifford_ansatz(individual_tuple, num_qubits, quantized_weights)


def _qnn_metrics(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    X_data: np.ndarray,
    y_data: np.ndarray,
    weight_map: Dict[int, float],
    n_classes: int,
    use_logistic: bool = True,
) -> Tuple[float, float]:
    total = len(y_data)
    if total == 0:
        return 0.0, 0.0

    _, weight_indices = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices)
    ansatz = _build_ansatz_circuit(
        individual, num_qubits, sorted_weight_indices, weight_map
    )

    exp_z_list = []
    for i in range(total):
        qc = _prepare_input_circuit(num_qubits, X_data[i])
        qc.compose(ansatz, inplace=True)
        exp_z_list.append(_exact_expectation_z(qc, num_qubits))
    X_exp_z = np.array(exp_z_list)

    stored_clf = getattr(individual, "_readout_clf", None)
    if use_logistic:
        clf = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=200,
            random_state=RANDOM_SEED,
        )
        n_splits = 3
        probs = cross_val_predict(
            clf, X_exp_z, y_data, cv=n_splits, method="predict_proba"
        )
        clf.fit(X_exp_z, y_data)
        individual._readout_clf = clf
    elif stored_clf is not None:
        probs = stored_clf.predict_proba(X_exp_z)
    else:
        probs = np.array([_exp_z_to_class_probs(ez, n_classes) for ez in exp_z_list])
    correct = int(np.sum(np.argmax(probs, axis=1) == y_data))
    soft_score = float(np.mean(probs[np.arange(total), y_data]))
    return correct / total, soft_score


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    instances: List[Tuple[np.ndarray, int]],
    n_classes: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Tuple[float, float], Dict[int, float], List[list], float]:
    _, weight_indices_set = get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    inherited = getattr(individual, "_seed_weights", None)
    if not inherited:
        inherited = getattr(individual, "stored_thetas", None)

    best_weights = {}

    simulation_seconds = 0.0
    sim_start = time.perf_counter()

    if APPROACH == "rotation" and num_weights > 0:
        input_circuits = [
            _prepare_input_circuit(num_qubits, features) for features, _ in instances
        ]

        def objective(weight_vector) -> float:
            weight_map = dict(zip(sorted_weight_indices, weight_vector))
            ansatz = _build_ansatz_circuit(
                individual, num_qubits, sorted_weight_indices, weight_map
            )

            eps = 1e-12
            loss = 0.0
            for base_qc, (_, label) in zip(input_circuits, instances):
                qc = base_qc.copy()
                qc.compose(ansatz, inplace=True)
                exp_z = _exact_expectation_z(qc, num_qubits)
                class_probs = _exp_z_to_class_probs(exp_z, n_classes)
                loss -= np.log(class_probs[label] + eps)
            return loss / len(instances) if instances else 0.0

        maxiter = max(
            50,
            num_weights
            * _CONFIG.get("evaluation", {}).get("cobyla_maxiter_factor", 15),
        )

        if inherited:
            x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
            starts = [x0]
        else:
            starts = [
                _RANDOM_GENERATOR.uniform(-np.pi, np.pi, size=num_weights),
                _RANDOM_GENERATOR.uniform(-np.pi, np.pi, size=num_weights),
                _RANDOM_GENERATOR.uniform(-np.pi, np.pi, size=num_weights),
                _RANDOM_GENERATOR.uniform(-np.pi, np.pi, size=num_weights),
            ]
            maxiter = max(30, maxiter // 3)

        sim_start = time.perf_counter()
        best_result = None
        for x0 in starts:
            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 1.5},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        best_weights = dict(
            zip(sorted_weight_indices, (float(w) for w in best_result.x))
        )
    else:
        if inherited:
            best_weights = inherited

    X_train_batch = np.array([inst[0] for inst in instances])
    y_train_batch = np.array([inst[1] for inst in instances])

    if APPROACH == "rotation" and num_weights > 0:
        sim_end = time.perf_counter()
        simulation_seconds += sim_end - sim_start
        sim_start = sim_end
    _, soft_score = _qnn_metrics(
        individual,
        num_qubits,
        X_train_batch,
        y_train_batch,
        best_weights,
        n_classes,
    )
    simulation_seconds += time.perf_counter() - sim_start

    depth = _effective_depth(
        build_quantum_circuit(
            individual, num_qubits, MANUAL_INPUT_VALUES, best_weights, measure=False
        )
    )

    depth_obj = float(depth)

    readout_data = getattr(individual, "_readout_clf", None)
    return ((-soft_score, depth_obj), best_weights, readout_data, simulation_seconds)


def validate_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed_weights: dict | None = None,
) -> Tuple[float, float]:
    best_weights = seed_weights if seed_weights is not None else {}

    val_acc, soft_score = _qnn_metrics(
        individual,
        num_qubits,
        X_val,
        y_val,
        best_weights,
        n_classes,
        use_logistic=False,
    )
    return val_acc, soft_score


def update_hof(block_gates: list) -> None:
    if APPROACH != "clifford":
        return
    if block_gates not in q_strategy.BLOCK_HOF:
        q_strategy.BLOCK_HOF.append(block_gates)
    if len(q_strategy.BLOCK_HOF) > 100:
        q_strategy.BLOCK_HOF.pop(0)
