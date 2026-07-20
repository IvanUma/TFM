from __future__ import annotations

import functools
import logging
import time
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
from qiskit.quantum_info import Statevector, StabilizerState, Operator
from scipy.optimize import minimize

from . import qnn_common as common
from . import config as _qnn_config

cx_quantum_circuit = _qnn_config.cx_quantum_circuit
from .constants import (
    SIGMOID_BETA,
    READOUT_C,
    READOUT_MAX_ITER,
    READOUT_CV_SPLITS,
    COBYLA_RHOBEG,
    COBYLA_MIN_MAXITER,
    COBYLA_MIN_MAXITER_NO_INHERIT,
    COBYLA_RANDOM_STARTS,
    COBYLA_MAXITER_FACTOR_DEFAULT,
    NUMERICAL_EPS,
    BLOCK_HOF_MAXLEN,
)

logger = logging.getLogger(__name__)


def _deep_tuple(obj):
    if isinstance(obj, list):
        return tuple(_deep_tuple(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(_deep_tuple(item) for item in obj)
    try:
        hash(obj)
        return obj
    except TypeError:
        return str(obj)


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


def _prepare_input_circuit(
    num_qubits: int, encoded_sample: np.ndarray
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    if _qnn_config._ENCODING_MODE == "clifford_angle":
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
    if _qnn_config.APPROACH == "clifford":
        return {"blocks": _qnn_config.q_strategy.describe_blocks(individual)}
    return {"param_genes": _qnn_config.q_strategy.describe_param_genes(individual)}


def serialize_architecture(individual: EvolutionaryIndividual) -> list:
    return _qnn_config.serialize_individual(individual)


def deserialize_architecture(data: list) -> EvolutionaryIndividual:
    return _qnn_config.deserialize_individual(data)


def _effective_depth(qc: QuantumCircuit) -> int:
    if _qnn_config.APPROACH == "rotation":
        pm = _ROTATION_SIMPLIFY_PM
    elif _qnn_config.APPROACH == "clifford":
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
    if _qnn_config.APPROACH == "clifford":
        return StabilizerState(qc).probabilities()
    return Statevector.from_instruction(qc).probabilities()


def _exact_expectation_z(qc: QuantumCircuit, num_qubits: int) -> np.ndarray:
    probs = _exact_probabilities(qc)
    return _expectation_z_from_probs(probs, num_qubits)


def _exp_z_to_class_probs(
    exp_z: np.ndarray, n_classes: int, beta: float = SIGMOID_BETA
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


_SIGN_MATRIX_CACHE: Dict[int, np.ndarray] = {}


def _get_sign_matrix(num_qubits: int) -> np.ndarray:
    if num_qubits not in _SIGN_MATRIX_CACHE:
        dim = 2**num_qubits
        signs = np.empty((dim, num_qubits))
        for idx in range(dim):
            bits = format(idx, f"0{num_qubits}b")
            for q in range(num_qubits):
                signs[idx, q] = 1.0 if bits[q] == "0" else -1.0
        _SIGN_MATRIX_CACHE[num_qubits] = signs
    return _SIGN_MATRIX_CACHE[num_qubits]


@functools.lru_cache(maxsize=4096)
def _get_input_statevector(num_qubits: int, encoded_tuple: tuple) -> np.ndarray:
    qc = _prepare_input_circuit(num_qubits, np.array(encoded_tuple))
    return Statevector.from_instruction(qc).data


_CLIFFORD_INPUT_CACHE: Dict[Tuple, np.ndarray] = {}


def _get_clifford_input_matrix(X_data: np.ndarray, num_qubits: int) -> np.ndarray:
    key = (num_qubits, X_data.shape[0])
    if key not in _CLIFFORD_INPUT_CACHE:
        _CLIFFORD_INPUT_CACHE[key] = np.array(
            [
                Statevector.from_instruction(_prepare_input_circuit(num_qubits, x)).data
                for x in X_data
            ]
        )
    return _CLIFFORD_INPUT_CACHE[key]


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
                    _qnn_config.MANUAL_INPUT_VALUES[
                        p_idx % len(_qnn_config.MANUAL_INPUT_VALUES)
                    ]
                    if _qnn_config.MANUAL_INPUT_VALUES
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
    return _qnn_config.build_quantum_circuit(
        list(individual_tuple),
        num_qubits,
        _qnn_config.MANUAL_INPUT_VALUES,
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

    if _qnn_config.APPROACH == "rotation":
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

    _, weight_indices = _qnn_config.get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices)
    ansatz = _build_ansatz_circuit(
        individual, num_qubits, sorted_weight_indices, weight_map
    )

    if _qnn_config.APPROACH == "rotation":
        input_state_matrix = np.array(
            [_get_input_statevector(num_qubits, tuple(x)) for x in X_data]
        )
    else:
        input_state_matrix = _get_clifford_input_matrix(X_data, num_qubits)
    sign_matrix = _get_sign_matrix(num_qubits)
    U = Operator(ansatz).data
    evolved = input_state_matrix @ U.T
    probs_sv = np.abs(evolved) ** 2
    X_exp_z = probs_sv @ sign_matrix

    stored_clf = getattr(individual, "_readout_clf", None)
    if use_logistic:
        clf = LogisticRegression(
            C=READOUT_C,
            solver="lbfgs",
            max_iter=READOUT_MAX_ITER,
            random_state=_qnn_config.RANDOM_SEED,
        )
        probs = cross_val_predict(
            clf, X_exp_z, y_data, cv=READOUT_CV_SPLITS, method="predict_proba"
        )
        clf.fit(X_exp_z, y_data)
        individual._readout_clf = clf
    elif stored_clf is not None:
        probs = stored_clf.predict_proba(X_exp_z)
    else:
        probs = np.array([_exp_z_to_class_probs(ez, n_classes) for ez in X_exp_z])
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
    _, weight_indices_set = _qnn_config.get_param_indices(individual)
    sorted_weight_indices = sorted(weight_indices_set)
    num_weights = len(sorted_weight_indices)

    inherited = getattr(individual, "_seed_weights", None)
    if not inherited:
        inherited = getattr(individual, "stored_thetas", None)

    best_weights = {}

    simulation_seconds = 0.0
    sim_start = time.perf_counter()

    if _qnn_config.APPROACH == "rotation" and num_weights > 0:
        input_state_matrix = np.array(
            [
                _get_input_statevector(num_qubits, tuple(features))
                for features, _ in instances
            ]
        )
        labels_arr = np.array([label for _, label in instances])
        sign_matrix = _get_sign_matrix(num_qubits)

        def objective(weight_vector) -> float:
            weight_map = dict(zip(sorted_weight_indices, weight_vector))
            ansatz = _build_ansatz_circuit(
                individual, num_qubits, sorted_weight_indices, weight_map
            )
            U = Operator(ansatz).data

            evolved = input_state_matrix @ U.T
            probs = np.abs(evolved) ** 2
            exp_z = probs @ sign_matrix

            loss = 0.0
            for ez, label in zip(exp_z, labels_arr):
                class_probs = _exp_z_to_class_probs(ez, n_classes)
                loss -= np.log(class_probs[label] + NUMERICAL_EPS)
            return loss / len(instances) if instances else 0.0

        cobyla_factor = _qnn_config._CONFIG.get("evaluation", {}).get(
            "cobyla_maxiter_factor", COBYLA_MAXITER_FACTOR_DEFAULT
        )
        maxiter = max(
            COBYLA_MIN_MAXITER,
            num_weights * cobyla_factor,
        )

        if inherited:
            x0 = np.array([inherited.get(i, 0.0) for i in sorted_weight_indices])
            starts = [x0]
        else:
            starts = [
                _qnn_config._RANDOM_GENERATOR.uniform(-np.pi, np.pi, size=num_weights)
                for _ in range(COBYLA_RANDOM_STARTS)
            ]
            maxiter = max(COBYLA_MIN_MAXITER_NO_INHERIT, maxiter // 3)

        sim_start = time.perf_counter()
        best_result = None
        for x0 in starts:
            result = minimize(
                objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": COBYLA_RHOBEG},
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

    if _qnn_config.APPROACH == "rotation" and num_weights > 0:
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
        _qnn_config.build_quantum_circuit(
            individual,
            num_qubits,
            _qnn_config.MANUAL_INPUT_VALUES,
            best_weights,
            measure=False,
        )
    )

    depth_power = float(
        _qnn_config._CONFIG.get("evolution", {}).get("depth_power", 1.2)
    )
    depth_obj = float(depth) ** depth_power

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
    if seed_weights is None:
        seed_weights = getattr(individual, "stored_thetas", {})
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
    if _qnn_config.APPROACH != "clifford":
        return
    if block_gates not in _qnn_config.q_strategy.BLOCK_HOF:
        _qnn_config.q_strategy.BLOCK_HOF.append(block_gates)
    if len(_qnn_config.q_strategy.BLOCK_HOF) > BLOCK_HOF_MAXLEN:
        _qnn_config.q_strategy.BLOCK_HOF.pop(0)
