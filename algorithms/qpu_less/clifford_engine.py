import threading
from typing import Tuple, List

from .pauli_algebra import (
    PauliTerm,
    PauliSum,
    evaluate_on_zero_state,
    multiply_terms,
    simplify_sum,
)

CLIFFORD_CACHE = {}
CACHE_LOCK = threading.Lock()


def evolve_operator_backwards(graph, individual) -> float:
    H_P = PauliSum()
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        H_P.append(PauliTerm(0.5 * w, {}))
        H_P.append(PauliTerm(-0.5 * w, {u: "Z", v: "Z"}))

    current_operators = simplify_sum(H_P)

    for gate in reversed(individual):
        gate_type = gate[0]
        next_sum = PauliSum()

        if gate_type == "H":
            q = gate[1]
            for term in current_operators.terms:
                ops = term.operators.copy()
                op = ops.get(q, "I")
                coef = term.coefficient
                if op == "X":
                    ops[q] = "Z"
                elif op == "Z":
                    ops[q] = "X"
                elif op == "Y":
                    coef *= -1
                next_sum.append(PauliTerm(coef, ops))

        elif gate_type == "S":
            q = gate[1]
            for term in current_operators.terms:
                ops = term.operators.copy()
                op = ops.get(q, "I")
                coef = term.coefficient
                if op == "X":
                    ops[q] = "Y"
                elif op == "Y":
                    ops[q] = "X"
                    coef *= -1
                next_sum.append(PauliTerm(coef, ops))

        elif gate_type == "CX":
            c, t = gate[1], gate[2]
            for term in current_operators.terms:
                evolved_term = PauliTerm(term.coefficient, {})
                for q_i, op_i in term.operators.items():
                    transformed_single = PauliTerm(1.0, {q_i: op_i})

                    if q_i == c:
                        if op_i == "X":
                            transformed_single = PauliTerm(1.0, {c: "X", t: "X"})
                        elif op_i == "Y":
                            transformed_single = PauliTerm(1.0, {c: "Y", t: "X"})
                        elif op_i == "Z":
                            transformed_single = PauliTerm(1.0, {c: "Z"})
                    elif q_i == t:
                        if op_i == "X":
                            transformed_single = PauliTerm(1.0, {t: "X"})
                        elif op_i == "Y":
                            transformed_single = PauliTerm(1.0, {c: "Z", t: "Y"})
                        elif op_i == "Z":
                            transformed_single = PauliTerm(1.0, {c: "Z", t: "Z"})

                    evolved_term = multiply_terms(evolved_term, transformed_single)

                next_sum.append(evolved_term)

        else:
            raise ValueError(f"Unsupported Clifford gate: {gate}")

        current_operators = simplify_sum(next_sum)

    return evaluate_on_zero_state(current_operators)


def evaluate_individual_algebraic(individual, num_qubits, graph) -> Tuple[float, float]:
    ind_key = tuple(individual)

    with CACHE_LOCK:
        if ind_key in CLIFFORD_CACHE:
            expected_cut = CLIFFORD_CACHE[ind_key]
        else:
            expected_cut = evolve_operator_backwards(graph, individual)
            CLIFFORD_CACHE[ind_key] = expected_cut

    cut_value = abs(expected_cut)

    circuit_depth = compute_depth(individual, num_qubits)

    return -cut_value, circuit_depth


def compute_depth(circuit, num_qubits):
    layer = [0] * num_qubits

    for gate in circuit:
        gate_type = gate[0]
        if gate_type in ("H", "S"):
            q = gate[1]
            layer[q] += 1

        elif gate_type == "CX":
            c = gate[1]
            t = gate[2]

            depth = max(layer[c], layer[t]) + 1

            layer[c] = depth
            layer[t] = depth

    return max(layer) if layer else 0
