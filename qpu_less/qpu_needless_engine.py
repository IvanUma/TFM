import math
from pauli_algebra import PauliTerm, PauliSum, commutator, multiply_terms, simplify_sum


# Lemma 3.2
def evolve_sum_under_problem_edge(
    p_sum: PauliSum, u: int, v: int, gamma: float, weight: float
) -> PauliSum:
    # En la Ecuación 12, el circuito QAOA aplica e^{-i gamma H_p}.
    # Por tanto, el parámetro de rotación es θ = -gamma * weight
    theta = -gamma * weight

    # B es el operador de la interacción (Z_u ⊗ Z_v) que cumple B^2 = I (condición del Lema 3.2)
    B = PauliTerm(1.0, {u: "Z", v: "Z"})
    result_sum = PauliSum()

    for A in p_sum.terms:
        # Calculamos el conmutador exacto [A, B]
        comm = commutator(A, B)

        # Si [A,B] = 0 (conmutan), el operador A no sufre cambios.
        if len(comm.terms) == 0:
            result_sum.append(A)
            continue

        comm_term = comm.terms[0]

        # Componente 1 del Lema 3.2: El operador original A
        result_sum.append(A)

        # Componente 2 del Lema 3.2: -i * cos(θ) * sin(θ) * [A, B]
        c1 = -1j * math.cos(theta) * math.sin(theta) * comm_term.coefficient
        result_sum.append(PauliTerm(c1, comm_term.operators))

        # Componente 3 del Lema 3.2: sin^2(θ) * B * [A, B]
        b_comm = multiply_terms(B, comm_term)
        c2 = (math.sin(theta) ** 2) * b_comm.coefficient
        result_sum.append(PauliTerm(c2, b_comm.operators))

    return simplify_sum(result_sum)


# Proposition 3.5
def evolve_sum_under_mixer(p_sum: PauliSum, beta: float) -> PauliSum:
    current_sum = p_sum
    active_qubits = set()
    for t in current_sum.terms:
        active_qubits.update(t.operators.keys())

    for q in active_qubits:
        next_sum = PauliSum()
        for term in current_sum.terms:
            op = term.operators.get(q, "I")

            if op == "I" or op == "X":
                # El conmutador [A_j, X_j] es cero. El término queda intacto.
                next_sum.append(term)

            elif op == "Z":
                # A_j = Z
                ops_z = term.operators.copy()
                ops_y = term.operators.copy()
                ops_y[q] = "Y"
                next_sum.append(PauliTerm(term.coefficient * math.cos(2 * beta), ops_z))
                next_sum.append(PauliTerm(term.coefficient * math.sin(2 * beta), ops_y))

            elif op == "Y":
                # A_j = Y
                ops_y = term.operators.copy()
                ops_z = term.operators.copy()
                ops_z[q] = "Z"
                next_sum.append(PauliTerm(term.coefficient * math.cos(2 * beta), ops_y))
                next_sum.append(
                    PauliTerm(-term.coefficient * math.sin(2 * beta), ops_z)
                )

        current_sum = simplify_sum(next_sum)

    return current_sum


# Lemma 3.8
def evaluate_on_plus_state(p_sum: PauliSum) -> float:
    total_val = 0.0
    for term in p_sum.terms:
        if any(op in ["Y", "Z"] for op in term.operators.values()):
            continue

        total_val += term.coefficient.real

    return total_val
