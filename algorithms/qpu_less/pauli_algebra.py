from typing import Dict, List


class PauliTerm:
    def __init__(self, coefficient: complex, operators: Dict[int, str]):
        self.coefficient = complex(coefficient)
        self.operators = {qubit: op for qubit, op in operators.items() if op != "I"}

    def __repr__(self):
        if len(self.operators) == 0:
            return f"({self.coefficient.real:+.4f}{self.coefficient.imag:+.4f}j) * I"
        op_str = " ⊗ ".join([f"{op}_{q}" for q, op in sorted(self.operators.items())])
        return (
            f"({self.coefficient.real:+.4f}{self.coefficient.imag:+.4f}j) * ({op_str})"
        )


class PauliSum:
    def __init__(self, terms: List[PauliTerm] = None):
        self.terms = terms if terms is not None else []

    def append(self, term: PauliTerm):
        self.terms.append(term)

    def __repr__(self):
        if not self.terms:
            return "0.0"
        return "\n".join([str(t) for t in self.terms])


PAULI_MULT = {
    ("I", "I"): (1.0, "I"),
    ("I", "X"): (1.0, "X"),
    ("I", "Y"): (1.0, "Y"),
    ("I", "Z"): (1.0, "Z"),
    ("X", "I"): (1.0, "X"),
    ("X", "X"): (1.0, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1.0, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1.0, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1.0, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1.0, "I"),
}


def multiply_terms(t1: PauliTerm, t2: PauliTerm) -> PauliTerm:
    all_qubits = set(t1.operators.keys()) | set(t2.operators.keys())
    new_ops = {}
    new_coeff = t1.coefficient * t2.coefficient

    for q in all_qubits:
        op1 = t1.operators.get(q, "I")
        op2 = t2.operators.get(q, "I")
        phase, res_op = PAULI_MULT[(op1, op2)]
        new_coeff *= phase
        if res_op != "I":
            new_ops[q] = res_op

    return PauliTerm(new_coeff, new_ops)


def commutator(t1: PauliTerm, t2: PauliTerm) -> PauliSum:
    ab = multiply_terms(t1, t2)
    ba = multiply_terms(t2, t1)
    res_coeff = ab.coefficient - ba.coefficient

    if abs(res_coeff) < 1e-9:
        return PauliSum([])
    return PauliSum([PauliTerm(res_coeff, ab.operators)])


def simplify_sum(p_sum: PauliSum) -> PauliSum:
    aggregated = {}
    for term in p_sum.terms:
        op_key = ",".join([f"{q}:{op}" for q, op in sorted(term.operators.items())])
        aggregated[op_key] = aggregated.get(op_key, 0.0 + 0j) + term.coefficient

    new_terms = []
    for op_key, coeff in aggregated.items():
        if abs(coeff) > 1e-9:
            ops = {}
            if op_key:
                for item in op_key.split(","):
                    q_str, op = item.split(":")
                    ops[int(q_str)] = op
            new_terms.append(PauliTerm(coeff, ops))

    return PauliSum(new_terms)


def evaluate_on_plus_state(p_sum: PauliSum) -> float:
    expectation = 0.0 + 0j

    for term in p_sum.terms:
        if all(op == "X" for op in term.operators.values()):
            expectation += term.coefficient

    if abs(expectation.imag) > 1e-8:
        raise ValueError(f"Expected real expectation value, got {expectation}")

    return float(expectation.real)


def evaluate_on_zero_state(p_sum: PauliSum) -> float:
    expectation = 0.0 + 0j

    for term in p_sum.terms:
        if all(op == "Z" for op in term.operators.values()):
            expectation += term.coefficient

    if abs(expectation.imag) > 1e-8:
        raise ValueError(f"Expected real expectation value, got {expectation}")

    return float(expectation.real)
