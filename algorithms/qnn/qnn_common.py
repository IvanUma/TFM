from __future__ import annotations

import logging
import random
from typing import Callable, List, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesSimpleCommutation

logger = logging.getLogger(__name__)

_SIMPLIFY_PASS_MANAGER = PassManager(
    [CommutativeCancellation(), Optimize1qGatesSimpleCommutation()]
)

QuantumGen = Union[Tuple[str, int], Tuple[str, int, int]]
EvolutionaryIndividual = List[QuantumGen]


def enumerate_qubit_pairs(max_qubits: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(max_qubits) for j in range(i + 1, max_qubits)]


def apply_block(qc: QuantumCircuit, block: EvolutionaryIndividual) -> None:
    for gen in block:
        if gen[0] == "H":
            qc.h(gen[1])
        elif gen[0] == "S":
            qc.s(gen[1])
        elif gen[0] == "CX":
            qc.cx(gen[1], gen[2])


def simplify_gate_sequence(gates: List[QuantumGen], num_qubits: int) -> List[QuantumGen]:
    if len(gates) <= 2:
        return list(gates)
    qc = QuantumCircuit(num_qubits)
    apply_block(qc, gates)
    qc = _SIMPLIFY_PASS_MANAGER.run(qc)
    optimized: List[QuantumGen] = []
    for inst in qc.data:
        name = inst.operation.name.upper()
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "H":
            optimized.append(("H", qubits[0]))
        elif name == "S":
            optimized.append(("S", qubits[0]))
        elif name == "CX":
            optimized.append(("CX", qubits[0], qubits[1]))
    return optimized if optimized else list(gates)


def cx_quantum_circuit(ind1: EvolutionaryIndividual, ind2: EvolutionaryIndividual, num_qubits: int, indpb: float = 0.5):
    size = min(len(ind1), len(ind2))
    if size <= num_qubits:
        return ind1, ind2
    for i in range(num_qubits, size):
        if random.random() < indpb and ind1[i][0] == ind2[i][0]:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    indpb: float,
    gate_generator: Callable[[int], QuantumGen],
):
    i = num_qubits
    while i < len(individual):
        if random.random() < indpb:
            action = random.choice(["INSERT", "DELETE", "REPLACE"])
            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue
            elif action == "INSERT":
                individual.insert(i, gate_generator(num_qubits))
            else:
                individual[i] = gate_generator(num_qubits)
        i += 1
    return (individual,)