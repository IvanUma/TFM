from __future__ import annotations

import random
from typing import List, Tuple, Union

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    Optimize1qGatesSimpleCommutation,
)

import max_cut_common as common

QuantumGen = Union[
    Tuple[str, int],
    Tuple[str, int, int],
    Tuple[str, int, str, int],
]
EvolutionaryIndividual = List[QuantumGen]

CLIFFORD_GATES: List[str] = ["H", "S", "CX"]
PARAMETRIC_GATES: List[str] = ["RX", "RY", "RZ"]

load_external_maxcut_instance = common.load_external_maxcut_instance


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> QuantumGen:
    gate: str = random.choice(CLIFFORD_GATES)
    if gate == "CX" and graph_instance.number_of_edges() > 0:
        edges: List[Tuple[int, int]] = list(graph_instance.edges())
        chosen_edge: Tuple[int, int] = random.choice(edges)
        if random.random() > 0.5:
            return ("CX", chosen_edge[0], chosen_edge[1])
        return ("CX", chosen_edge[1], chosen_edge[0])

    q: int = random.randint(0, num_qubits - 1)
    if gate == "S":
        return ("S", q)
    return ("H", q)


def generate_random_param_block(
    num_qubits: int, graph_instance: nx.Graph, param_idx: int
) -> QuantumGen:
    gate = random.choice(PARAMETRIC_GATES)
    q = random.randint(0, num_qubits - 1)
    return ("PARAM_BLOCK", param_idx, gate, q)


def generate_guided_individual(
    num_qubits: int,
    length: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", q) for q in range(num_qubits)]
    max_params = 3

    for _ in range(length):
        if random.random() < 0.25:
            p_idx = random.randint(0, max_params - 1)
            individual.append(
                generate_random_param_block(num_qubits, graph_instance, p_idx)
            )
        else:
            individual.append(generate_random_gate(num_qubits, graph_instance))
    return individual


def generate_heuristic_individual(
    num_qubits: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", q) for q in range(num_qubits)]
    if graph_instance.number_of_nodes() == 0:
        return individual

    added_edges = set()
    start_node = random.choice(list(graph_instance.nodes()))

    for u, v in nx.dfs_edges(graph_instance, source=start_node):
        individual.append(("CX", u, v))
        added_edges.add(tuple(sorted((u, v))))

    for u, v in graph_instance.edges():
        edge = tuple(sorted((u, v)))
        if edge not in added_edges:
            individual.append(("CX", u, v))
            added_edges.add(edge)

    for _ in range(num_qubits // 2 + 1):
        q = random.randint(0, num_qubits - 1)
        individual.append(("S", q))

    return individual


def cx_quantum_circuit(
    ind1: EvolutionaryIndividual,
    ind2: EvolutionaryIndividual,
    num_qubits: int,
) -> Tuple[EvolutionaryIndividual, EvolutionaryIndividual]:
    size = min(len(ind1), len(ind2)) - num_qubits
    if size < 2:
        return ind1, ind2

    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    cxpoint1 += num_qubits
    cxpoint2 += num_qubits
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2],
        ind1[cxpoint1:cxpoint2],
    )
    return ind1, ind2


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    graph_instance: nx.Graph,
    indpb: float,
) -> Tuple[EvolutionaryIndividual]:
    i: int = num_qubits
    max_params = 3
    while i < len(individual):
        gen = individual[i]

        if gen[0] == "PARAM_BLOCK":
            if random.random() < indpb:
                param_idx = gen[1]
                if random.random() < 0.5:
                    new_gate = random.choice(PARAMETRIC_GATES)
                    individual[i] = ("PARAM_BLOCK", param_idx, new_gate, gen[3])
                else:
                    new_q = random.randint(0, num_qubits - 1)
                    individual[i] = ("PARAM_BLOCK", param_idx, gen[2], new_q)
                i += 1
                continue

        if random.random() < indpb:
            action: str = random.choice(["INSERT", "DELETE", "REPLACE"])
            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue
            if action == "REPLACE":
                if random.random() < 0.25:
                    p_idx = random.randint(0, max_params - 1)
                    individual[i] = generate_random_param_block(
                        num_qubits, graph_instance, p_idx
                    )
                else:
                    individual[i] = generate_random_gate(num_qubits, graph_instance)
            if action == "INSERT":
                if random.random() < 0.25:
                    p_idx = random.randint(0, max_params - 1)
                    individual.insert(
                        i,
                        generate_random_param_block(num_qubits, graph_instance, p_idx),
                    )
                else:
                    individual.insert(
                        i, generate_random_gate(num_qubits, graph_instance)
                    )
                i += 1
        i += 1
    return (individual,)


def simplify_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
) -> EvolutionaryIndividual:
    if any(gen[0] == "PARAM_BLOCK" for gen in individual):
        return individual

    prefix = individual[:num_qubits]
    mutable = list(individual[num_qubits:])

    qc = QuantumCircuit(num_qubits)
    for gen in mutable:
        gate_type = gen[0]
        if gate_type == "H":
            qc.h(gen[1])
        elif gate_type == "S":
            qc.s(gen[1])
        elif gate_type == "CX":
            qc.cx(gen[1], gen[2])

    pm = PassManager(
        [
            CommutativeCancellation(),
            Optimize1qGatesSimpleCommutation(),
        ]
    )
    qc_optimized = pm.run(qc)

    optimized_mutable: EvolutionaryIndividual = []
    for instruction in qc_optimized.data:
        gate_name = instruction.operation.name.upper()
        qubits = [qc_optimized.find_bit(q).index for q in instruction.qubits]

        if gate_name == "H":
            optimized_mutable.append(("H", qubits[0]))
        elif gate_name == "S":
            optimized_mutable.append(("S", qubits[0]))
        elif gate_name == "CX":
            optimized_mutable.append(("CX", qubits[0], qubits[1]))

    return prefix + optimized_mutable


def build_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    theta_values: list[float] | None = None,
    measure: bool = False,
) -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)
    for gen in individual:
        gate_type: str = gen[0]
        if gate_type == "PARAM_BLOCK":
            param_idx, rot_gate, qubit = gen[1], gen[2], gen[3]

            if theta_values is None:
                theta = 0.0
            else:
                try:
                    theta = theta_values[param_idx]
                except Exception:
                    theta = 0.0

            if rot_gate == "RX":
                qc.rx(theta, qubit)
            elif rot_gate == "RY":
                qc.ry(theta, qubit)
            elif rot_gate == "RZ":
                qc.rz(theta, qubit)
        else:
            if gate_type == "H":
                qc.h(gen[1])
            elif gate_type == "S":
                qc.s(gen[1])
            elif gate_type == "CX":
                qc.cx(gen[1], gen[2])
    if measure:
        qc.measure_all()
    return qc


def get_cache_key(num_qubits: int, individual: EvolutionaryIndividual) -> tuple:
    hashable_ind = []
    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            hashable_ind.append(("PARAM_BLOCK", gen[1], gen[2], gen[3]))
        else:
            hashable_ind.append(gen)
    return (num_qubits, tuple(hashable_ind))
