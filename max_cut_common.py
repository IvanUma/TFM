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

QuantumGen = Union[Tuple[str, int], Tuple[str, int, int]]
EvolutionaryIndividual = List[QuantumGen]

CLIFFORD_GATES: List[str] = ["H", "S", "CX"]


def load_external_maxcut_instance(file_path: str) -> Tuple[nx.Graph, int, int]:
    graph_instance = nx.Graph()
    with open(file_path, "r") as f:
        header = f.readline().split()
        num_nodes = int(header[0])
        optimal_cut = int(header[2]) if len(header) > 2 else 0
        graph_instance.add_nodes_from(range(num_nodes))

        for line in f:
            if line.strip():
                parts = list(map(float, line.split()))
                weight = parts[2] if len(parts) > 2 else 1
                graph_instance.add_edge(
                    int(parts[0]) - 1, int(parts[1]) - 1, weight=weight
                )

    return graph_instance, num_nodes, optimal_cut


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


def generate_guided_individual(
    num_qubits: int, length: int, graph_instance: nx.Graph
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", q) for q in range(num_qubits)]
    for _ in range(length):
        individual.append(generate_random_gate(num_qubits, graph_instance))
    return individual


def generate_heuristic_individual(
    num_qubits: int, graph_instance: nx.Graph
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
    ind1: EvolutionaryIndividual, ind2: EvolutionaryIndividual, num_qubits: int
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
    while i < len(individual):
        if random.random() < indpb:
            action: str = random.choice(["INSERT", "DELETE", "REPLACE"])
            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue
            if action == "REPLACE":
                individual[i] = generate_random_gate(num_qubits, graph_instance)
            if action == "INSERT":
                individual.insert(i, generate_random_gate(num_qubits, graph_instance))
                i += 1
        i += 1
    return (individual,)


def simplify_circuit(
    individual: EvolutionaryIndividual, num_qubits: int
) -> EvolutionaryIndividual:
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
    individual: EvolutionaryIndividual, num_qubits: int, measure: bool = False
) -> QuantumCircuit:
    """Construye un objeto QuantumCircuit de Qiskit a partir del genoma del individuo."""
    qc: QuantumCircuit = QuantumCircuit(num_qubits)
    for gen in individual:
        gate_type: str = gen[0]
        if gate_type == "H":
            qc.h(gen[1])
        elif gate_type == "S":
            qc.s(gen[1])
        elif gate_type == "CX":
            qc.cx(gen[1], gen[2])
    if measure:
        qc.measure_all()
    return qc
