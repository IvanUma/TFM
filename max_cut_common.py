from __future__ import annotations

import random
from typing import List, Tuple, Union

import networkx as nx
from qiskit import QuantumCircuit

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
    """Simplifica el circuito eliminando compuertas que se cancelan de forma continua

    o que conmutan a través de otras compuertas sin interactuar con los mismos qubits.
    """
    prefix = individual[:num_qubits]
    mutable = list(individual[num_qubits:])

    simplified: EvolutionaryIndividual = []

    for gate in mutable:
        gate_type = gate[0]

        if gate_type not in ["H", "CX"]:
            simplified.append(gate)
            continue

        gate_qubits = set(gate[1:])
        can_cancel = False

        for j in range(len(simplified) - 1, -1, -1):
            prev_gate = simplified[j]
            prev_qubits = set(prev_gate[1:])

            if not gate_qubits.isdisjoint(prev_qubits):
                if gate == prev_gate:
                    simplified.pop(j)
                    can_cancel = True
                break

        if not can_cancel:
            simplified.append(gate)

    return prefix + simplified


def build_quantum_circuit(
    individual: EvolutionaryIndividual, num_qubits: int, measure: bool = False
) -> QuantumCircuit:
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
