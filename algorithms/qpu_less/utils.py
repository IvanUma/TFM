from __future__ import annotations

import random
from typing import List, Tuple, Union

import networkx as nx

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
                    int(parts[0]) - 1,
                    int(parts[1]) - 1,
                    weight=weight,
                )

    return graph_instance, num_nodes, optimal_cut


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> QuantumGen:
    gate = random.choice(CLIFFORD_GATES)

    if gate == "CX" and graph_instance.number_of_edges() > 0:
        chosen_edge = random.choice(list(graph_instance.edges()))
        if random.random() > 0.5:
            return ("CX", chosen_edge[0], chosen_edge[1])
        return ("CX", chosen_edge[1], chosen_edge[0])

    qubit = random.randint(0, num_qubits - 1)
    if gate == "S":
        return ("S", qubit)
    return ("H", qubit)


def generate_guided_individual(
    num_qubits: int,
    length: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", qubit) for qubit in range(num_qubits)]

    for _ in range(length):
        individual.append(generate_random_gate(num_qubits, graph_instance))

    return individual


def generate_heuristic_individual(
    num_qubits: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", qubit) for qubit in range(num_qubits)]

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
        individual.append(("S", random.randint(0, num_qubits - 1)))

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
) -> Tuple[EvolutionaryIndividual,]:
    index = num_qubits

    while index < len(individual):
        if random.random() < indpb:
            action = random.choice(["INSERT", "DELETE", "REPLACE"])

            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(index)
                continue

            if action == "REPLACE":
                individual[index] = generate_random_gate(num_qubits, graph_instance)

            if action == "INSERT":
                individual.insert(index, generate_random_gate(num_qubits, graph_instance))
                index += 1

        index += 1

    return (individual,)


def simplify_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
) -> EvolutionaryIndividual:
    _ = num_qubits
    simplified: EvolutionaryIndividual = []

    for gate in individual:
        if simplified and gate == simplified[-1] and gate[0] in {"H", "CX"}:
            simplified.pop()
            continue
        simplified.append(gate)

        if len(simplified) >= 4:
            last_four = simplified[-4:]
            if all(item == gate for item in last_four) and gate[0] == "S":
                del simplified[-4:]

    return simplified


def get_cache_key(num_qubits: int, individual: EvolutionaryIndividual) -> tuple:
    return (num_qubits, tuple(individual))
