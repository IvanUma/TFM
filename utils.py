import random
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
from deap import creator
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

QuantumGen = Union[Tuple[str, int], Tuple[str, int, int]]
EvolutionaryIndividual = List[QuantumGen]

CLIFFORD_GATES: List[str] = ["H", "S", "CX"]


def load_external_maxcut_instance(file_path: str) -> Tuple[nx.Graph, int, int]:
    graph_instance: nx.Graph = nx.Graph()
    with open(file_path, "r") as f:
        lines = f.readlines()
    first_line = lines[0].split()
    num_nodes = int(first_line[0])
    optimal_cut = int(first_line[2])
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            graph_instance.add_edge(u, v)
    return graph_instance, num_nodes, optimal_cut


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> QuantumGen:
    gate: str = random.choice(CLIFFORD_GATES)
    if gate == "CX" and len(graph_instance.edges()) > 0:
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
    """
    Genera un individuo usando heurística basada en la topología del grafo.
    Aplica una Búsqueda en Profundidad (DFS) para encadenar las puertas CX
    de manera contigua, replicando el comportamiento de correlación local.
    """
    individual: EvolutionaryIndividual = [("H", q) for q in range(num_qubits)]
    if graph_instance.number_of_nodes() == 0:
        return individual

    added_edges = set()
    start_node = random.choice(list(graph_instance.nodes()))

    # 1. Camino conectado (DFS)
    for u, v in nx.dfs_edges(graph_instance, source=start_node):
        individual.append(("CX", u, v))
        added_edges.add(tuple(sorted((u, v))))

    # 2. Rellenar las aristas restantes que cierran los ciclos
    for u, v in graph_instance.edges():
        edge = tuple(sorted((u, v)))
        if edge not in added_edges:
            individual.append(("CX", u, v))
            added_edges.add(edge)

    # 3. Añadir rotaciones de fase locales
    for _ in range(5):
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
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(mutable) - 1:
            gate1 = mutable[i]
            gate2 = mutable[i + 1]
            if gate1 == gate2 and gate1[0] in ["H", "CX"]:
                mutable.pop(i + 1)
                mutable.pop(i)
                changed = True
                continue
            i += 1
    return creator.Individual(prefix + mutable)


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


def max_cut_fitness(
    counts: Dict[str, int],
    simulation_shots: int,
    graph_instance: nx.Graph,
    individual: EvolutionaryIndividual,
    gate_penalty_weight: float = 0.01,
) -> float:
    expected_cut_value = 0.0
    for state, count in counts.items():
        probability: float = count / simulation_shots
        cut_edges = 0
        corrected_state: str = state[::-1]
        for u, v in graph_instance.edges():
            if corrected_state[u] != corrected_state[v]:
                cut_edges += 1
        expected_cut_value += cut_edges * probability
    gate_penalty = len(individual) * gate_penalty_weight
    final_fitness = -expected_cut_value + gate_penalty
    return final_fitness


def evaluate_circuit(
    individual: EvolutionaryIndividual, graph_instance: nx.Graph, num_qubits: int
) -> Tuple[float]:
    simplified = simplify_circuit(individual, num_qubits)
    individual[:] = simplified
    qc = build_quantum_circuit(individual, num_qubits, measure=True)
    simulator: AerSimulator = AerSimulator()
    simulation_shots: int = 3000
    result: Any = simulator.run(qc, shots=simulation_shots).result()
    counts: Dict[str, int] = result.get_counts()
    raw_fitness = max_cut_fitness(
        counts, simulation_shots, graph_instance, individual, gate_penalty_weight=0.0
    )
    max_gate_threshold = 30
    current_length = len(individual)
    if current_length > max_gate_threshold:
        penalty = 0.04 * ((current_length - max_gate_threshold) ** 2)
    else:
        penalty = 0.001 * current_length
    return (raw_fitness + penalty,)
