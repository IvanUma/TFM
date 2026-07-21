from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    Optimize1qGatesSimpleCommutation,
)

logger = logging.getLogger(__name__)

_SIMPLIFY_PASS_MANAGER = PassManager(
    [CommutativeCancellation(), Optimize1qGatesSimpleCommutation()]
)

QuantumGen = Union[
    Tuple[str, int],
    Tuple[str, int, int],
]

EvolutionaryIndividual = List[QuantumGen]

BRUTE_FORCE_NODE_LIMIT = 20


def compute_optimal_cut(
    graph_instance: nx.Graph,
    brute_force_limit: int = BRUTE_FORCE_NODE_LIMIT,
) -> Tuple[float, bool]:
    n = graph_instance.number_of_nodes()
    if n == 0:
        return 0.0, True

    edges = list(graph_instance.edges(data=True))

    if n <= brute_force_limit:
        nodes = list(graph_instance.nodes())
        best = 0.0
        for mask in range(1, 1 << (n - 1)):
            partition = {nodes[i] for i in range(1, n) if mask & (1 << (i - 1))}
            cut = sum(
                data.get("weight", 1.0)
                for u, v, data in edges
                if (u in partition) != (v in partition)
            )
            best = max(best, cut)
        return best, True

    logger.warning(
        "Graph with %d nodes exceeds brute force limit (%d); "
        "falling back to networkx one_exchange heuristic (approximate optimum).",
        n,
        brute_force_limit,
    )
    cut_value, _ = nx.algorithms.approximation.one_exchange(
        graph_instance, weight="weight"
    )
    return cut_value, False


def load_external_maxcut_instance(file_path: str) -> Tuple[nx.Graph, int, float]:
    graph_instance = nx.Graph()

    with open(file_path, "r") as f:
        header = f.readline().split()
        num_nodes = int(header[0])
        header_optimal = float(header[2]) if len(header) > 2 else None

        graph_instance.add_nodes_from(range(num_nodes))

        for line in f:
            if not line.strip():
                continue
            parts = list(map(float, line.split()))
            weight = parts[2] if len(parts) > 2 else 1.0
            graph_instance.add_edge(
                int(parts[0]) - 1,
                int(parts[1]) - 1,
                weight=weight,
            )

    if header_optimal is not None:
        optimal_cut = header_optimal
    else:
        optimal_cut, is_exact = compute_optimal_cut(graph_instance)
        logger.info(
            "Classical optimum not present in %s; computed %s -> %.3f",
            file_path,
            "exactly" if is_exact else "heuristically",
            optimal_cut,
        )

    return graph_instance, num_nodes, optimal_cut


def precompute_edge_arrays(graph_instance: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = list(graph_instance.edges(data=True))
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    w = np.array([e[2].get("weight", 1.0) for e in edges], dtype=np.float64)
    return u, v, w


def max_cut_fitness_from_arrays(
    weights: Dict[str, float],
    edge_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    num_nodes: int,
    alpha: float = 1.0,
) -> float:
    u, v, w = edge_arrays
    bitstrings = list(weights.keys())
    if not bitstrings:
        return 0.0

    counts_arr = np.array([weights[b] for b in bitstrings])

    bits = np.array([
        [int(c) for c in bs.replace(" ", "").zfill(num_nodes)[::-1]]
        for bs in bitstrings
    ])

    cut_values = (bits[:, u] != bits[:, v]) @ w

    order = np.argsort(-cut_values)
    cut_values = cut_values[order]
    counts_arr = counts_arr[order]

    total = float(counts_arr.sum())
    if total == 0:
        return 0.0

    target = min(max(alpha, 1e-6), 1.0) * total

    cum = np.cumsum(counts_arr)
    take = np.minimum(counts_arr, np.maximum(0.0, target - (cum - counts_arr)))
    accumulated = float(take.sum())
    weighted = float((cut_values * take).sum())
    return weighted / accumulated if accumulated > 0 else 0.0


def max_cut_fitness(
    weights: Dict[str, float],
    graph_instance: nx.Graph,
    alpha: float = 1.0,
) -> float:
    return max_cut_fitness_from_arrays(
        weights,
        precompute_edge_arrays(graph_instance),
        graph_instance.number_of_nodes(),
        alpha,
    )


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


def generate_heuristic_individual(
    num_qubits: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual = [("H", q) for q in range(num_qubits)]

    if graph_instance.number_of_nodes() == 0:
        return individual

    added_edges = set()
    start = random.choice(list(graph_instance.nodes()))

    for u, v in nx.dfs_edges(graph_instance, source=start):
        individual.append(("CX", u, v))
        added_edges.add(tuple(sorted((u, v))))

    for u, v in graph_instance.edges():
        edge = tuple(sorted((u, v)))
        if edge not in added_edges:
            individual.append(("CX", u, v))

    for _ in range(num_qubits // 2 + 1):
        individual.append(("S", random.randint(0, num_qubits - 1)))

    return individual


def cx_quantum_circuit(
    ind1: EvolutionaryIndividual,
    ind2: EvolutionaryIndividual,
    num_qubits: int,
    indpb: float = 0.5,
):
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
    graph_instance: nx.Graph,
    indpb: float,
    gate_generator: Callable[[int, nx.Graph], QuantumGen],
):
    i = num_qubits
    while i < len(individual):
        if random.random() < indpb:
            action = random.choice(["INSERT", "DELETE", "REPLACE"])

            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue
            elif action == "INSERT":
                individual.insert(i, gate_generator(num_qubits, graph_instance))
            else:
                individual[i] = gate_generator(num_qubits, graph_instance)

        i += 1

    return (individual,)


def simplify_gate_sequence(
    gates: List[QuantumGen],
    num_qubits: int,
) -> List[QuantumGen]:
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
