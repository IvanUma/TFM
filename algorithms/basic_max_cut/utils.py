from __future__ import annotations

import heapq
from functools import lru_cache
from typing import Dict, Tuple

import networkx as nx
from qiskit_aer import AerSimulator

import max_cut_common as common

EvolutionaryIndividual = common.EvolutionaryIndividual
build_quantum_circuit = common.build_quantum_circuit
cx_quantum_circuit = common.cx_quantum_circuit
generate_guided_individual = common.generate_guided_individual
generate_heuristic_individual = common.generate_heuristic_individual
load_external_maxcut_instance = common.load_external_maxcut_instance
mut_quantum_circuit = common.mut_quantum_circuit
simplify_circuit = common.simplify_circuit

SIMULATOR = AerSimulator()
STATE_CACHE: Dict[tuple, float] = {}


def cut_value_for_state(state: str, edges_tuple: tuple) -> float:
    key = (state, edges_tuple)
    value = STATE_CACHE.get(key)
    if value is not None:
        return value
    corrected_state = state[::-1]
    value = sum(
        weight
        for u, v, weight in edges_tuple
        if corrected_state[u] != corrected_state[v]
    )
    STATE_CACHE[key] = value
    return value


def cvar_from_counts(counts: Dict[str, int], edges_tuple: tuple, gamma: float) -> float:
    cutoff = max(1, int(sum(counts.values()) * gamma))
    scored = [
        (cut_value_for_state(state, edges_tuple), count)
        for state, count in counts.items()
    ]
    best_states = heapq.nlargest(len(scored), scored, key=lambda x: x[0])
    remaining = cutoff
    total = 0.0
    for cut_value, count in best_states:
        selected = min(count, remaining)
        total += cut_value * selected
        remaining -= selected
        if remaining == 0:
            break
    return total / cutoff


def max_cut_fitness(
    counts: Dict[str, int],
    simulation_shots: int,
    graph_instance: nx.Graph,
) -> float:
    expected_cut_value = 0.0
    for state, count in counts.items():
        probability: float = count / simulation_shots
        cut_edges = 0.0
        corrected_state: str = state[::-1]
        for u, v, data in graph_instance.edges(data=True):
            if corrected_state[u] != corrected_state[v]:
                cut_edges += data.get("weight", 1.0)
        expected_cut_value += cut_edges * probability
    return expected_cut_value


@lru_cache(maxsize=10000)
def _eval_cached(
    individual_tuple: tuple,
    edges_tuple: tuple,
    num_qubits: int,
    shots: int,
    gamma: float,
) -> float:
    individual = list(individual_tuple)
    qc = build_quantum_circuit(individual, num_qubits, measure=True)
    result = SIMULATOR.run(qc, shots=shots).result()
    counts = result.get_counts()
    return cvar_from_counts(counts, edges_tuple, gamma)


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    graph_instance: nx.Graph,
    num_qubits: int,
    shots: int,
    gamma: float,
) -> Tuple[float]:
    simplified = simplify_circuit(individual, num_qubits)
    individual[:] = simplified

    edges_tuple = tuple(
        sorted(
            (u, v, graph_instance[u][v].get("weight", 1.0))
            for u, v in graph_instance.edges()
        )
    )
    raw_fitness = _eval_cached(tuple(individual), edges_tuple, num_qubits, shots, gamma)

    qc_physical = build_quantum_circuit(individual, num_qubits, measure=False)
    current_depth = qc_physical.depth()

    max_degree = (
        max(dict(graph_instance.degree()).values())
        if graph_instance.number_of_nodes() > 0
        else 1
    )

    max_depth_threshold = max(15, max_degree * 6)

    penalty = (
        0.02 * ((current_depth - max_depth_threshold) ** 2)
        if current_depth > max_depth_threshold
        else 0.001 * current_depth
    )

    return (raw_fitness - penalty,)
