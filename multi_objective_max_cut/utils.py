from __future__ import annotations

import threading
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

CIRCUIT_CACHE: Dict[tuple, dict] = {}
CACHE_LOCK = threading.Lock()
HIT_THRESHOLD = 5
THREAD_LOCAL = threading.local()


def get_simulator() -> AerSimulator:
    simulator = getattr(THREAD_LOCAL, "simulator", None)
    if simulator is None:
        simulator = AerSimulator()
        THREAD_LOCAL.simulator = simulator
    return simulator


def cut_value_for_state(state: str, edges_tuple: tuple) -> float:
    corrected_state = state[::-1]
    return sum(
        weight
        for u, v, weight in edges_tuple
        if corrected_state[u] != corrected_state[v]
    )


def cvar_from_counts(counts: Dict[str, int], edges_tuple: tuple, gamma: float) -> float:
    cutoff = max(1, int(sum(counts.values()) * gamma))
    remaining = cutoff
    weighted_total = 0.0

    scored_counts = sorted(
        (
            (cut_value_for_state(state, edges_tuple), count)
            for state, count in counts.items()
        ),
        reverse=True,
    )

    for cut_value, count in scored_counts:
        selected = min(count, remaining)
        weighted_total += cut_value * selected
        remaining -= selected
        if remaining == 0:
            break

    return weighted_total / cutoff


def max_cut_fitness(
    counts: Dict[str, int], simulation_shots: int, graph_instance: nx.Graph
) -> float:
    expected_cut_value = 0.0
    for state, count in counts.items():
        probability: float = count / simulation_shots
        cut_edges = 0.0
        corrected_state: str = state[::-1]
        for u, v, data in graph_instance.edges(data=True):
            weight = data.get("weight", 1.0)
            if corrected_state[u] != corrected_state[v]:
                cut_edges += weight
        expected_cut_value += cut_edges * probability
    return -expected_cut_value


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    graph_instance: nx.Graph,
    num_qubits: int,
    shots: int,
    gamma: float = 0.1,
) -> Tuple[float, float]:
    simplified = simplify_circuit(individual, num_qubits)
    individual[:] = simplified

    edges_tuple = tuple(
        sorted(
            (u, v, graph_instance[u][v].get("weight", 1.0))
            for u, v in graph_instance.edges()
        )
    )

    ind_key = (num_qubits, tuple(individual))

    with CACHE_LOCK:
        cached_data = CIRCUIT_CACHE.get(ind_key)

    if cached_data is not None:
        with CACHE_LOCK:
            cached_data["hits"] += 1
            current_hits = cached_data["hits"]
            cached_shots = cached_data["simulator_shots"]

        if current_hits >= HIT_THRESHOLD and shots > cached_shots:
            qc_meas = build_quantum_circuit(individual, num_qubits, measure=True)
            qc_phys = build_quantum_circuit(individual, num_qubits, measure=False)

            counts = get_simulator().run(qc_meas, shots=shots).result().get_counts()
            circuit_depth = float(qc_phys.depth())

            with CACHE_LOCK:
                cached_data["counts"] = counts
                cached_data["depth"] = circuit_depth
                cached_data["simulator_shots"] = shots
        else:
            counts = cached_data["counts"]
            circuit_depth = cached_data["depth"]
    else:
        qc_meas = build_quantum_circuit(individual, num_qubits, measure=True)
        qc_phys = build_quantum_circuit(individual, num_qubits, measure=False)

        counts = get_simulator().run(qc_meas, shots=shots).result().get_counts()
        circuit_depth = float(qc_phys.depth())

        with CACHE_LOCK:
            CIRCUIT_CACHE[ind_key] = {
                "counts": counts,
                "depth": circuit_depth,
                "hits": 0,
                "simulator_shots": shots,
            }

    cvar_cut = cvar_from_counts(counts, edges_tuple, gamma)

    max_degree = (
        max(dict(graph_instance.degree()).values())
        if graph_instance.number_of_nodes() > 0
        else 1
    )

    max_depth_threshold = max(15, max_degree * 3 * 2)

    if circuit_depth > max_depth_threshold:
        penalized_depth = circuit_depth + 2.0 * (
            (circuit_depth - max_depth_threshold) ** 2
        )
    else:
        penalized_depth = circuit_depth

    return (-cvar_cut, penalized_depth)
