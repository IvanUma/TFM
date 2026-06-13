from __future__ import annotations

import heapq
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

THREAD_LOCAL = threading.local()

CIRCUIT_CACHE: Dict[tuple, dict] = {}
STATE_CACHE: Dict[tuple, float] = {}
CACHE_LOCK = threading.Lock()

HIT_THRESHOLD = 5


def get_simulator() -> AerSimulator:
    simulator = getattr(THREAD_LOCAL, "simulator", None)

    if simulator is None:
        simulator = AerSimulator()
        THREAD_LOCAL.simulator = simulator

    return simulator


def cut_value_for_state(
    state: str,
    edges_tuple: tuple,
) -> float:

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


def cvar_from_counts(
    counts: Dict[str, int],
    edges_tuple: tuple,
    gamma: float,
) -> float:

    cutoff = max(
        1,
        int(sum(counts.values()) * gamma),
    )

    scored = [
        (
            cut_value_for_state(state, edges_tuple),
            count,
        )
        for state, count in counts.items()
    ]

    best_states = heapq.nlargest(
        len(scored),
        scored,
        key=lambda x: x[0],
    )

    remaining = cutoff
    total = 0.0

    for cut_value, count in best_states:
        selected = min(
            count,
            remaining,
        )

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
        probability = count / simulation_shots

        cut_edges = 0.0

        corrected_state = state[::-1]

        for u, v, data in graph_instance.edges(data=True):
            if corrected_state[u] != corrected_state[v]:
                cut_edges += data.get(
                    "weight",
                    1.0,
                )

        expected_cut_value += cut_edges * probability

    return -expected_cut_value


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    graph_instance: nx.Graph,
    num_qubits: int,
    shots: int,
    gamma: float = 0.1,
) -> Tuple[float, float]:

    simplified = simplify_circuit(
        individual,
        num_qubits,
    )

    individual[:] = simplified

    edges_tuple = tuple(
        sorted(
            (
                u,
                v,
                graph_instance[u][v].get(
                    "weight",
                    1.0,
                ),
            )
            for u, v in graph_instance.edges()
        )
    )

    ind_key = (
        num_qubits,
        tuple(individual),
    )

    with CACHE_LOCK:
        cached = CIRCUIT_CACHE.get(ind_key)

    if cached is None:
        qc_phys = build_quantum_circuit(
            individual,
            num_qubits,
        )

        qc_meas = qc_phys.copy()
        qc_meas.measure_all()

        depth = float(qc_phys.depth())

        counts = (
            get_simulator()
            .run(
                qc_meas,
                shots=shots,
            )
            .result()
            .get_counts()
        )

        cached = {
            "qc_phys": qc_phys,
            "qc_meas": qc_meas,
            "depth": depth,
            "counts": counts,
            "shots": shots,
            "hits": 0,
        }

        with CACHE_LOCK:
            CIRCUIT_CACHE[ind_key] = cached

    else:
        cached["hits"] += 1

        if cached["hits"] >= HIT_THRESHOLD and shots > cached["shots"]:
            counts = (
                get_simulator()
                .run(
                    cached["qc_meas"],
                    shots=shots,
                )
                .result()
                .get_counts()
            )

            cached["counts"] = counts
            cached["shots"] = shots

    counts = cached["counts"]
    depth = cached["depth"]

    cvar_cut = cvar_from_counts(
        counts,
        edges_tuple,
        gamma,
    )

    max_degree = max(
        dict(graph_instance.degree()).values(),
        default=1,
    )

    threshold = max(
        15,
        max_degree * 6,
    )

    if depth > threshold:
        penalized_depth = depth + 2.0 * (depth - threshold) ** 2
    else:
        penalized_depth = depth

    return (
        -cvar_cut,
        penalized_depth,
    )
