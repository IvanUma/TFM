from __future__ import annotations

import importlib
import heapq
import json
import threading
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
from qiskit_aer import AerSimulator

CONFIG_PATH = Path(__file__).with_name("config.json")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

APPROACH = CONFIG.get("approach", "clifford").lower()

if APPROACH == "parametric":
    q_strategy = importlib.import_module(
        "algorithms.multi_objective_max_cut.max_cut_parametric"
    )
elif APPROACH == "clifford":
    q_strategy = importlib.import_module(
        "algorithms.multi_objective_max_cut.max_cut_clifford"
    )
else:
    raise ValueError(
        f"Unsupported multiobjective approach: {APPROACH!r}. "
        "Expected 'clifford' or 'parametric'."
    )

EvolutionaryIndividual = q_strategy.EvolutionaryIndividual
build_quantum_circuit = q_strategy.build_quantum_circuit
cx_quantum_circuit = q_strategy.cx_quantum_circuit
generate_guided_individual = q_strategy.generate_guided_individual
generate_heuristic_individual = q_strategy.generate_heuristic_individual
load_external_maxcut_instance = q_strategy.load_external_maxcut_instance
mut_quantum_circuit = q_strategy.mut_quantum_circuit
simplify_circuit = q_strategy.simplify_circuit
get_cache_key = getattr(
    q_strategy,
    "get_cache_key",
    lambda num_qubits, individual: (num_qubits, tuple(individual)),
)

THREAD_LOCAL = threading.local()

CIRCUIT_CACHE: Dict[tuple, dict] = {}
STATE_CACHE: Dict[tuple, float] = {}

HIT_THRESHOLD = 5


def get_simulator() -> AerSimulator:
    simulator = getattr(THREAD_LOCAL, "simulator", None)

    if simulator is None:
        simulator = AerSimulator(method="matrix_product_state")
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
    cutoff = max(1, int(sum(counts.values()) * gamma))
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
        probability = count / simulation_shots
        cut_edges = 0.0
        corrected_state = state[::-1]

        for u, v, data in graph_instance.edges(data=True):
            if corrected_state[u] != corrected_state[v]:
                cut_edges += data.get("weight", 1.0)

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

    ind_key = get_cache_key(num_qubits, individual)
    cached = CIRCUIT_CACHE.get(ind_key)

    if cached is None:
        qc_phys = build_quantum_circuit(
            individual,
            num_qubits,
        )
        qc_meas = qc_phys.copy()
        qc_meas.measure_all()
        depth = float(qc_phys.depth())
        counts = get_simulator().run(qc_meas, shots=shots).result().get_counts()

        cached = {
            "qc_phys": qc_phys,
            "qc_meas": qc_meas,
            "depth": depth,
            "counts": counts,
            "shots": shots,
            "hits": 0,
        }
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
