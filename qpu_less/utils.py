from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import networkx as nx

from qpu_needless_engine import (
    PauliSum,
    PauliTerm,
    evaluate_on_plus_state,
    evolve_sum_under_mixer,
    evolve_sum_under_problem_edge,
    simplify_sum,
)

ANALYTICAL_CACHE: Dict[Tuple[float, ...], float] = {}
CACHE_LOCK = threading.Lock()


def compute_qaoa_energy_analytically(
    graph_instance: nx.Graph,
    gammas: List[float],
    betas: List[float],
) -> float:

    # 1. Construimos el Hamiltoniano del problema H_P para MaxCut: 0.5 * W * (I - Z_u Z_v)
    H_P = PauliSum()
    for u, v, data in graph_instance.edges(data=True):
        weight = data.get("weight", 1.0)
        H_P.append(PauliTerm(0.5 * weight, {}))
        H_P.append(PauliTerm(-0.5 * weight, {u: "Z", v: "Z"}))

    current_operators = simplify_sum(H_P)

    for layer in range(len(gammas) - 1, -1, -1):
        gamma = gammas[layer]
        beta = betas[layer]

        current_operators = evolve_sum_under_mixer(current_operators, beta)

        for u, v, data in graph_instance.edges(data=True):
            weight = data.get("weight", 1.0)
            current_operators = evolve_sum_under_problem_edge(
                current_operators, u, v, gamma, weight
            )

    return evaluate_on_plus_state(current_operators)


def evaluate_analytical_individual(
    individual: List[float],
    graph_instance: nx.Graph,
) -> Tuple[float, float]:

    p = len(individual) // 2
    gammas = individual[:p]
    betas = individual[p:]

    ind_key = tuple(individual)

    with CACHE_LOCK:
        expected_cut = ANALYTICAL_CACHE.get(ind_key)

    if expected_cut is None:
        expected_cut = compute_qaoa_energy_analytically(graph_instance, gammas, betas)

        with CACHE_LOCK:
            ANALYTICAL_CACHE[ind_key] = expected_cut

    max_degree = max(dict(graph_instance.degree()).values(), default=1)

    theoretical_layer_depth = max_degree + 1
    depth = float(1 + p * theoretical_layer_depth)

    threshold = max(15, max_degree * 6)

    if depth > threshold:
        penalized_depth = depth + 2.0 * (depth - threshold) ** 2
    else:
        penalized_depth = depth

    return (
        -expected_cut,
        penalized_depth,
    )
