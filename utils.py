from typing import Dict, List, Tuple, Union
import networkx as nx

QuantumGen = Union[Tuple[str, int], Tuple[str, int, int]]
EvolutionaryIndividual = List[QuantumGen]


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
