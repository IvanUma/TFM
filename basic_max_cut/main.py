import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import networkx as nx
import numpy as np
from deap import algorithms, base, creator, tools
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import max_cut_fitness


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


QuantumGen = Union[Tuple[str, int], Tuple[str, int, int]]
EvolutionaryIndividual = List[QuantumGen]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox: base.Toolbox = base.Toolbox()
CLIFFORD_GATES: List[str] = ["H", "S", "CX"]


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> QuantumGen:
    gate: str = random.choice(CLIFFORD_GATES)
    if gate == "CX" and len(graph_instance.edges()) > 0:
        edges: List[Tuple[int, int]] = list(graph_instance.edges())
        chosen_edge: Tuple[int, int] = random.choice(edges)
        if random.random() > 0.5:
            return ("CX", chosen_edge[0], chosen_edge[1])
        else:
            return ("CX", chosen_edge[1], chosen_edge[0])
    elif gate == "S":
        q: int = random.randint(0, num_qubits - 1)
        return ("S", q)
    else:
        q: int = random.randint(0, num_qubits - 1)
        return ("H", q)


def generate_guided_individual(
    num_qubits: int, length: int, graph_instance: nx.Graph
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = []
    for q in range(num_qubits):
        individual.append(("H", q))
    for _ in range(length):
        individual.append(generate_random_gate(num_qubits, graph_instance))
    return creator.Individual(individual)


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
) -> Tuple[EvolutionaryIndividual,]:
    i: int = num_qubits

    while i < len(individual):
        if random.random() < indpb:
            action: str = random.choice(["INSERT", "DELETE", "REPLACE"])

            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue

            elif action == "REPLACE":
                individual[i] = generate_random_gate(num_qubits, graph_instance)

            elif action == "INSERT":
                new_gate = generate_random_gate(num_qubits, graph_instance)
                individual.insert(i, new_gate)
                i += 1
        i += 1

    return (individual,)


def evaluate_circuit(
    individual: EvolutionaryIndividual,
    graph_instance: nx.Graph,
    num_qubits: int,
    gate_penalty_weight: float = 0.01,
) -> Tuple[float,]:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)

    for gen in individual:
        gate_type: str = gen[0]
        if gate_type == "H":
            qc.h(gen[1])
        elif gate_type == "S":
            qc.s(gen[1])
        elif gate_type == "CX":
            qc.cx(gen[1], gen[2])

    qc.measure_all()

    simulator: AerSimulator = AerSimulator()
    simulation_shots: int = 3000
    result: Any = simulator.run(qc, shots=simulation_shots).result()
    counts: Dict[str, int] = result.get_counts()

    return (
        max_cut_fitness(
            counts, simulation_shots, graph_instance, individual, gate_penalty_weight
        ),
    )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    FILE_PATH: str = str(project_root / "max_cut_instances" / "petersen.txt")

    graph: nx.Graph
    num_qubits: int
    optimal_classical_cut: int
    graph, num_qubits, optimal_classical_cut = load_external_maxcut_instance(FILE_PATH)

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=num_qubits,
        length=20,
        graph_instance=graph,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_circuit,
        graph_instance=graph,
        num_qubits=num_qubits,
        gate_penalty_weight=0.01,
    )

    toolbox.register("mate", cx_quantum_circuit, num_qubits=num_qubits)
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=num_qubits,
        graph_instance=graph,
        indpb=0.2,
    )
    toolbox.register("select", tools.selTournament, tournsize=2)

    print(f"Loaded External Instance: {FILE_PATH}")
    print(f"Nodes / Qubits detected: {num_qubits}")
    print(f"Target Optimal Classical MaxCut: {optimal_classical_cut} cuts\n")

    # Hiperparámetros específicos para mu + lambda
    MU: int = 100  # Tamaño de la población de padres
    LAMBDA: int = (
        150  # Número de hijos generados por generación (se recomienda LAMBDA >= MU)
    )

    population: List[EvolutionaryIndividual] = toolbox.population(n=MU)
    crossover_prob: float = 0.5
    mutation_prob: float = 0.4
    generations: int = 100

    statistics: tools.Statistics = tools.Statistics(lambda ind: ind.fitness.values)
    statistics.register("mean", np.mean)
    statistics.register("min", np.min)
    statistics.register("max", np.max)

    print("Starting evolutionary quantum architecture search (Mu + Lambda)...\n")

    final_population: List[EvolutionaryIndividual]
    logbook: tools.Logbook

    # Ejecución del algoritmo Mu + Lambda (Elitismo natural sin necesidad de HoF)
    final_population, logbook = algorithms.eaMuPlusLambda(
        population=population,
        toolbox=toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        stats=statistics,
        verbose=True,
    )

    # Al ser elitista, el mejor absoluto histórico está garantizado dentro de la población final
    best_individual: EvolutionaryIndividual = tools.selBest(final_population, k=1)[0]

    qc_final: QuantumCircuit = QuantumCircuit(num_qubits)
    for gen in best_individual:
        if gen[0] == "H":
            qc_final.h(gen[1])
        elif gen[0] == "S":
            qc_final.s(gen[1])
        elif gen[0] == "CX":
            qc_final.cx(gen[1], gen[2])
    qc_final.measure_all()

    sim: AerSimulator = AerSimulator()
    res: Any = sim.run(qc_final, shots=3000).result()
    final_counts: Dict[str, int] = res.get_counts()

    from utils import max_cut_fitness as fitness_calc

    best_quantum_cut: float = -fitness_calc(
        final_counts, 3000, graph, [], gate_penalty_weight=0.0
    )

    approximation_ratio: float = best_quantum_cut / optimal_classical_cut

    print("\n=================== FINAL BENCHMARK ===================")
    print(f"Target Benchmark Instance: {FILE_PATH}")
    print(f"Exact Classical Maximum Cut: {optimal_classical_cut}")
    print(f"Best Quantum Expected Cut Found: {best_quantum_cut:.4f}")
    print(f"Final Approximation Ratio (Alpha): {approximation_ratio:.4f}")
    print(f"Total Gates in Best Circuit: {len(best_individual)}")
    print("=======================================================")
