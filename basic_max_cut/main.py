import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
import networkx as nx
import numpy as np
from deap import algorithms, base, creator, tools
from qiskit_aer import AerSimulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EvolutionaryIndividual,
    build_quantum_circuit,
    cx_quantum_circuit,
    evaluate_circuit,
    generate_heuristic_individual,
    generate_guided_individual,
    load_external_maxcut_instance,
    max_cut_fitness,
    mut_quantum_circuit,
)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox: base.Toolbox = base.Toolbox()


def main() -> None:
    project_root = Path(__file__).parent.parent
    file_path = str(project_root / "max_cut_instances" / "petersen.txt")
    graph: nx.Graph
    num_qubits: int
    optimal_classical_cut: int
    graph, num_qubits, optimal_classical_cut = load_external_maxcut_instance(file_path)

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=num_qubits,
        length=20,
        graph_instance=graph,
    )
    toolbox.register(
        "individual_heuristic",
        generate_heuristic_individual,
        num_qubits=num_qubits,
        graph_instance=graph,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", evaluate_circuit, graph_instance=graph, num_qubits=num_qubits
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

    print(f"Loaded External Instance: {file_path}")
    print(f"Nodes / Qubits detected: {num_qubits}")
    print(f"Target Optimal Classical MaxCut: {optimal_classical_cut} cuts\n")

    mu = 100
    lambda_ = 150
    num_heuristic = int(mu * 0.2)
    num_random = mu - num_heuristic
    population: List[EvolutionaryIndividual] = []

    for _ in range(num_heuristic):
        population.append(creator.Individual(toolbox.individual_heuristic()))
    for _ in range(num_random):
        population.append(creator.Individual(toolbox.individual()))

    crossover_prob = 0.5
    mutation_prob = 0.4
    generations = 100
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(key=len)
    statistics = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    statistics.register("min", np.min)
    statistics.register("max", np.max)
    statistics.register("mean", np.mean)

    print(
        "Starting evolutionary quantum architecture search (Upgraded Mu + Lambda)...\n"
    )
    final_population, logbook = algorithms.eaMuPlusLambda(
        population=population,
        toolbox=toolbox,
        mu=mu,
        lambda_=lambda_,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        stats=statistics,
        verbose=True,
    )

    best_individual: EvolutionaryIndividual = tools.selBest(final_population, k=1)[0]
    qc_draw_copy = build_quantum_circuit(best_individual, num_qubits)
    qc_final = build_quantum_circuit(best_individual, num_qubits, measure=True)
    sim = AerSimulator()
    final_counts = sim.run(qc_final, shots=3000).result().get_counts()
    best_quantum_cut = -max_cut_fitness(
        final_counts, 3000, graph, [], gate_penalty_weight=0.0
    )
    approximation_ratio = best_quantum_cut / optimal_classical_cut

    print("\n=================== FINAL QUANTUM CIRCUIT ===================")
    print(qc_draw_copy.draw(output="text"))
    print("=============================================================")
    print("\n=================== FINAL BENCHMARK ===================")
    print(f"Target Benchmark Instance: {file_path}")
    print(f"Exact Classical Maximum Cut: {optimal_classical_cut}")
    print(f"Best Quantum Expected Cut Found: {best_quantum_cut:.4f}")
    print(f"Final Approximation Ratio (Alpha): {approximation_ratio:.4f}")
    print(f"Total Gates in Best Circuit: {len(best_individual)}")
    print("=======================================================")

    execution_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    problem_name = Path(file_path).stem
    circuit_name = f"{problem_name}_q{num_qubits}_g{len(best_individual)}"
    output_base = f"{circuit_name}_{execution_stamp}"
    results_dir = project_root / "results" / problem_name
    results_dir.mkdir(parents=True, exist_ok=True)

    generations_axis = logbook.select("gen")
    best_fitness = logbook.chapters["fitness"].select("min")
    mean_size = logbook.chapters["size"].select("mean")

    output_data = {
        "config": {
            "instance_file": file_path,
            "num_qubits": num_qubits,
            "generations": generations,
            "mu": mu,
            "lambda": lambda_,
        },
        "results": {
            "exact_classical_cut": int(optimal_classical_cut),
            "best_quantum_cut": float(best_quantum_cut),
            "approximation_ratio": float(approximation_ratio),
            "best_individual_gate_count": len(best_individual),
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "best_fitness": [float(f) for f in best_fitness],
            "mean_gate_count": [float(s) for s in mean_size],
        },
    }

    json_filename = results_dir / f"{output_base}.json"
    png_filename = results_dir / f"{output_base}.png"

    with open(json_filename, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"[SERVER INFO] Datos del logbook guardados con éxito en '{json_filename}'")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Mejor Fitness (Minimización)", color=color)
    ax1.plot(
        generations_axis, best_fitness, color=color, linewidth=2, label="Mejor Fitness"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Longitud Media del Circuito (Puertas)", color=color)
    ax2.plot(
        generations_axis,
        mean_size,
        color=color,
        linestyle="--",
        linewidth=2,
        label="Tamaño Medio",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    plt.title("Dinámica Evolutiva: Optimización de Fitness vs. Control de Bloat")
    fig.tight_layout()
    plt.savefig(png_filename, dpi=300)
    print(f"[SERVER INFO] Gráfica de control guardada con éxito en '{png_filename}'")


if __name__ == "__main__":
    main()
