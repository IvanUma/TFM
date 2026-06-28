from __future__ import annotations

import functools
import json
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from deap import algorithms, base, creator, tools
from qiskit import qpy
from qiskit_aer import AerSimulator

matplotlib.use("Agg")

if not hasattr(creator, "MultiFitness"):
    creator.create("MultiFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "MultiIndividual"):
    creator.create("MultiIndividual", list, fitness=creator.MultiFitness)

toolbox: base.Toolbox = base.Toolbox()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from algorithms.multi_objective_max_cut.utils import (
        APPROACH,
        CONFIG,
        CONFIG_PATH,
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

    nodes_input = input(
        "Enter the number of nodes for the instance group (e.g., 30): "
    ).strip()
    if not nodes_input:
        print("[ERROR] No node size was specified.")
        return

    instances_dir = project_root / "max_cut_instances"
    pattern = f"instance_{nodes_input}nodes_*.txt"
    instance_files = sorted(list(instances_dir.glob(pattern)))

    if not instance_files:
        print(
            f"[ERROR] No instances found matching pattern '{pattern}' in '{instances_dir}'"
        )
        return

    print(
        f"[INFO] Found {len(instance_files)} instances for {nodes_input} nodes. Loading datasets..."
    )

    graphs_data = []
    num_qubits = None

    for f_path in instance_files:
        graph, n_q, optimal_classical_cut = load_external_maxcut_instance(str(f_path))
        if num_qubits is None:
            num_qubits = n_q
        elif n_q != num_qubits:
            print(
                f"[WARNING] Skipping {f_path.name} due to qubit size mismatch (expected {num_qubits}, got {n_q})"
            )
            continue
        graphs_data.append((graph, optimal_classical_cut, f_path.name))

    if not graphs_data:
        print("[ERROR] No valid graphs could be loaded.")
        return

    union_graph = nx.Graph()
    union_graph.add_nodes_from(range(num_qubits))
    for graph, _, _ in graphs_data:
        for u, v in graph.edges():
            union_graph.add_edge(u, v)

    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        return

    population_config = CONFIG["population"]
    variation_config = CONFIG["variation"]
    evolution_config = CONFIG["evolution"]
    gamma_config = CONFIG["gamma_schedule"]
    evaluation_config = CONFIG["evaluation"]

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=num_qubits,
        length=max(
            evolution_config["guided_individual_length_min"],
            num_qubits * evolution_config["guided_individual_length_factor"],
        ),
        graph_instance=union_graph,
    )

    toolbox.register(
        "individual_heuristic",
        generate_heuristic_individual,
        num_qubits=num_qubits,
        graph_instance=union_graph,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_quantum_circuit, num_qubits=num_qubits)
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=num_qubits,
        graph_instance=union_graph,
        indpb=variation_config["mutation_indpb"],
    )

    toolbox.register("select", tools.selNSGA2)

    pool = Pool(processes=2)
    toolbox.register("map", pool.map)

    print(f"\n--- GENERALIZED META-LEARNING CONFIGURATION ---")
    print(f"Total instances loaded: {len(graphs_data)}")
    print(f"Qubits / Scale: {num_qubits}")
    print(f"Approach: {APPROACH.upper()}\n")

    mu = population_config["mu"]
    lambda_ = population_config["lambda"]
    population: List[EvolutionaryIndividual] = []

    for _ in range(int(mu * 0.2)):
        population.append(creator.MultiIndividual(toolbox.individual_heuristic()))

    for _ in range(mu - len(population)):
        population.append(creator.MultiIndividual(toolbox.individual()))

    crossover_prob = variation_config["crossover_prob"]
    mutation_prob = variation_config["mutation_prob"]
    generations = evolution_config["generations"]
    initial_gamma = gamma_config["initial"]
    final_gamma = gamma_config["final"]

    logbook = tools.Logbook()
    logbook.header = ["gen", "shots", "gamma", "best_avg_ar", "best_depth"]

    stats_ar = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(ar=stats_ar, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    graphs_eval_list = [(g, opt) for g, opt, _ in graphs_data]

    for gen in range(generations):
        progress = gen / (generations - 1) if generations > 1 else 1.0
        current_shots = int(
            evaluation_config["shots_start"]
            + (evaluation_config["shots_end"] - evaluation_config["shots_start"])
            * progress
        )
        current_gamma = max(
            final_gamma,
            initial_gamma - (initial_gamma - final_gamma) * progress,
        )

        toolbox.register(
            "evaluate",
            functools.partial(
                evaluate_circuit,
                graphs_data=graphs_eval_list,
                num_qubits=num_qubits,
                shots=current_shots,
                gamma=current_gamma,
            ),
        )

        population, _ = algorithms.eaMuPlusLambda(
            population=population,
            toolbox=toolbox,
            mu=mu,
            lambda_=lambda_,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=1,
            stats=None,
            halloffame=None,
            verbose=False,
        )

        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]

        best_individual = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        best_avg_ar = -best_individual.fitness.values[0]
        best_depth = best_individual.fitness.values[1]

        record = statistics.compile(population)
        logbook.record(
            gen=gen,
            shots=current_shots,
            gamma=current_gamma,
            best_avg_ar=best_avg_ar,
            best_depth=best_depth,
            **record,
        )

        print(
            f"Gen {gen}: Mean Approx Ratio = {best_avg_ar:.4f} | Depth = {best_depth:.1f}"
        )

    pool.close()
    pool.join()

    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]

    simulator = AerSimulator()
    best_validation_avg_ar = -1.0
    best_individual = None

    for ind in pareto_front:
        total_val_ar = 0.0
        for g_idx, (graph, optimal_classical_cut, _) in enumerate(graphs_data):
            if (
                APPROACH == "parametric"
                and hasattr(ind, "stored_thetas")
                and isinstance(ind.stored_thetas, dict)
            ):
                thetas = ind.stored_thetas.get(g_idx, None)
            else:
                thetas = None

            qc = build_quantum_circuit(
                ind, num_qubits, theta_values=thetas, measure=True
            )
            counts = (
                simulator.run(qc, shots=evaluation_config["final_validation_shots"])
                .result()
                .get_counts()
            )
            cut = -max_cut_fitness(
                counts, evaluation_config["final_validation_shots"], graph
            )
            total_val_ar += cut / optimal_classical_cut

        avg_val_ar = total_val_ar / len(graphs_data)
        if avg_val_ar > best_validation_avg_ar:
            best_validation_avg_ar = avg_val_ar
            best_individual = ind

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = (
        project_root
        / "results"
        / f"group_{nodes_input}nodes"
        / "multiobjective"
        / APPROACH
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"group_{nodes_input}nodes_{APPROACH}_g{generations}_{timestamp}"

    sample_thetas = (
        best_individual.stored_thetas.get(0, None)
        if (
            APPROACH == "parametric"
            and hasattr(best_individual, "stored_thetas")
            and isinstance(best_individual.stored_thetas, dict)
        )
        else None
    )

    qc_draw = build_quantum_circuit(
        best_individual, num_qubits, theta_values=sample_thetas
    )
    qc_final = build_quantum_circuit(
        best_individual, num_qubits, theta_values=sample_thetas, measure=True
    )

    qc_draw.draw(output="mpl", filename=str(output_dir / f"{output_stem}.pdf"))

    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    generations_axis = logbook.select("gen")
    history_ar = logbook.select("best_avg_ar")
    history_depth = logbook.select("best_depth")

    instance_parameters = {}
    if (
        APPROACH == "parametric"
        and hasattr(best_individual, "stored_thetas")
        and isinstance(best_individual.stored_thetas, dict)
    ):
        for g_idx, (_, _, filename) in enumerate(graphs_data):
            instance_parameters[filename] = best_individual.stored_thetas.get(g_idx, [])

    output_data = {
        "config": {
            "approach": APPROACH,
            "config_file": str(CONFIG_PATH),
            "node_scale": nodes_input,
            "num_instances_evaluated": len(graphs_data),
            "generations": generations,
            "mu": mu,
            "lambda": lambda_,
            "population": population_config,
            "variation": variation_config,
            "evolution": evolution_config,
            "gamma_schedule": gamma_config,
            "evaluation": evaluation_config,
        },
        "results": {
            "best_quantum_avg_approximation_ratio": float(best_validation_avg_ar),
            "best_individual_depth": int(qc_draw.depth()),
            "evaluated_files": [f[2] for f in graphs_data],
            "optimized_parameters": instance_parameters,
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "best_avg_ar": [float(c) for c in history_ar],
            "best_depth": [float(d) for d in history_depth],
        },
    }

    with open(output_dir / f"{output_stem}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"[SERVER INFO] Datos generalizados guardados con éxito en '{output_dir / f'{output_stem}.json'}'"
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Avg Approx Ratio", color=color)
    ax1.plot(
        generations_axis, history_ar, color=color, linewidth=2, label="Best Avg AR"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Depth", color=color)
    ax2.plot(
        generations_axis,
        history_depth,
        color=color,
        linestyle="--",
        linewidth=2,
        label="Depth",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"QNAS Evolutionary Dynamics ({APPROACH.upper()}): Avg AR vs Depth")
    fig.tight_layout()
    plt.savefig(output_dir / f"{output_stem}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
