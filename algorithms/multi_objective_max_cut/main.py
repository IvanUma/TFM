from __future__ import annotations

import copy
import functools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from qiskit import qpy
from qiskit_aer import AerSimulator

matplotlib.use("Agg")

if not hasattr(creator, "MultiFitness"):
    creator.create("MultiFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "MultiIndividual"):
    creator.create("MultiIndividual", list, fitness=creator.MultiFitness)

toolbox: base.Toolbox = base.Toolbox()


def evaluate_population(individuals, toolbox, update_hof) -> int:
    invalid = [ind for ind in individuals if not ind.fitness.valid]
    if not invalid:
        return 0

    results = toolbox.map(toolbox.evaluate, invalid)
    for ind, (fit, weights, hof_candidates) in zip(invalid, results):
        ind.fitness.values = fit
        ind.stored_thetas = weights
        for block in hof_candidates:
            update_hof(block)

    return len(invalid)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from algorithms.multi_objective_max_cut.utils import (
        APPROACH,
        CONFIG,
        CONFIG_PATH,
        ENABLE_INPUT_PARAMS,
        MAX_QUBITS,
        PARAM_BLOCK_PROB,
        EvolutionaryIndividual,
        build_quantum_circuit,
        build_universal_input_values,
        cx_quantum_circuit,
        evaluate_circuit,
        generate_heuristic_individual,
        generate_guided_individual,
        load_external_maxcut_instance,
        max_cut_fitness,
        mut_quantum_circuit,
        update_hof,
    )

    instance_input = input(
        "Enter the path to the single MaxCut instance file: "
    ).strip()
    instance_path = Path(instance_input)

    if not instance_path.exists():
        print(f"[ERROR] Instance file not found: {instance_path}")
        return

    graph, instance_num_qubits, optimal_classical_cut = load_external_maxcut_instance(
        str(instance_path)
    )
    filename = instance_path.name

    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        return

    if MAX_QUBITS and instance_num_qubits > MAX_QUBITS:
        print(
            f"[ERROR] Instance requires {instance_num_qubits} qubits but "
            f"circuit_scale.max_qubits is {MAX_QUBITS}"
        )
        return

    population_config = CONFIG["population"]
    variation_config = CONFIG["variation"]
    evolution_config = CONFIG["evolution"]
    gamma_config = CONFIG["gamma_schedule"]
    evaluation_config = CONFIG["evaluation"]
    execution_config = CONFIG.get("execution", {})

    if MAX_QUBITS:
        circuit_qubits = MAX_QUBITS
        input_values = build_universal_input_values(graph, MAX_QUBITS)
    else:
        circuit_qubits = instance_num_qubits
        edges_list = list(graph.edges(data=True))
        input_values = [data.get("weight", 1.0) for _, _, data in edges_list]

    dynamic_max_params = len(input_values) if input_values else 1

    toolbox.register("clone", copy.deepcopy)

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=circuit_qubits,
        length=max(
            evolution_config["guided_individual_length_min"],
            circuit_qubits * evolution_config["guided_individual_length_factor"],
        ),
        graph_instance=graph,
        max_params=dynamic_max_params,
        enable_input_params=ENABLE_INPUT_PARAMS,
        param_block_prob=PARAM_BLOCK_PROB,
        max_qubits=MAX_QUBITS,
    )

    toolbox.register(
        "individual_heuristic",
        generate_heuristic_individual,
        num_qubits=circuit_qubits,
        graph_instance=graph,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_quantum_circuit, num_qubits=circuit_qubits)
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=circuit_qubits,
        graph_instance=graph,
        indpb=variation_config["mutation_indpb"],
        max_params=dynamic_max_params,
        enable_input_params=ENABLE_INPUT_PARAMS,
        param_block_prob=PARAM_BLOCK_PROB,
        max_qubits=MAX_QUBITS,
    )
    toolbox.register("select", tools.selNSGA2)

    use_multiprocessing = execution_config.get("multiprocessing", True)
    requested_processes = execution_config.get("processes")
    if not requested_processes:
        cpu_total = os.cpu_count() or 1
        requested_processes = max(1, cpu_total - 1)

    pool = Pool(processes=requested_processes) if use_multiprocessing else None
    toolbox.register("map", pool.map if pool is not None else map)

    print("\n--- SINGLE INSTANCE CONFIGURATION ---")
    print(f"File loaded: {filename}")
    print(f"Instance nodes: {instance_num_qubits}")
    print(f"Circuit qubits: {circuit_qubits}" + (" (universal)" if MAX_QUBITS else ""))
    print(f"Approach: {APPROACH.upper()}")
    print(f"Processes: {requested_processes if pool is not None else 1}\n")

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
    patience = evolution_config.get("patience", generations)
    improvement_epsilon = evolution_config.get("improvement_epsilon", 0.0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "shots", "gamma", "best_avg_ar", "best_depth"]

    stats_ar = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(ar=stats_ar, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    best_ar_ever = -1.0
    stagnant_generations = 0
    last_gen = 0

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
                num_qubits=circuit_qubits,
                graph_instance=graph,
                optimal_classical_cut=optimal_classical_cut,
                input_values=input_values,
                shots=current_shots,
                gamma=current_gamma,
            ),
        )

        if gen == 0:
            evaluate_population(population, toolbox, update_hof)

        offspring = algorithms.varOr(
            population, toolbox, lambda_, crossover_prob, mutation_prob
        )
        evaluate_population(offspring, toolbox, update_hof)
        population[:] = toolbox.select(population + offspring, mu)

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

        print(f"Gen {gen}: Approx Ratio = {best_avg_ar:.4f} | Depth = {best_depth:.1f}")

        last_gen = gen

        if best_avg_ar > best_ar_ever + improvement_epsilon:
            best_ar_ever = best_avg_ar
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= patience:
            print(f"[INFO] Early stopping at generation {gen} (patience={patience})")
            break

    if pool is not None:
        pool.close()
        pool.join()

    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]

    simulator = (
        AerSimulator(method="stabilizer")
        if APPROACH == "clifford"
        else AerSimulator(method="statevector")
    )

    validation_circuits = []
    for ind in pareto_front:
        thetas = getattr(ind, "stored_thetas", [])
        validation_circuits.append(
            build_quantum_circuit(
                ind, circuit_qubits, input_values, thetas, measure=True
            )
        )

    validation_results = simulator.run(
        validation_circuits, shots=evaluation_config["final_validation_shots"]
    ).result()

    best_validation_avg_ar = -1.0
    best_individual = None

    for idx, ind in enumerate(pareto_front):
        counts = validation_results.get_counts(idx)
        cut = max_cut_fitness(counts, graph, alpha=1.0)
        avg_val_ar = cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0

        if avg_val_ar > best_validation_avg_ar:
            best_validation_avg_ar = avg_val_ar
            best_individual = ind

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    instance_name = instance_path.stem
    output_dir = project_root / "results" / "single_instance" / instance_name / APPROACH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{instance_name}_opt_{APPROACH}_g{last_gen + 1}_{timestamp}"
    sample_thetas = getattr(best_individual, "stored_thetas", [])

    qc_draw = build_quantum_circuit(
        best_individual, circuit_qubits, input_values, sample_thetas
    )
    qc_final = build_quantum_circuit(
        best_individual, circuit_qubits, input_values, sample_thetas, measure=True
    )

    qc_draw.draw(output="mpl", filename=str(output_dir / f"{output_stem}.pdf"))

    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    generations_axis = logbook.select("gen")
    history_ar = logbook.select("best_avg_ar")
    history_depth = logbook.select("best_depth")

    output_data = {
        "config": {
            "approach": APPROACH,
            "config_file": str(CONFIG_PATH),
            "instance_evaluated": filename,
            "instance_num_qubits": instance_num_qubits,
            "circuit_qubits": circuit_qubits,
            "max_qubits": MAX_QUBITS,
            "generations_configured": generations,
            "generations_run": last_gen + 1,
            "mu": mu,
            "lambda": lambda_,
            "population": population_config,
            "variation": variation_config,
            "evolution": evolution_config,
            "gamma_schedule": gamma_config,
            "evaluation": evaluation_config,
        },
        "results": {
            "best_quantum_approximation_ratio": float(best_validation_avg_ar),
            "best_individual_depth": int(qc_draw.depth()),
            "optimized_parameters": sample_thetas,
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
        f"[SERVER INFO] Data successfully saved in '{output_dir / f'{output_stem}.json'}'"
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Approx Ratio", color=color)
    ax1.plot(generations_axis, history_ar, color=color, linewidth=2, label="AR")
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

    plt.title(f"QNAS Evolutionary Dynamics ({APPROACH.upper()}): AR vs Depth")
    fig.tight_layout()
    plt.savefig(output_dir / f"{output_stem}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
