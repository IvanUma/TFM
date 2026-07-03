from __future__ import annotations

import copy
import functools
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple

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


def cpu_seconds_snapshot() -> float:
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


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


def load_instance_set(
    instances_dir: Path,
    max_qubits: int,
    load_external_maxcut_instance,
    build_universal_input_values,
    instance_qubits_filter=None,
) -> List[Tuple[str, object, float, List[float]]]:
    instance_files = sorted(p for p in instances_dir.iterdir() if p.is_file())
    loaded = []
    skipped = 0

    for path in instance_files:
        graph, num_nodes, optimal_cut = load_external_maxcut_instance(str(path))

        if instance_qubits_filter is not None and num_nodes != instance_qubits_filter:
            skipped += 1
            continue

        if num_nodes > max_qubits:
            raise ValueError(
                f"Instance {path.name} requires {num_nodes} qubits but "
                f"circuit_scale.max_qubits is {max_qubits}"
            )

        input_values = build_universal_input_values(graph, max_qubits)
        loaded.append((path.name, graph, optimal_cut, input_values))

    if instance_qubits_filter is not None:
        print(
            f"[INFO] Filtered to {len(loaded)} instances with "
            f"{instance_qubits_filter} nodes ({skipped} skipped)"
        )

    return loaded


def split_instances(
    instances: List[Tuple[str, object, float, List[float]]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    shuffled = list(instances)
    rng.shuffle(shuffled)

    if len(shuffled) <= 1:
        return shuffled, []

    num_validation = max(1, int(len(shuffled) * validation_fraction))
    validation = shuffled[:num_validation]
    training = shuffled[num_validation:]

    if not training:
        training = shuffled
        validation = []

    return training, validation


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from algorithms.general_max_cut.utils import (
        APPROACH,
        CONFIG,
        CONFIG_PATH,
        ENABLE_INPUT_PARAMS,
        INSTANCE_QUBITS_FILTER,
        MAX_QUBITS,
        PARAM_BLOCK_PROB,
        SPLIT_SEED,
        VALIDATION_FRACTION,
        EvolutionaryIndividual,
        build_quantum_circuit,
        build_universal_input_values,
        cx_quantum_circuit,
        describe_architecture,
        evaluate_circuit,
        generate_heuristic_individual,
        generate_guided_individual,
        load_external_maxcut_instance,
        max_cut_fitness,
        mut_quantum_circuit,
        update_hof,
    )

    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        return

    instances_dir = project_root / "max_cut_instances"

    if not instances_dir.is_dir():
        print(f"[ERROR] Instances directory not found: {instances_dir}")
        return

    try:
        all_instances = load_instance_set(
            instances_dir,
            MAX_QUBITS,
            load_external_maxcut_instance,
            build_universal_input_values,
            instance_qubits_filter=INSTANCE_QUBITS_FILTER,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    if not all_instances:
        print(f"[ERROR] No instance files found in {instances_dir}")
        return

    training_instances, validation_instances = split_instances(
        all_instances, VALIDATION_FRACTION, SPLIT_SEED
    )

    circuit_qubits = MAX_QUBITS
    training_data = [(graph, opt, inp) for _, graph, opt, inp in training_instances]
    validation_data = validation_instances
    run_label = instances_dir.name
    filename = instances_dir.name

    population_config = CONFIG["population"]
    variation_config = CONFIG["variation"]
    evolution_config = CONFIG["evolution"]
    gamma_config = CONFIG["gamma_schedule"]
    evaluation_config = CONFIG["evaluation"]
    execution_config = CONFIG.get("execution", {})

    dynamic_max_params = len(training_data[0][2]) if training_data else 1

    toolbox.register("clone", copy.deepcopy)

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=circuit_qubits,
        length=max(
            evolution_config["guided_individual_length_min"],
            circuit_qubits * evolution_config["guided_individual_length_factor"],
        ),
        graph_instance=training_data[0][0],
        max_params=dynamic_max_params,
        enable_input_params=ENABLE_INPUT_PARAMS,
        param_block_prob=PARAM_BLOCK_PROB,
        max_qubits=MAX_QUBITS,
    )

    toolbox.register(
        "individual_heuristic",
        generate_heuristic_individual,
        num_qubits=circuit_qubits,
        graph_instance=training_data[0][0],
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_quantum_circuit, num_qubits=circuit_qubits)
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=circuit_qubits,
        graph_instance=training_data[0][0],
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

    print("\n--- RUN CONFIGURATION ---")
    print(f"Source: {filename}")
    print(f"Training instances: {len(training_data)}")
    print(f"Validation instances: {len(validation_data)}")
    print(f"Circuit qubits: {circuit_qubits}")
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
    logbook.header = [
        "gen",
        "shots",
        "gamma",
        "best_avg_ar",
        "best_depth",
        "wall_seconds",
        "cpu_seconds",
    ]

    stats_ar = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(ar=stats_ar, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    best_ar_ever = -1.0
    stagnant_generations = 0
    last_gen = 0

    run_start_wall = time.perf_counter()
    run_start_cpu = cpu_seconds_snapshot()

    for gen in range(generations):
        gen_start_wall = time.perf_counter()
        gen_start_cpu = cpu_seconds_snapshot()

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
                instances=training_data,
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

        gen_wall_seconds = time.perf_counter() - gen_start_wall
        gen_cpu_seconds = cpu_seconds_snapshot() - gen_start_cpu

        logbook.record(
            gen=gen,
            shots=current_shots,
            gamma=current_gamma,
            best_avg_ar=best_avg_ar,
            best_depth=best_depth,
            wall_seconds=gen_wall_seconds,
            cpu_seconds=gen_cpu_seconds,
            **record,
        )

        print(
            f"Gen {gen}: Approx Ratio = {best_avg_ar:.4f} | Depth = {best_depth:.1f} | "
            f"Wall = {gen_wall_seconds:.2f}s | CPU = {gen_cpu_seconds:.2f}s"
        )

        last_gen = gen

        if best_avg_ar > best_ar_ever + improvement_epsilon:
            best_ar_ever = best_avg_ar
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= patience:
            print(f"[INFO] Early stopping at generation {gen} (patience={patience})")
            break

    total_wall_seconds = time.perf_counter() - run_start_wall
    total_cpu_seconds = cpu_seconds_snapshot() - run_start_cpu
    avg_wall_per_gen = total_wall_seconds / (last_gen + 1)
    avg_cpu_per_gen = total_cpu_seconds / (last_gen + 1)

    print(
        f"\n[TIMING] Total: Wall = {total_wall_seconds:.2f}s | CPU = {total_cpu_seconds:.2f}s "
        f"over {last_gen + 1} generations"
    )
    print(
        f"[TIMING] Average per generation: Wall = {avg_wall_per_gen:.2f}s | "
        f"CPU = {avg_cpu_per_gen:.2f}s\n"
    )

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

    def training_instances_as_named(data):
        return [(f"training_{i}", g, o, inp) for i, (g, o, inp) in enumerate(data)]

    evaluation_set = (
        validation_data
        if validation_data
        else training_instances_as_named(training_data)
    )

    best_validation_avg_ar = -1.0
    best_individual = None
    per_individual_reports = []

    for ind in pareto_front:
        thetas = getattr(ind, "stored_thetas", [])
        circuits = [
            build_quantum_circuit(ind, circuit_qubits, inst_input, thetas, measure=True)
            for _, _, _, inst_input in evaluation_set
        ]
        results = simulator.run(
            circuits, shots=evaluation_config["final_validation_shots"]
        ).result()

        per_instance = []
        for idx, (name, graph, optimal_cut, _) in enumerate(evaluation_set):
            counts = results.get_counts(idx)
            cut = max_cut_fitness(counts, graph, alpha=1.0)
            ar = cut / optimal_cut if optimal_cut > 0 else 0.0
            per_instance.append({"instance": name, "approx_ratio": float(ar)})

        avg_ar = sum(item["approx_ratio"] for item in per_instance) / len(per_instance)

        if avg_ar > best_validation_avg_ar:
            best_validation_avg_ar = avg_ar
            best_individual = ind
            per_individual_reports = per_instance

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = project_root / "results" / "general_max_cut" / run_label / APPROACH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{run_label}_opt_{APPROACH}_g{last_gen + 1}_{timestamp}"
    sample_thetas = getattr(best_individual, "stored_thetas", [])
    example_input_values = evaluation_set[0][3]

    qc_draw = build_quantum_circuit(
        best_individual, circuit_qubits, example_input_values, sample_thetas
    )
    qc_final = build_quantum_circuit(
        best_individual,
        circuit_qubits,
        example_input_values,
        sample_thetas,
        measure=True,
    )

    qc_draw.draw(output="mpl", filename=str(output_dir / f"{output_stem}.pdf"))

    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    architecture = describe_architecture(best_individual)
    if any(architecture.values()):
        with open(
            output_dir / f"{output_stem}_architecture.json", "w", encoding="utf-8"
        ) as f:
            json.dump(architecture, f, indent=4)

    generations_axis = logbook.select("gen")
    history_ar = logbook.select("best_avg_ar")
    history_depth = logbook.select("best_depth")
    history_wall_seconds = logbook.select("wall_seconds")
    history_cpu_seconds = logbook.select("cpu_seconds")

    output_data = {
        "config": {
            "approach": APPROACH,
            "config_file": str(CONFIG_PATH),
            "source": filename,
            "circuit_qubits": circuit_qubits,
            "max_qubits": MAX_QUBITS,
            "training_instances": len(training_data),
            "validation_instances": len(validation_data),
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
            "best_validation_avg_approx_ratio": float(best_validation_avg_ar),
            "per_instance_validation": per_individual_reports,
            "best_individual_depth": int(qc_draw.depth()),
            "optimized_parameters": sample_thetas,
        },
        "timing": {
            "total_wall_seconds": float(total_wall_seconds),
            "total_cpu_seconds": float(total_cpu_seconds),
            "avg_wall_seconds_per_generation": float(avg_wall_per_gen),
            "avg_cpu_seconds_per_generation": float(avg_cpu_per_gen),
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "best_avg_ar": [float(c) for c in history_ar],
            "best_depth": [float(d) for d in history_depth],
            "wall_seconds": [float(w) for w in history_wall_seconds],
            "cpu_seconds": [float(c) for c in history_cpu_seconds],
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
