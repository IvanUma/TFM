from __future__ import annotations

import copy
import functools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
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


def evaluate_population(individuals, toolbox, champion_weights=None) -> Tuple[int, float]:
    invalid = [ind for ind in individuals if not ind.fitness.valid]
    if not invalid:
        return 0, 0.0
    for ind in invalid:
        seed = getattr(ind, "stored_thetas", None) or champion_weights
        ind._seed_weights = seed
    results = toolbox.map(toolbox.evaluate, invalid)
    per_individual_simulation_seconds = []
    for ind, (fit, weights, _, simulation_seconds) in zip(invalid, results):
        ind.fitness.values = fit
        ind.stored_thetas = weights
        del ind._seed_weights
        per_individual_simulation_seconds.append(simulation_seconds)
    batch_simulation_seconds = (
        max(per_individual_simulation_seconds) if getattr(toolbox, "parallel_evaluation", False)
        else sum(per_individual_simulation_seconds)
    )
    return len(invalid), batch_simulation_seconds


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from algorithms.qnn.utils import (
        APPROACH,
        CONFIG,
        CONFIG_PATH,
        DATASET_NAME,
        ENABLE_INPUT_PARAMS,
        MANUAL_INPUT_VALUES,
        NUM_PARAMS,
        PARAM_BLOCK_PROB,
        TEST_SPLIT,
        SIMULATOR_DEVICE,
        build_quantum_circuit,
        cx_quantum_circuit,
        describe_architecture,
        evaluate_circuit,
        generate_guided_individual,
        mut_quantum_circuit,
        serialize_architecture,
    )
    from algorithms.qnn.qnn_data import load_qnn_data

    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        return

    dataset_name = DATASET_NAME
    X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test, dataset_info = load_qnn_data(
        dataset_name, test_split=TEST_SPLIT, random_state=42
    )

    circuit_qubits = dataset_info["n_qubits"]
    n_classes = dataset_info["n_classes"]

    training_data = list(zip(X_train_enc, y_train))

    population_config = CONFIG["population"]
    variation_config = CONFIG["variation"]
    evolution_config = CONFIG["evolution"]
    evaluation_config = CONFIG["evaluation"]
    execution_config = CONFIG.get("execution", {})

    base_indpb = variation_config["mutation_indpb"]
    current_indpb = base_indpb

    toolbox.register("clone", copy.deepcopy)
    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=circuit_qubits,
        length=max(
            evolution_config["guided_individual_length_min"],
            circuit_qubits * evolution_config["guided_individual_length_factor"],
        ),
        max_params=NUM_PARAMS,
        enable_input_params=ENABLE_INPUT_PARAMS,
        param_block_prob=PARAM_BLOCK_PROB,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_quantum_circuit, num_qubits=circuit_qubits)
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=circuit_qubits,
        indpb=current_indpb,
        max_params=NUM_PARAMS,
        enable_input_params=ENABLE_INPUT_PARAMS,
        param_block_prob=PARAM_BLOCK_PROB,
    )
    toolbox.register("select", tools.selNSGA2)

    use_multiprocessing = execution_config.get("multiprocessing", True)
    requested_processes = execution_config.get("processes")
    if not requested_processes:
        if SIMULATOR_DEVICE == "GPU":
            requested_processes = 2
        else:
            cpu_total = os.cpu_count() or 1
            requested_processes = max(1, cpu_total - 1)

    pool = Pool(processes=requested_processes) if use_multiprocessing else None
    toolbox.parallel_evaluation = pool is not None
    toolbox.register("map", pool.map if pool is not None else map)

    print("\n--- QNN RUN CONFIGURATION ---")
    print(f"Dataset: {dataset_name} ({dataset_info['n_features']} features -> {circuit_qubits} qubits, {n_classes} classes)")
    print(f"Training samples: {dataset_info['n_train']} | Validation: {dataset_info['n_val']} | Test: {dataset_info['n_test']}")
    print(f"Input parameters: {NUM_PARAMS} {MANUAL_INPUT_VALUES}")
    print(f"Approach: {APPROACH.upper()}")
    print(f"Processes: {requested_processes if pool is not None else 1}")
    print(f"Simulator: statevector on {SIMULATOR_DEVICE}\n")

    mu = population_config["mu"]
    lambda_ = population_config["lambda"]
    population: List = []
    for _ in range(mu):
        population.append(creator.MultiIndividual(toolbox.individual()))

    crossover_prob = variation_config["crossover_prob"]
    mutation_prob = variation_config["mutation_prob"]
    generations = evolution_config["generations"]
    patience = evolution_config.get("patience", generations)
    improvement_epsilon = evolution_config.get("improvement_epsilon", 0.0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "shots", "val_acc", "best_depth", "wall_seconds", "cpu_seconds", "simulation_seconds"]

    stats_acc = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(acc=stats_acc, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    best_train_acc_ever = -1.0
    absolute_champion = None
    champion_thetas = {}
    stagnant_generations = 0
    run_start_wall = time.perf_counter()
    run_start_cpu = cpu_seconds_snapshot()
    total_simulation_seconds = 0.0

    for gen in range(generations):
        gen_start_wall = time.perf_counter()
        gen_start_cpu = cpu_seconds_snapshot()
        gen_simulation_seconds = 0.0

        progress = gen / (generations - 1) if generations > 1 else 1.0
        current_shots = int(
            evaluation_config["shots_start"]
            + (evaluation_config["shots_end"] - evaluation_config["shots_start"]) * progress
        )

        toolbox.register(
            "evaluate",
            functools.partial(
                evaluate_circuit,
                num_qubits=circuit_qubits,
                instances=training_data,
                shots=current_shots,
                n_classes=n_classes,
                X_val=X_val_enc,
                y_val=y_val,
            ),
        )

        if gen == 0:
            _, seconds_spent = evaluate_population(population, toolbox, champion_weights=None)
            gen_simulation_seconds += seconds_spent

        offspring = algorithms.varOr(
            population, toolbox, lambda_, crossover_prob, mutation_prob
        )
        _, seconds_spent = evaluate_population(
            offspring, toolbox, champion_weights=champion_thetas if absolute_champion else None,
        )
        gen_simulation_seconds += seconds_spent
        population[:] = toolbox.select(population + offspring, mu)

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        best_individual = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        best_acc = -best_individual.fitness.values[0]
        best_depth = best_individual.fitness.values[1]
        record = statistics.compile(population)

        gen_wall_seconds = time.perf_counter() - gen_start_wall
        gen_cpu_seconds = cpu_seconds_snapshot() - gen_start_cpu
        total_simulation_seconds += gen_simulation_seconds

        logbook.record(
            gen=gen, shots=current_shots, best_acc=best_acc, best_depth=best_depth,
            wall_seconds=gen_wall_seconds, cpu_seconds=gen_cpu_seconds, simulation_seconds=gen_simulation_seconds, **record,
        )

        sim_share_pct = (gen_simulation_seconds / gen_wall_seconds) * 100.0 if gen_wall_seconds > 0 else 0.0
        print(f"Gen {gen}: Val Acc = {best_acc:.4f} | Depth = {best_depth:.1f} | Wall = {gen_wall_seconds:.2f}s | Sim = {gen_simulation_seconds:.2f}s ({sim_share_pct:.1f}% of wall)")

        last_gen = gen

        if best_acc > best_train_acc_ever + improvement_epsilon:
            best_train_acc_ever = best_acc
            stagnant_generations = 0
            absolute_champion = toolbox.clone(best_individual)
            champion_thetas = copy.deepcopy(getattr(best_individual, "stored_thetas", {}))
        else:
            stagnant_generations += 1

        if stagnant_generations >= patience:
            print(f"[INFO] Early stopping at generation {gen} (patience={patience})")
            break

        if stagnant_generations > patience // 3:
            current_indpb = min(0.3, base_indpb * (1.0 + 0.5 * stagnant_generations / patience))
        else:
            current_indpb = base_indpb
        toolbox.register(
            "mutate",
            mut_quantum_circuit,
            num_qubits=circuit_qubits,
            indpb=current_indpb,
            max_params=NUM_PARAMS,
            enable_input_params=ENABLE_INPUT_PARAMS,
            param_block_prob=PARAM_BLOCK_PROB,
        )

    total_wall_seconds = time.perf_counter() - run_start_wall
    total_cpu_seconds = cpu_seconds_snapshot() - run_start_cpu
    avg_wall_per_gen = total_wall_seconds / (last_gen + 1)
    avg_cpu_per_gen = total_cpu_seconds / (last_gen + 1)
    avg_simulation_per_gen = total_simulation_seconds / (last_gen + 1)

    total_sim_share_pct = (total_simulation_seconds / total_wall_seconds) * 100.0 if total_wall_seconds > 0 else 0.0
    print(f"\n[TIMING] Total: Wall = {total_wall_seconds:.2f}s | Sim = {total_simulation_seconds:.2f}s ({total_sim_share_pct:.1f}%) over {last_gen + 1} gens")
    print(f"[TIMING] Avg/gen: Wall = {avg_wall_per_gen:.2f}s | Sim = {avg_simulation_per_gen:.2f}s\n")

    if pool is not None:
        pool.close()
        pool.join()

    if absolute_champion is None:
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        absolute_champion = pareto_front[0]
        champion_thetas = getattr(absolute_champion, "stored_thetas", {})

    final_shots = evaluation_config["final_validation_shots"]
    from algorithms.qnn.utils import validate_circuit
    final_val_acc = validate_circuit(
        absolute_champion, circuit_qubits, X_val_enc, y_val, final_shots, n_classes, seed_weights=champion_thetas,
    )
    final_test_acc = validate_circuit(
        absolute_champion, circuit_qubits, X_test_enc, y_test, final_shots, n_classes, seed_weights=champion_thetas,
    )

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = project_root / "results" / "qnn" / dataset_name / APPROACH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{dataset_name}_{circuit_qubits}q_{APPROACH}_g{last_gen + 1}_{timestamp}"

    qc_draw = build_quantum_circuit(absolute_champion, circuit_qubits, MANUAL_INPUT_VALUES, champion_thetas)
    qc_final = build_quantum_circuit(absolute_champion, circuit_qubits, MANUAL_INPUT_VALUES, champion_thetas, measure=True)

    qc_draw.draw(output="mpl", filename=str(output_dir / f"{output_stem}.pdf"))
    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    architecture = describe_architecture(absolute_champion)
    if any(architecture.values()):
        with open(output_dir / f"{output_stem}_architecture.json", "w", encoding="utf-8") as f:
            json.dump(architecture, f, indent=4)

    genotype_payload = {
        "approach": APPROACH,
        "num_qubits": circuit_qubits,
        "num_params": NUM_PARAMS,
        "manual_input_values": MANUAL_INPUT_VALUES,
        "weights": champion_thetas,
        "genes": serialize_architecture(absolute_champion),
    }
    with open(output_dir / f"{output_stem}_genotype.json", "w", encoding="utf-8") as f:
        json.dump(genotype_payload, f, indent=4)

    generations_axis = logbook.select("gen")
    history_acc = logbook.select("best_acc")
    history_depth = logbook.select("best_depth")
    history_wall_seconds = logbook.select("wall_seconds")
    history_cpu_seconds = logbook.select("cpu_seconds")
    history_simulation_seconds = logbook.select("simulation_seconds")

    output_data = {
        "config": {
            "approach": APPROACH,
            "config_file": str(CONFIG_PATH),
            "dataset": dataset_name,
            "circuit_qubits": circuit_qubits,
            "n_classes": n_classes,
            "num_params": NUM_PARAMS,
            "manual_input_values": MANUAL_INPUT_VALUES,
            "simulator_device": SIMULATOR_DEVICE,
            "training_samples": dataset_info["n_train"],
            "validation_samples": dataset_info["n_val"],
            "test_samples": dataset_info["n_test"],
            "generations_configured": generations,
            "generations_run": last_gen + 1,
            "mu": mu,
            "lambda": lambda_,
            "population": population_config,
            "variation": variation_config,
            "evolution": evolution_config,
            "evaluation": evaluation_config,
        },
        "results": {
            "final_validation_accuracy": float(final_val_acc),
            "final_test_accuracy": float(final_test_acc),
            "best_individual_depth": int(qc_draw.depth()),
            "optimized_parameters": champion_thetas,
        },
        "timing": {
            "total_wall_seconds": float(total_wall_seconds),
            "total_cpu_seconds": float(total_cpu_seconds),
            "total_simulation_seconds": float(total_simulation_seconds),
            "avg_wall_seconds_per_generation": float(avg_wall_per_gen),
            "avg_cpu_seconds_per_generation": float(avg_cpu_per_gen),
            "avg_simulation_seconds_per_generation": float(avg_simulation_per_gen),
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "best_accuracy": [float(c) for c in history_acc],
            "best_depth": [float(d) for d in history_depth],
            "wall_seconds": [float(w) for w in history_wall_seconds],
            "cpu_seconds": [float(c) for c in history_cpu_seconds],
            "simulation_seconds": [float(s) for s in history_simulation_seconds],
        },
    }

    with open(output_dir / f"{output_stem}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n[VALIDATION] Final val acc: {final_val_acc:.4f} | Test acc: {final_test_acc:.4f}")
    print(f"[SAVED] {output_dir / f'{output_stem}.json'}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Accuracy", color=color)
    ax1.plot(generations_axis, history_acc, color=color, linewidth=2, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Depth", color=color)
    ax2.plot(generations_axis, history_depth, color=color, linestyle="--", linewidth=2, label="Depth")
    ax2.tick_params(axis="y", labelcolor=color)
    plt.title(f"QNAS Evolutionary Dynamics ({APPROACH.upper()}): Accuracy vs Depth")
    fig.tight_layout()
    plt.savefig(output_dir / f"{output_stem}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--approach", type=str, default=None)
    _args, _ = _parser.parse_known_args()

    _config_path = Path(__file__).parent / "config.json"
    with open(_config_path) as _f:
        _config = json.load(_f)

    _approaches = _config.pop("approaches", None)

    if _approaches and len(_approaches) > 1 and "--internal" not in sys.argv:
        _orig = _config.get("approach", _approaches[0])
        for _a in _approaches:
            print(f"\n{'='*60}")
            print(f"  Running approach: {_a}")
            print(f"{'='*60}\n")
            _config["approach"] = _a
            with open(_config_path, "w") as _f:
                json.dump(_config, _f, indent=4)
            _result = subprocess.run([sys.executable, __file__, "--internal", "--approach", _a])
            if _result.returncode != 0:
                print(f"[ERROR] Approach {_a} failed (code {_result.returncode})")
        _config["approach"] = _orig
        with open(_config_path, "w") as _f:
            json.dump(_config, _f, indent=4)
    else:
        if _args.approach:
            with open(_config_path) as _f:
                _config_single = json.load(_f)
            _config_single["approach"] = _args.approach
            with open(_config_path, "w") as _f:
                json.dump(_config_single, _f, indent=4)
        main()