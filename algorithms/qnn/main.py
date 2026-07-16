from __future__ import annotations

import copy
import functools
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools
from qiskit import qpy

if not hasattr(creator, "MultiFitness"):
    creator.create("MultiFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "MultiIndividual"):
    creator.create("MultiIndividual", list, fitness=creator.MultiFitness)

toolbox: base.Toolbox = base.Toolbox()


def _init_worker(dataset_name: str, approach: str) -> None:
    from algorithms.qnn.config import init_config

    init_config(dataset_name, approach)


def evaluate_population(
    individuals, toolbox, champion_weights=None
) -> Tuple[int, float]:
    from algorithms.qnn.config import CONFIG
    from algorithms.qnn.constants import WEIGHT_INHERITANCE_NOISE_STD, WEIGHT_INHERITANCE_MIN_PROB

    invalid = [ind for ind in individuals if not ind.fitness.valid]
    if not invalid:
        return 0, 0.0

    inheritance_prob = CONFIG["evolution"].get("champion_inheritance_prob", 0.15)
    inheritance_prob = max(inheritance_prob, WEIGHT_INHERITANCE_MIN_PROB)

    for ind in invalid:
        own_prior = getattr(ind, "stored_thetas", None)
        if own_prior:
            ind._seed_weights = {
                k: v + random.gauss(0, WEIGHT_INHERITANCE_NOISE_STD)
                for k, v in own_prior.items()
            }
        elif champion_weights is not None and random.random() < inheritance_prob:
            noise = {
                k: v + random.gauss(0, WEIGHT_INHERITANCE_NOISE_STD)
                for k, v in champion_weights.items()
            }
            ind._seed_weights = noise

    results = toolbox.map(toolbox.evaluate, invalid)
    per_individual_simulation_seconds = []
    for ind, (fit, weights, readout_data, simulation_seconds) in zip(invalid, results):
        ind.fitness.values = fit
        ind.stored_thetas = weights
        if readout_data is not None:
            ind._readout_clf = readout_data
        if hasattr(ind, "_seed_weights"):
            del ind._seed_weights
        per_individual_simulation_seconds.append(simulation_seconds)
    batch_simulation_seconds = (
        max(per_individual_simulation_seconds)
        if getattr(toolbox, "parallel_evaluation", False)
        else sum(per_individual_simulation_seconds)
    )
    return len(invalid), batch_simulation_seconds


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    import argparse as _argparse

    _main_parser = _argparse.ArgumentParser()
    _main_parser.add_argument("--approach", type=str, default=None)
    _main_parser.add_argument("--dataset", type=str, default=None)
    _main_args, _ = _main_parser.parse_known_args()

    _dataset = _main_args.dataset
    _approach = _main_args.approach
    if _dataset is None:
        _default_cfg_path = Path(__file__).parent / "config.json"
        with open(_default_cfg_path, encoding="utf-8") as _f:
            _default_cfg = json.load(_f)
        _dataset = _default_cfg.get("qnn", {}).get("dataset", "iris")
        if _approach is None:
            _approach = _default_cfg.get("approach", "rotation")

    from algorithms.qnn import config as qnn_config
    from algorithms.qnn.constants import (
        N_IMMIGRANTS_DIVISOR,
        N_IMMIGRANTS_MIN,
        CHAMPION_IMMIGRANT_PROB,
        MUTATION_ADAPT_PATIENCE_DIVISOR,
        MUTATION_ADAPT_SCALE,
        DIVERSITY_UNIQUE_FIT_ROUND,
        RESET_MUTATED_RATIO,
    )
    from algorithms.qnn.plotting import plot_evolution_progress
    from algorithms.qnn.qnn_data import load_qnn_data
    from algorithms.qnn.utils import (
        cx_quantum_circuit,
        describe_architecture,
        evaluate_circuit,
        serialize_architecture,
        validate_circuit,
        _effective_depth,
    )

    qnn_config.init_config(_dataset, approach=_approach)

    build_quantum_circuit = qnn_config.build_quantum_circuit
    generate_guided_individual = qnn_config.generate_guided_individual
    mut_quantum_circuit = qnn_config.mut_quantum_circuit

    if not qnn_config.CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {qnn_config.CONFIG_PATH}")
        return

    dataset_name = qnn_config.DATASET_NAME
    encoding_mode = "clifford_angle" if qnn_config.APPROACH == "clifford" else "amplitude"
    X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test, dataset_info = (
        load_qnn_data(
            dataset_name,
            test_split=qnn_config.TEST_SPLIT,
            val_split=qnn_config.VAL_SPLIT,
            random_state=42,
            encoding_mode=encoding_mode,
        )
    )

    circuit_qubits = dataset_info["n_qubits"]
    n_classes = dataset_info["n_classes"]

    training_data = list(zip(X_train_enc, y_train))

    population_config = qnn_config.CONFIG["population"]
    variation_config = qnn_config.CONFIG["variation"]
    evolution_config = qnn_config.CONFIG["evolution"]
    execution_config = qnn_config.CONFIG.get("execution", {})
    MAX_CHAMPION_RESETS = evolution_config.get("max_champion_resets", 2)
    champion_reset_count = 0

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
        max_params=qnn_config.NUM_PARAMS,
        enable_input_params=qnn_config.ENABLE_INPUT_PARAMS,
        param_block_prob=qnn_config.PARAM_BLOCK_PROB,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    crossover_indpb = variation_config.get("crossover_indpb", 0.5)
    toolbox.register(
        "mate", cx_quantum_circuit, num_qubits=circuit_qubits, indpb=crossover_indpb
    )
    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=circuit_qubits,
        indpb=current_indpb,
        max_params=qnn_config.NUM_PARAMS,
        enable_input_params=qnn_config.ENABLE_INPUT_PARAMS,
        param_block_prob=qnn_config.PARAM_BLOCK_PROB,
    )
    toolbox.register("select", tools.selNSGA2)

    use_multiprocessing = execution_config.get("multiprocessing", True)
    requested_processes = execution_config.get("processes")
    if not requested_processes:
        if qnn_config.SIMULATOR_DEVICE == "GPU":
            requested_processes = 2
        else:
            cpu_total = os.cpu_count() or 1
            requested_processes = max(1, cpu_total - 1)

    pool = (
        Pool(
            processes=requested_processes,
            initializer=_init_worker,
            initargs=(_dataset, _approach),
        )
        if use_multiprocessing
        else None
    )
    toolbox.parallel_evaluation = pool is not None
    toolbox.register("map", pool.map if pool is not None else map)

    print("\n--- QNN RUN CONFIGURATION ---")
    print(
        f"Dataset: {dataset_name} ({dataset_info['n_features']} features -> {circuit_qubits} qubits, {n_classes} classes)"
    )
    print(
        f"Training: {dataset_info['n_train']} | Validation: {dataset_info['n_val']} | Test: {dataset_info['n_test']}"
    )
    print(
        f"Approach: {qnn_config.APPROACH.upper()} | Processes: {requested_processes if pool is not None else 1}"
    )
    sim_method = "stabilizer" if qnn_config.APPROACH == "clifford" else "statevector"
    print(f"Simulator: {sim_method} on {qnn_config.SIMULATOR_DEVICE}\n")

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
    logbook.header = [
        "gen",
        "val_acc",
        "champion_acc",
        "best_depth",
        "champion_depth",
        "wall_seconds",
        "simulation_seconds",
    ]

    stats_acc = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(acc=stats_acc, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    champion_val_acc_ever = 0.0
    champion_depth_ever = 0.0
    absolute_champion = None
    champion_thetas = {}
    stagnant_generations = 0
    champion_check_k = evolution_config.get("champion_check_k", 5)
    patience_window = evolution_config.get("patience_window", max(1, patience // 4))
    recent_val_accs = deque(maxlen=patience_window)
    champion_smoothed_ever = 0.0
    run_start_wall = time.perf_counter()
    total_simulation_seconds = 0.0

    toolbox.register(
        "evaluate",
        functools.partial(
            evaluate_circuit,
            num_qubits=circuit_qubits,
            instances=training_data,
            n_classes=n_classes,
            X_val=X_val_enc,
            y_val=y_val,
        ),
    )
    print("Evaluando población inicial...")
    evaluate_population(population, toolbox, champion_weights=None)

    for gen in range(generations):
        gen_start_wall = time.perf_counter()
        gen_simulation_seconds = 0.0

        toolbox.register(
            "evaluate",
            functools.partial(
                evaluate_circuit,
                num_qubits=circuit_qubits,
                instances=training_data,
                n_classes=n_classes,
                X_val=X_val_enc,
                y_val=y_val,
            ),
        )

        offspring = algorithms.varOr(
            population, toolbox, lambda_, crossover_prob, mutation_prob
        )
        _, seconds_spent = evaluate_population(
            offspring,
            toolbox,
            champion_weights=champion_thetas if absolute_champion else None,
        )
        gen_simulation_seconds += seconds_spent

        if stagnant_generations > patience // 4:
            n_immigrants = max(N_IMMIGRANTS_MIN, mu // N_IMMIGRANTS_DIVISOR)
            immigrants = [
                creator.MultiIndividual(toolbox.individual())
                for _ in range(n_immigrants)
            ]
            if absolute_champion is not None and random.random() < CHAMPION_IMMIGRANT_PROB:
                champion_copy = toolbox.clone(absolute_champion)
                mut_quantum_circuit(
                    champion_copy,
                    circuit_qubits,
                    0.5,
                    max_params=qnn_config.NUM_PARAMS,
                    enable_input_params=qnn_config.ENABLE_INPUT_PARAMS,
                    param_block_prob=qnn_config.PARAM_BLOCK_PROB,
                )
                del champion_copy.fitness.values
                immigrants.append(champion_copy)
            _, immigrant_seconds = evaluate_population(
                immigrants, toolbox, champion_weights=None
            )
            gen_simulation_seconds += immigrant_seconds
            offspring.extend(immigrants)

        population[:] = toolbox.select(population + offspring, mu)

        if absolute_champion is not None:
            champion_clone = toolbox.clone(absolute_champion)
            if champion_clone not in population:
                population[-1] = champion_clone

        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]
        train_ranked = sorted(pareto_front, key=lambda ind: ind.fitness.values[0])
        top_k = train_ranked[:champion_check_k]
        best_val_acc = -1.0
        best_val_ind = None
        if toolbox.parallel_evaluation:
            validate_fn = functools.partial(
                validate_circuit,
                num_qubits=circuit_qubits,
                X_val=X_val_enc,
                y_val=y_val,
                n_classes=n_classes,
            )
            results = toolbox.map(validate_fn, top_k)
            for val_ind, (val_acc, _) in zip(top_k, results):
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_ind = val_ind
        else:
            for val_ind in top_k:
                val_acc, _ = validate_circuit(
                    val_ind,
                    circuit_qubits,
                    X_val_enc,
                    y_val,
                    n_classes,
                    seed_weights=getattr(val_ind, "stored_thetas", {}),
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_ind = val_ind

        true_val_acc = best_val_acc
        best_individual = best_val_ind or train_ranked[0]
        best_soft_dyn = -best_individual.fitness.values[0]
        best_depth = best_individual.fitness.values[1]
        record = statistics.compile(population)
        gen_wall_seconds = time.perf_counter() - gen_start_wall
        total_simulation_seconds += gen_simulation_seconds

        recent_val_accs.append(true_val_acc)

        logbook.record(
            gen=gen,
            train_soft=best_soft_dyn,
            best_acc=true_val_acc,
            champion_acc=champion_val_acc_ever,
            best_depth=best_depth,
            champion_depth=champion_depth_ever,
            wall_seconds=gen_wall_seconds,
            simulation_seconds=gen_simulation_seconds,
            **record,
        )

        sim_share_pct = (
            (gen_simulation_seconds / gen_wall_seconds) * 100.0
            if gen_wall_seconds > 0
            else 0.0
        )
        print(
            f"Gen {gen:>3d} | Train = {best_soft_dyn:.4f} | Val  = {true_val_acc:.4f} | "
            f"Champ = {champion_val_acc_ever:.4f} | Depth = {best_depth:3.0f} | "
            f"Wall = {gen_wall_seconds:6.2f}s | Sim = {gen_simulation_seconds:6.2f}s ({sim_share_pct:5.1f}%)"
        )

        last_gen = gen

        smoothed_val = np.mean(recent_val_accs)
        progress_detected = smoothed_val > champion_smoothed_ever + improvement_epsilon

        if true_val_acc > champion_val_acc_ever + improvement_epsilon:
            champion_val_acc_ever = true_val_acc
            champion_depth_ever = best_depth
            absolute_champion = toolbox.clone(best_individual)
            champion_thetas = copy.deepcopy(
                getattr(best_individual, "stored_thetas", {})
            )
            if qnn_config.APPROACH == "clifford":
                from algorithms.qnn.utils import update_hof

                for block_gene in best_individual:
                    if block_gene[0] == "PARAM_BLOCK":
                        update_hof(block_gene[2])
        elif (
            abs(true_val_acc - champion_val_acc_ever) <= improvement_epsilon
            and best_depth < champion_depth_ever
        ):
            champion_depth_ever = best_depth
            absolute_champion = toolbox.clone(best_individual)
            champion_thetas = copy.deepcopy(
                getattr(best_individual, "stored_thetas", {})
            )
            if qnn_config.APPROACH == "clifford":
                from algorithms.qnn.utils import update_hof

                for block_gene in best_individual:
                    if block_gene[0] == "PARAM_BLOCK":
                        update_hof(block_gene[2])

        if progress_detected:
            champion_smoothed_ever = smoothed_val
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= patience:
            print(f"[INFO] Early stopping en generación {gen} (paciencia={patience})")
            break

        if stagnant_generations > patience // MUTATION_ADAPT_PATIENCE_DIVISOR:
            current_indpb = min(
                0.7,
                base_indpb * (1.0 + MUTATION_ADAPT_SCALE * stagnant_generations / patience),
            )
        else:
            current_indpb = base_indpb

        toolbox.register(
            "mutate",
            mut_quantum_circuit,
            num_qubits=circuit_qubits,
            indpb=current_indpb,
            max_params=qnn_config.NUM_PARAMS,
            enable_input_params=qnn_config.ENABLE_INPUT_PARAMS,
            param_block_prob=qnn_config.PARAM_BLOCK_PROB,
        )

        if stagnant_generations > patience // 2 and absolute_champion is not None:
            unique_fitness = {
                tuple(round(v, DIVERSITY_UNIQUE_FIT_ROUND) for v in ind.fitness.values)
                for ind in population
            }
            low_diversity = len(unique_fitness) < mu // 4
            near_timeout = stagnant_generations >= patience - patience_window

            if (
                low_diversity or near_timeout
            ) and champion_reset_count < MAX_CHAMPION_RESETS:
                reason = (
                    "Baja diversidad"
                    if low_diversity
                    else "Cerca del límite de paciencia"
                )
                champion_reset_count += 1
                print(
                    f"[DIVERSITY] {reason} ({len(unique_fitness)} fit únicos). "
                    f"Reiniciando población alrededor del campeón "
                    f"(gen {gen}, intento {champion_reset_count}/{MAX_CHAMPION_RESETS})"
                )
                keeper = toolbox.clone(absolute_champion)
                new_pop = [keeper]
                n_mutated = int((mu - 1) * RESET_MUTATED_RATIO)
                n_fresh = (mu - 1) - n_mutated
                for _ in range(n_mutated):
                    ind = toolbox.clone(absolute_champion)
                    mut_quantum_circuit(
                        ind,
                        circuit_qubits,
                        current_indpb,
                        max_params=qnn_config.NUM_PARAMS,
                        enable_input_params=qnn_config.ENABLE_INPUT_PARAMS,
                        param_block_prob=qnn_config.PARAM_BLOCK_PROB,
                    )
                    del ind.fitness.values
                    new_pop.append(ind)
                for _ in range(n_fresh):
                    ind = creator.MultiIndividual(toolbox.individual())
                    new_pop.append(ind)
                _, _ = evaluate_population(new_pop[1:], toolbox, champion_weights=None)
                population[:] = new_pop
                stagnant_generations = 0

    total_wall_seconds = time.perf_counter() - run_start_wall
    avg_wall_per_gen = total_wall_seconds / (last_gen + 1)
    avg_simulation_per_gen = total_simulation_seconds / (last_gen + 1)

    total_sim_share_pct = (
        (total_simulation_seconds / total_wall_seconds) * 100.0
        if total_wall_seconds > 0
        else 0.0
    )
    print(
        f"\n[TIMING] Total: Wall = {total_wall_seconds:.2f}s | Sim = {total_simulation_seconds:.2f}s ({total_sim_share_pct:.1f}%) sobre {last_gen + 1} gens"
    )
    print(
        f"[TIMING] Promedio/gen: Wall = {avg_wall_per_gen:.2f}s | Sim = {avg_simulation_per_gen:.2f}s\n"
    )

    if pool is not None:
        pool.close()
        pool.join()

    if absolute_champion is None:
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]
        absolute_champion = pareto_front[0]
        champion_thetas = getattr(absolute_champion, "stored_thetas", {})

    final_val_acc, final_val_soft = validate_circuit(
        absolute_champion,
        circuit_qubits,
        X_val_enc,
        y_val,
        n_classes,
        seed_weights=champion_thetas,
    )
    final_test_acc, final_test_soft = validate_circuit(
        absolute_champion,
        circuit_qubits,
        X_test_enc,
        y_test,
        n_classes,
        seed_weights=champion_thetas,
    )

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = project_root / "results" / "qnn" / dataset_name / qnn_config.APPROACH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        f"{dataset_name}_{circuit_qubits}q_{qnn_config.APPROACH}_g{last_gen + 1}_{timestamp}"
    )

    qc_draw = build_quantum_circuit(
        absolute_champion, circuit_qubits, qnn_config.MANUAL_INPUT_VALUES, champion_thetas
    )
    qc_final = build_quantum_circuit(
        absolute_champion,
        circuit_qubits,
        qnn_config.MANUAL_INPUT_VALUES,
        champion_thetas,
        measure=True,
    )

    qc_draw.draw(output="mpl", filename=str(output_dir / f"{output_stem}.pdf"))
    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    architecture = describe_architecture(absolute_champion)
    if any(architecture.values()):
        with open(
            output_dir / f"{output_stem}_architecture.json", "w", encoding="utf-8"
        ) as f:
            json.dump(architecture, f, indent=4)

    genotype_payload = {
        "approach": qnn_config.APPROACH,
        "num_qubits": circuit_qubits,
        "num_params": qnn_config.NUM_PARAMS,
        "manual_input_values": qnn_config.MANUAL_INPUT_VALUES,
        "weights": champion_thetas,
        "genes": serialize_architecture(absolute_champion),
    }
    with open(output_dir / f"{output_stem}_genotype.json", "w", encoding="utf-8") as f:
        json.dump(genotype_payload, f, indent=4)

    generations_axis = logbook.select("gen")
    history_train_soft = logbook.select("train_soft")
    history_acc = logbook.select("best_acc")
    history_champion_acc = logbook.select("champion_acc")
    history_depth = logbook.select("best_depth")
    history_champion_depth = logbook.select("champion_depth")
    history_wall_seconds = logbook.select("wall_seconds")
    history_simulation_seconds = logbook.select("simulation_seconds")

    output_data = {
        "config": {
            "approach": qnn_config.APPROACH,
            "config_file": str(qnn_config.CONFIG_PATH),
            "dataset": dataset_name,
            "encoding_mode": encoding_mode,
            "circuit_qubits": circuit_qubits,
            "n_classes": n_classes,
            "simulator_device": qnn_config.SIMULATOR_DEVICE,
            "training_samples": dataset_info["n_train"],
            "validation_samples": dataset_info["n_val"],
            "test_samples": dataset_info["n_test"],
            "generations_run": last_gen + 1,
            "mu": mu,
            "lambda": lambda_,
        },
        "results": {
            "final_validation_accuracy": float(final_val_acc),
            "final_test_accuracy": float(final_test_acc),
            "best_individual_depth": int(_effective_depth(qc_draw)),
            "optimized_parameters": champion_thetas,
        },
        "timing": {
            "total_wall_seconds": float(total_wall_seconds),
            "total_simulation_seconds": float(total_simulation_seconds),
            "avg_wall_seconds_per_generation": float(avg_wall_per_gen),
            "avg_simulation_seconds_per_generation": float(avg_simulation_per_gen),
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "train_soft_score": [float(s) for s in history_train_soft],
            "best_accuracy": [float(c) for c in history_acc],
            "champion_accuracy": [float(c) for c in history_champion_acc],
            "best_depth": [float(d) for d in history_depth],
            "champion_depth": [float(d) for d in history_champion_depth],
            "wall_seconds": [float(w) for w in history_wall_seconds],
            "simulation_seconds": [float(s) for s in history_simulation_seconds],
        },
    }

    with open(output_dir / f"{output_stem}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\n[VALIDATION] Val Acc Final: {final_val_acc:.4f} (Soft: {final_val_soft:.4f}) | Test Acc: {final_test_acc:.4f} (Soft: {final_test_soft:.4f})"
    )
    print(f"[SAVED] {output_dir / f'{output_stem}.json'}")

    best_val_champion = max(history_champion_acc) if history_champion_acc else 0
    best_depth_champion = min(history_champion_depth) if history_champion_depth else 0
    best_train = max(history_train_soft) if history_train_soft else 0
    best_val = max(history_acc) if history_acc else 0
    plot_evolution_progress(
        generations=generations_axis,
        train_soft=history_train_soft,
        val_acc=history_acc,
        val_champion=history_champion_acc,
        depth_per_gen=history_depth,
        depth_champion=history_champion_depth,
        val_champion_best=best_val_champion,
        depth_champion_best=best_depth_champion,
        train_best=best_train,
        val_best=best_val,
        approach=qnn_config.APPROACH,
        output_stem=output_stem,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse as _argparse

    _run_parser = _argparse.ArgumentParser()
    _run_parser.add_argument("--approach", type=str, default=None)
    _run_parser.add_argument("--dataset", type=str, default=None)
    _run_args, _ = _run_parser.parse_known_args()

    _config_path = Path(__file__).parent / "config.json"
    with open(_config_path, encoding="utf-8") as _f:
        _config = json.load(_f)

    _approaches = _config.get("approaches", None)
    _datasets = _config.get("qnn", {}).get("datasets_to_run", None)

    if _run_args.dataset == "all":
        _configs_dir = Path(__file__).parent / "configs"
        _datasets = sorted([f.stem for f in _configs_dir.glob("*.json")])
        _run_args.dataset = None

    if _run_args.approach == "all":
        _approaches = ["clifford", "rotation"]
        _run_args.approach = None

    _explicit_dataset = _run_args.dataset is not None
    _explicit_approach = _run_args.approach is not None

    if _run_args.approach == "both":
        _dataset_to_use = _run_args.dataset or _config.get("qnn", {}).get(
            "dataset", "iris"
        )
        for _a in ["clifford", "rotation"]:
            print(f"\n{'=' * 60}")
            print(f"  Dataset: {_dataset_to_use} | Approach: {_a}")
            print(f"{'=' * 60}\n")
            _result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--internal",
                    "--dataset",
                    _dataset_to_use,
                    "--approach",
                    _a,
                ]
            )
            if _result.returncode != 0:
                print(
                    f"[ERROR] Dataset {_dataset_to_use} / Approach {_a} falló (código {_result.returncode})"
                )
    else:
        _run_multi = (
            not _explicit_dataset
            and not _explicit_approach
            and (
                (_approaches and len(_approaches) > 1)
                or (_datasets and len(_datasets) > 1)
            )
            and "--internal" not in sys.argv
        )

        if _run_multi:
            _orig_approach = _config.get("approach", (_approaches or ["clifford"])[0])
            _orig_dataset = _config.get("qnn", {}).get(
                "dataset", (_datasets or ["iris"])[0]
            )

            _dataset_list = _datasets if _datasets else [_orig_dataset]
            _approach_list = _approaches if _approaches else [_orig_approach]

            for _ds in _dataset_list:
                for _a in _approach_list:
                    print(f"\n{'=' * 60}")
                    print(f"  Dataset: {_ds} | Approach: {_a}")
                    print(f"{'=' * 60}\n")
                    _result = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--internal",
                            "--dataset",
                            _ds,
                            "--approach",
                            _a,
                        ]
                    )
                    if _result.returncode != 0:
                        print(
                            f"[ERROR] Dataset {_ds} / Approach {_a} falló (código {_result.returncode})"
                        )
        else:
            main()
