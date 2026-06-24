from __future__ import annotations

import functools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from qiskit import qpy
from qiskit.quantum_info import Statevector, SparsePauliOp

matplotlib.use("Agg")

if not hasattr(creator, "MultiFitness"):
    creator.create("MultiFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "MultiIndividual"):
    creator.create("MultiIndividual", list, fitness=creator.MultiFitness)

toolbox: base.Toolbox = base.Toolbox()


def evaluate_circuit_analytically(individual, num_qubits, graph, build_quantum_circuit):
    qc = build_quantum_circuit(individual, num_qubits, measure=False)
    state = Statevector.from_instruction(qc)
    total_energy = 0.0
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        paulis = ["I"] * num_qubits
        paulis[u] = "Z"
        paulis[v] = "Z"
        op = SparsePauliOp("".join(reversed(paulis)))
        total_energy += 0.5 * w * (1.0 - state.expectation_value(op).real)
    return -total_energy, float(qc.depth())


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    sys.path.insert(0, str(project_root))
    from algorithms.multi_objective_max_cut.max_cut_clifford import (
        EvolutionaryIndividual,
        build_quantum_circuit,
        cx_quantum_circuit,
        generate_heuristic_individual,
        generate_guided_individual,
        load_external_maxcut_instance,
        mut_quantum_circuit,
    )

    instance_name = input("Enter the instance filename: ").strip()
    if not instance_name:
        print("[ERROR] No file was specified.")
        return

    instance_path = Path(instance_name)
    if not instance_path.is_absolute():
        instance_path = project_root / "max_cut_instances" / instance_path

    file_path = str(instance_path)

    if not Path(file_path).exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    graph, num_qubits, optimal_classical_cut = load_external_maxcut_instance(file_path)

    toolbox.register(
        "individual",
        generate_guided_individual,
        num_qubits=num_qubits,
        length=max(20, num_qubits * 2),
        graph_instance=graph,
    )

    toolbox.register(
        "individual_heuristic",
        generate_heuristic_individual,
        num_qubits=num_qubits,
        graph_instance=graph,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cx_quantum_circuit, num_qubits=num_qubits)

    toolbox.register(
        "mutate",
        mut_quantum_circuit,
        num_qubits=num_qubits,
        graph_instance=graph,
        indpb=0.2,
    )

    toolbox.register("select", tools.selNSGA2)

    pool = Pool()
    toolbox.register("map", pool.map)

    print(f"Loaded External Instance: {file_path}")
    print(f"Nodes / Qubits detected: {num_qubits}")
    print(f"Target Optimal Classical MaxCut: {optimal_classical_cut} cuts\n")

    mu = 300
    lambda_ = 450

    population: List[EvolutionaryIndividual] = []

    for _ in range(int(mu * 0.2)):
        population.append(creator.MultiIndividual(toolbox.individual_heuristic()))

    for _ in range(mu - len(population)):
        population.append(creator.MultiIndividual(toolbox.individual()))

    crossover_prob = 0.6
    mutation_prob = 0.35
    generations = 350

    logbook = tools.Logbook()
    logbook.header = [
        "gen",
        "best_analytical_cut",
        "alpha",
        "best_depth",
    ]

    stats_cut = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])

    statistics = tools.MultiStatistics(cut=stats_cut, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    toolbox.register(
        "evaluate",
        functools.partial(
            evaluate_circuit_analytically,
            num_qubits=num_qubits,
            graph=graph,
            build_quantum_circuit=build_quantum_circuit,
        ),
    )

    for gen in range(generations):
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
            population,
            len(population),
            first_front_only=True,
        )[0]

        best_individual = min(
            pareto_front,
            key=lambda ind: ind.fitness.values[0],
        )

        best_cut = -best_individual.fitness.values[0]
        best_depth = best_individual.fitness.values[1]
        current_alpha = (
            best_cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0
        )

        record = statistics.compile(population)

        logbook.record(
            gen=gen,
            best_analytical_cut=best_cut,
            alpha=current_alpha,
            best_depth=best_depth,
            **record,
        )

        print(
            f"Gen {gen}: Exact Cut = {best_cut:.4f} | Alpha = {current_alpha:.4f} | Depth = {best_depth:.1f}"
        )

    pool.close()
    pool.join()

    pareto_front = tools.sortNondominated(
        population,
        len(population),
        first_front_only=True,
    )[0]

    best_quantum_cut = -1.0
    best_individual = None

    for ind in pareto_front:
        res_cut, _ = evaluate_circuit_analytically(
            ind, num_qubits, graph, build_quantum_circuit
        )
        cut = -res_cut
        if cut > best_quantum_cut:
            best_quantum_cut = cut
            best_individual = ind

    approximation_ratio = (
        best_quantum_cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0
    )

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    output_dir = project_root / "results" / Path(file_path).name / "analytical"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"{Path(file_path).name}_q{num_qubits}_g{generations}_{timestamp}"

    qc_draw = build_quantum_circuit(best_individual, num_qubits)
    qc_final = build_quantum_circuit(
        best_individual,
        num_qubits,
        measure=True,
    )

    qc_draw.draw(
        output="mpl",
        filename=str(output_dir / f"{output_stem}.pdf"),
    )

    with open(output_dir / f"{output_stem}.qpy", "wb") as f:
        qpy.dump(qc_final, f)

    generations_axis = logbook.select("gen")
    history_cut = logbook.select("best_analytical_cut")
    history_alpha = logbook.select("alpha")
    history_depth = logbook.select("best_depth")

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
            "best_individual_depth": int(qc_draw.depth()),
        },
        "history": {
            "generation": [int(g) for g in generations_axis],
            "best_exact_cut": [float(c) for c in history_cut],
            "alpha": [float(a) for a in history_alpha],
            "best_depth": [float(d) for d in history_depth],
        },
    }

    with open(output_dir / f"{output_stem}.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"[SERVER INFO] Data saved successfully to '{output_dir / f'{output_stem}.json'}'"
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Exact Cut", color=color)
    ax1.plot(
        generations_axis,
        history_cut,
        color=color,
        linewidth=2,
        label="Best Exact Cut",
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

    plt.title("Evolutionary Dynamics: Analytical Cut Optimization vs. Circuit Depth")
    fig.tight_layout()

    plt.savefig(
        output_dir / f"{output_stem}.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
