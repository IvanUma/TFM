from __future__ import annotations

import csv
import functools
import json
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
from deap import algorithms, base, creator, tools

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.qpu_less.clifford_engine import evaluate_individual_algebraic
from algorithms.qpu_less.utils import (
    cx_quantum_circuit,
    generate_heuristic_individual,
    generate_guided_individual,
    load_external_maxcut_instance,
    mut_quantum_circuit,
)

if not hasattr(creator, "MultiFitness"):
    creator.create("MultiFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "MultiIndividual"):
    creator.create("MultiIndividual", list, fitness=creator.MultiFitness)

toolbox: base.Toolbox = base.Toolbox()


def format_gate(gate: tuple) -> str:
    if gate[0] in {"H", "S"}:
        return f"{gate[0]} q[{gate[1]}]"
    if gate[0] == "CX":
        return f"CX q[{gate[1]}], q[{gate[2]}]"
    return str(gate)


def save_circuit_text(
    output_path: Path,
    best_individual: list[tuple],
    best_expected_cut: float,
    approximation_ratio: float,
    best_depth: float,
) -> None:
    lines = [
        "QPU-Less Clifford Circuit",
        "=========================",
        f"Expected MaxCut value: {best_expected_cut:.8f}",
        f"Approximation ratio: {approximation_ratio:.8f}",
        f"Gate count objective: {best_depth:.0f}",
        "",
        "Gate sequence:",
    ]
    lines.extend(
        f"{index:04d}: {format_gate(gate)}"
        for index, gate in enumerate(best_individual)
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history_csv(
    output_path: Path,
    generations_axis: list[int],
    history_cut: list[float],
    history_depth: list[float],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_exact_cut", "alpha", "best_depth"])
        writer.writerows(zip(generations_axis, history_cut, history_depth))


def save_evolution_plot(
    output_path: Path,
    generations_axis: list[int],
    history_cut: list[float],
    history_depth: list[float],
    optimal_cut: float,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    cut_color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Exact Cut", color=cut_color)
    ax1.plot(
        generations_axis,
        history_cut,
        color=cut_color,
        linewidth=2,
        label="Best Exact Cut",
    )

    ax1.axhline(
        y=optimal_cut,
        color="r",
        linestyle="-",
        alpha=0.3,
        label="Optimal Classical Cut",
    )

    ax1.tick_params(axis="y", labelcolor=cut_color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    depth_color = "tab:orange"
    ax2.set_ylabel("Best Circuit Gates", color=depth_color)
    ax2.plot(
        generations_axis,
        history_depth,
        color=depth_color,
        linestyle="--",
        linewidth=2,
        label="Best Circuit Gates",
    )
    ax2.tick_params(axis="y", labelcolor=depth_color)

    plt.title("QPU-Less Evolution: Analytical Cut vs. Circuit Size")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    instance_name = input("Enter the instance filename: ").strip()
    if not instance_name:
        print("[ERROR] No file was specified.")
        return

    instance_path = Path(instance_name)
    if not instance_path.is_absolute():
        instance_path = PROJECT_ROOT / "max_cut_instances" / instance_path

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
    print(
        "--- MODO ACTIVO: Evaluación por Imagen de Heisenberg Analítica (QPU-Less) ---"
    )

    mu, lambda_ = 100, 150
    population = []

    for _ in range(int(mu * 0.2)):
        population.append(creator.MultiIndividual(toolbox.individual_heuristic()))
    for _ in range(mu - len(population)):
        population.append(creator.MultiIndividual(toolbox.individual()))

    crossover_prob = 0.6
    mutation_prob = 0.35
    generations = 150

    logbook = tools.Logbook()
    logbook.header = ["gen", "best_analytical_cut", "alpha", "best_depth"]

    stats_cut = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_depth = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    statistics = tools.MultiStatistics(cut=stats_cut, depth=stats_depth)
    statistics.register("min", np.min)
    statistics.register("mean", np.mean)

    # AquÃ­ inyectamos nuestro motor
    toolbox.register(
        "evaluate",
        functools.partial(
            evaluate_individual_algebraic, num_qubits=num_qubits, graph=graph
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
            population, len(population), first_front_only=True
        )[0]
        best_individual = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        best_cut = -best_individual.fitness.values[0]
        best_depth = best_individual.fitness.values[1]

        current_alpha = (
            best_cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0
        )

        record = statistics.compile(population)
        logbook.record(
            gen=gen,
            best_analytical_cut=best_cut,
            best_depth=best_depth,
            **record,
        )

        print(
            f"Gen {gen}: Exact Cut (Álgebra) = {best_cut:.4f} | Alpha = {current_alpha:.4f} | Gates = {best_depth:.1f}"
        )

    pool.close()
    pool.join()

    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]
    best_individual = min(pareto_front, key=lambda ind: ind.fitness.values[0])
    best_quantum_cut = -best_individual.fitness.values[0]
    best_depth = best_individual.fitness.values[1]
    approximation_ratio = (
        best_quantum_cut / optimal_classical_cut if optimal_classical_cut > 0 else 0.0
    )

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / Path(file_path).name / "analytical"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{Path(file_path).name}_q{num_qubits}_g{generations}_{timestamp}"

    generations_axis = [int(g) for g in logbook.select("gen")]
    history_cut = [float(c) for c in logbook.select("best_analytical_cut")]
    history_depth = [float(d) for d in logbook.select("best_depth")]

    json_path = output_dir / f"{output_stem}.json"
    plot_path = output_dir / f"{output_stem}.png"
    history_csv_path = output_dir / f"{output_stem}_history.csv"
    circuit_txt_path = output_dir / f"{output_stem}_circuit.txt"

    save_evolution_plot(
        plot_path,
        generations_axis,
        history_cut,
        history_depth,
        optimal_classical_cut,
    )
    save_history_csv(
        history_csv_path,
        generations_axis,
        history_cut,
        history_depth,
    )
    save_circuit_text(
        circuit_txt_path,
        best_individual,
        best_quantum_cut,
        approximation_ratio,
        best_depth,
    )

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
            "best_expected_quantum_cut": float(best_quantum_cut),
            "approximation_ratio": float(approximation_ratio),
            "best_individual_depth": int(best_depth),
            "best_circuit": best_individual,
        },
        "history": {
            "generation": generations_axis,
            "best_exact_cut": history_cut,
            "best_depth": history_depth,
        },
        "output_files": {
            "json": str(json_path),
            "evolution_plot": str(plot_path),
            "history_csv": str(history_csv_path),
            "circuit_text": str(circuit_txt_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print("\n[EXITO] Evolucion completada en entorno QPU-Less.")
    print(f"Datos guardados en '{json_path}'")
    print(f"Grafica de evolucion guardada en '{plot_path}'")
    print(f"Historial CSV guardado en '{history_csv_path}'")
    print(f"Circuito guardado en '{circuit_txt_path}'")


if __name__ == "__main__":
    main()
