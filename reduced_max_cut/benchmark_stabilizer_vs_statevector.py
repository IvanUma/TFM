from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_clifford_circuit(qubits: int, n_gates: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(qubits)
    qc.h(range(qubits))

    for _ in range(n_gates):
        gate = rng.choice(["h", "s", "cx"])
        if gate == "cx":
            a, b = rng.sample(range(qubits), 2)
            qc.cx(a, b)
        elif gate == "h":
            qc.h(rng.randrange(qubits))
        else:
            qc.s(rng.randrange(qubits))

    return qc


def measure_simulation_time(
    qc: QuantumCircuit, method: str, shots: int, repeats: int = 5
) -> dict:
    simulator = AerSimulator(method=method, device="CPU", max_parallel_threads=1)

    qc_with_measure = qc.copy()
    qc_with_measure.measure_all()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        simulator.run(qc_with_measure, shots=shots).result().get_counts()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": method,
        "shots": shots,
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times, ddof=1)),
        "time_median": float(np.median(times)),
        "time_min": float(np.min(times)),
        "time_max": float(np.max(times)),
        "n_repeats": repeats,
    }


def run_benchmark(
    qubits_config: dict[int, dict],
    shots_list: list[int],
    csv_path: str | Path | None = None,
):
    if csv_path is None:
        csv_path = RESULTS_DIR / "benchmark_results.csv"

    all_results = []

    for qubits in sorted(qubits_config):
        cfg = qubits_config[qubits]
        gate_counts = cfg["gate_counts"]
        circuits_per_config = cfg.get("circuits_per_config", 5)
        repeats_per_measurement = cfg.get("repeats_per_measurement", 3)
        methods = ["stabilizer", "statevector"]

        print(f"\n{'=' * 60}")
        print(
            f"  QUBITS = {qubits}  |  methods={methods}  |  circuits={circuits_per_config}  |  repeats={repeats_per_measurement}"
        )
        print(f"{'=' * 60}")

        for n_gates in gate_counts:
            print(f"\n  --- Gates = {n_gates} ---")

            for circuit_idx in range(circuits_per_config):
                seed = circuit_idx + n_gates * 1000 + qubits * 100000

                qc = generate_clifford_circuit(qubits, n_gates, seed)
                depth_before = qc.depth()

                result_row = {
                    "qubits": qubits,
                    "n_gates": n_gates,
                    "depth": depth_before,
                    "circuit_idx": circuit_idx,
                    "seed": seed,
                }

                valid = True
                for method in methods:
                    for shots in shots_list:
                        try:
                            timing = measure_simulation_time(
                                qc, method, shots, repeats_per_measurement
                            )
                            result_row[f"time_mean_{method}_{shots}"] = timing[
                                "time_mean"
                            ]
                            result_row[f"time_std_{method}_{shots}"] = timing[
                                "time_std"
                            ]
                            result_row[f"time_median_{method}_{shots}"] = timing[
                                "time_median"
                            ]
                        except Exception as exc:
                            print(
                                f"    [ERROR] qubits={qubits} gates={n_gates}"
                                f" circuit={circuit_idx} method={method}"
                                f" shots={shots}: {exc}"
                            )
                            valid = False

                if not valid:
                    print(f"    Skipping circuit {circuit_idx + 1} due to errors")
                    continue

                if "statevector" in methods and "stabilizer" in methods:
                    ratio = (
                        result_row["time_mean_statevector_1024"]
                        / result_row["time_mean_stabilizer_1024"]
                    )
                    result_row["ratio_mean_statevector_over_stabilizer_1024"] = ratio

                    print(
                        f"    [{circuit_idx + 1:>2}/{circuits_per_config}]"
                        f" depth={depth_before:>4d}"
                        f"  stab_1024={result_row['time_mean_stabilizer_1024']:.6f}s"
                        f"  sv_1024={result_row['time_mean_statevector_1024']:.6f}s"
                        f"  ratio={ratio:.2f}x"
                    )
                else:
                    print(
                        f"    [{circuit_idx + 1:>2}/{circuits_per_config}]"
                        f" depth={depth_before:>4d}"
                        f"  stab_1024={result_row['time_mean_stabilizer_1024']:.6f}s"
                    )

                all_results.append(result_row)

        if all_results:
            fieldnames = list(all_results[0].keys())
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            print(f"  [PARTIAL] Saved {len(all_results)} results to {csv_path}")

    print(f"\n[DONE] Total circuits: {len(all_results)}")
    return all_results


if __name__ == "__main__":
    shots_list = [1024]

    qubits_config = {
        5: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 5,
        },
        7: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        10: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        12: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        15: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        18: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        20: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 3,
        },
        25: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 2,
        },
        30: {
            "gate_counts": [15, 30, 60],
            "circuits_per_config": 5,
            "repeats_per_measurement": 2,
        },
    }

    qubits_str = ", ".join(str(q) for q in qubits_config)
    print("=" * 60)
    print("  CLIFFORD BENCHMARK: Stabilizer vs Statevector")
    print("=" * 60)
    print(f"  Qubits: {qubits_str}")
    print(f"  Shots: {shots_list}")
    print(f"  Probando statevector en todos los qubits (incluyendo >25)")
    print("=" * 60)

    run_benchmark(
        qubits_config=qubits_config,
        shots_list=shots_list,
    )
