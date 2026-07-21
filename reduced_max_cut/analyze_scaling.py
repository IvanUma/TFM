from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

STATEVECTOR_QUBIT_LIMIT = 25


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = None
                    continue
                try:
                    parsed[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except (ValueError, AttributeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _aggregate_rows(
    rows: list[dict], key_field: str, methods: list[str], shots_list: list[int]
) -> list[dict]:
    """Aggregate rows that all share the same qubit count and were run with
    the same set of methods (stabilizer-only, or stabilizer+statevector)."""
    groups: dict = {}
    for row in rows:
        key = row[key_field]
        if key not in groups:
            groups[key] = {f"{m}_{s}": [] for m in methods for s in shots_list}
        for method in methods:
            for shots in shots_list:
                k = f"time_mean_{method}_{shots}"
                val = row.get(k)
                if val is None:
                    continue
                groups[key][f"{method}_{shots}"].append(val)

    aggregated = []
    for key, times in sorted(groups.items()):
        n_samples = len(next(iter(times.values()), []))
        if n_samples == 0:
            continue
        row = {key_field: key, "n_samples": n_samples}
        for series_key, vals_list in times.items():
            vals = np.array(vals_list)
            row[f"{series_key}_mean"] = float(np.mean(vals))
            row[f"{series_key}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
            row[f"{series_key}_median"] = float(np.median(vals))

        if "stabilizer" in methods and "statevector" in methods:
            for shots in shots_list:
                sv_mean = row[f"statevector_{shots}_mean"]
                stab_mean = row[f"stabilizer_{shots}_mean"]
                row[f"ratio_{shots}"] = (
                    float(sv_mean / stab_mean) if stab_mean > 0 else float("inf")
                )

        aggregated.append(row)

    return aggregated


def split_rows_by_qubits(rows: list[dict]) -> dict[int, list[dict]]:
    """Group raw CSV rows by qubit count, since each qubit count may have
    been benchmarked with a different set of methods and gate counts."""
    by_qubits: dict[int, list[dict]] = {}
    for row in rows:
        by_qubits.setdefault(int(row["qubits"]), []).append(row)
    return by_qubits


def methods_present(rows: list[dict]) -> list[str]:
    """Detect which methods actually have data in this slice of rows,
    rather than assuming stabilizer+statevector are always both present."""
    methods = []
    for method in ["stabilizer", "statevector"]:
        if any(row.get(f"time_mean_{method}_1") is not None for row in rows):
            methods.append(method)
    return methods


def aggregate_by_depth(
    rows: list[dict], methods: list[str], shots_list: list[int]
) -> list[dict]:
    return _aggregate_rows(rows, "depth", methods, shots_list)


def aggregate_by_gates(
    rows: list[dict], methods: list[str], shots_list: list[int]
) -> list[dict]:
    return _aggregate_rows(rows, "n_gates", methods, shots_list)


def plot_depth_vs_time(
    aggregated: list[dict], qubits: int, methods: list[str], shots: int = 1
):
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [r["depth"] for r in aggregated]
    for method, marker in zip(methods, ["o", "s"]):
        vals = [r[f"{method}_{shots}_mean"] for r in aggregated]
        errs = [r[f"{method}_{shots}_std"] for r in aggregated]
        ax.errorbar(
            depths,
            vals,
            yerr=errs,
            marker=marker,
            capsize=3,
            label=f"{method.title()} (shots={shots})",
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Mean Simulation Time (s)")
    ax.set_title(f"{qubits}-Qubit Clifford: Simulation Time vs Depth (shots={shots})")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if "stabilizer" in methods and "statevector" in methods:
        for r in aggregated:
            stab_val = r[f"stabilizer_{shots}_mean"]
            stv_val = r[f"statevector_{shots}_mean"]
            if stab_val <= 0 or stv_val <= 0:
                continue
            mid = (np.log(stab_val) + np.log(stv_val)) / 2
            ratio = stv_val / stab_val
            ax.annotate(
                f"{ratio:.1f}x",
                xy=(r["depth"], np.exp(mid)),
                ha="center",
                va="bottom",
                fontsize=7,
                color="purple",
            )

    plt.tight_layout()
    path = RESULTS_DIR / f"depth_vs_time_q{qubits}_shots_{shots}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def plot_ratio_vs_depth(aggregated: list[dict], qubits: int, shots_list: list[int]):
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [r["depth"] for r in aggregated]
    for shots, marker in zip(shots_list, ["o", "s"]):
        ratios = [r[f"ratio_{shots}"] for r in aggregated]
        ax.plot(
            depths,
            ratios,
            marker=marker,
            label=f"Statevector/Stabilizer (shots={shots})",
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Equal")

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Time Ratio (Statevector / Stabilizer)")
    ax.set_title(f"{qubits}-Qubit: Simulation Time Ratio vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log")

    plt.tight_layout()
    path = RESULTS_DIR / f"ratio_vs_depth_q{qubits}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def plot_gates_vs_time(
    aggregated: list[dict], qubits: int, methods: list[str], shots: int = 1
):
    fig, ax = plt.subplots(figsize=(10, 6))

    n_gates_list = [r["n_gates"] for r in aggregated]
    for method, marker in zip(methods, ["o", "s"]):
        vals = [r[f"{method}_{shots}_mean"] for r in aggregated]
        errs = [r[f"{method}_{shots}_std"] for r in aggregated]
        ax.errorbar(
            n_gates_list,
            vals,
            yerr=errs,
            marker=marker,
            capsize=3,
            label=f"{method.title()} (shots={shots})",
        )

    ax.set_xlabel("Number of Gates")
    ax.set_ylabel("Mean Simulation Time (s)")
    ax.set_title(
        f"{qubits}-Qubit Clifford: Simulation Time vs Gate Count (shots={shots})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    path = RESULTS_DIR / f"gates_vs_time_q{qubits}_shots_{shots}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def plot_time_vs_qubits(
    all_aggregated_by_qubits: dict[int, list[dict]], shots: int = 1
):
    """Cross-cutting view the per-qubit plots can't show: how simulation
    time scales with qubit count at a fixed gate count, for each method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    series: dict[str, dict[int, float]] = {"stabilizer": {}, "statevector": {}}
    for qubits, aggregated in sorted(all_aggregated_by_qubits.items()):
        target_rows = [
            r
            for r in aggregated
            if r["n_gates"] == max(r["n_gates"] for r in aggregated)
        ]
        if not target_rows:
            continue
        row = target_rows[0]
        for method in ["stabilizer", "statevector"]:
            key = f"{method}_{shots}_mean"
            if key in row:
                series[method][qubits] = row[key]

    for method, marker in zip(["stabilizer", "statevector"], ["o", "s"]):
        data = series[method]
        if not data:
            continue
        qs = sorted(data)
        ax.plot(
            qs,
            [data[q] for q in qs],
            marker=marker,
            label=f"{method.title()} (shots={shots})",
        )

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Mean Simulation Time (s), largest n_gates per qubit count")
    ax.set_title(f"Simulation Time vs Qubit Count (shots={shots})")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yscale("log")

    plt.tight_layout()
    path = RESULTS_DIR / f"time_vs_qubits_shots_{shots}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def print_table(
    by_gates: list[dict], qubits: int, methods: list[str], shots_list: list[int]
):
    print(f"\n  Qubits = {qubits}  (methods: {', '.join(methods)})")
    cols = []
    for method in methods:
        for shots in shots_list:
            cols.append(f"{method[:4].title()}({shots})")
    header = f"{'Gates':>6} | " + " | ".join(f"{c:>12}" for c in cols)
    if "stabilizer" in methods and "statevector" in methods:
        header += " | " + " | ".join(f"{'R(' + str(s) + ')':>8}" for s in shots_list)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in by_gates:
        line = f"{r['n_gates']:>6d} | "
        line += " | ".join(
            f"{r[f'{method}_{shots}_mean']:>12.6f}"
            for method in methods
            for shots in shots_list
        )
        if "stabilizer" in methods and "statevector" in methods:
            line += " | " + " | ".join(f"{r[f'ratio_{s}']:>7.2f}x" for s in shots_list)
        print(line)
    print(sep)


def print_scaling_summary(
    aggregated: list[dict], qubits: int, methods: list[str], field: str = "n_gates"
):
    """Print scaling exponents for whichever methods have data at this qubit count."""
    print(f"  --- Scaling Analysis (qubits={qubits}) ---")
    for method in methods:
        key = f"{method}_1_mean"
        xs = np.array([r[field] for r in aggregated if r[field] > 0])
        times = np.array([r[key] for r in aggregated if r[field] > 0])
        if len(xs) < 3:
            print(
                f"  {method.title()}: not enough data points to fit a scaling exponent"
            )
            continue
        log_x = np.log(xs)
        log_t = np.log(times)
        coeffs = np.polyfit(log_x, log_t, 1)
        exponent = coeffs[0]
        print(f"  {method.title()}:  time ~ {field}^{exponent:.3f}  (log-log slope)")
    print()


def main():
    rows = load_results(CSV_PATH)
    if not rows:
        print(f"[ERROR] No results found in {CSV_PATH}")
        print("  Run benchmark_stabilizer_vs_statevector.py first.")
        return

    shots_list = [1, 1024]
    by_qubits = split_rows_by_qubits(rows)

    all_by_gates: dict[int, list[dict]] = {}

    for qubits, qrows in sorted(by_qubits.items()):
        methods = methods_present(qrows)
        if not methods:
            print(f"[WARN] qubits={qubits}: no usable timing data found, skipping")
            continue

        by_depth = aggregate_by_depth(qrows, methods, shots_list)
        by_gates = aggregate_by_gates(qrows, methods, shots_list)
        all_by_gates[qubits] = by_gates

        print_table(by_gates, qubits, methods, shots_list)
        print_scaling_summary(by_gates, qubits, methods, field="n_gates")

        plot_depth_vs_time(by_depth, qubits, methods, shots=1)
        plot_depth_vs_time(by_depth, qubits, methods, shots=1024)
        plot_gates_vs_time(by_gates, qubits, methods, shots=1)
        plot_gates_vs_time(by_gates, qubits, methods, shots=1024)

        if "stabilizer" in methods and "statevector" in methods:
            plot_ratio_vs_depth(by_depth, qubits, shots_list)
        else:
            print(
                f"  [INFO] qubits={qubits}: statevector data unavailable "
                f"(likely > {STATEVECTOR_QUBIT_LIMIT} qubits) -> skipping ratio plot"
            )

        with open(RESULTS_DIR / f"aggregated_by_depth_q{qubits}.json", "w") as f:
            json.dump(by_depth, f, indent=2)
        with open(RESULTS_DIR / f"aggregated_by_gates_q{qubits}.json", "w") as f:
            json.dump(by_gates, f, indent=2)

    if all_by_gates:
        plot_time_vs_qubits(all_by_gates, shots=1)
        plot_time_vs_qubits(all_by_gates, shots=1024)

    print(f"\n  Saved aggregated JSON files and plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
