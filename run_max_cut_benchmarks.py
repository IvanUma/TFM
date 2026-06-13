from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

ALGORITHMS = {
    "basic": PROJECT_ROOT / "basic_max_cut" / "main.py",
    "multiobjective": PROJECT_ROOT / "multi_objective_max_cut" / "main.py",
}


def run_algorithm(
    algorithm: str,
    instance_name: str,
) -> None:

    script_path = ALGORITHMS[algorithm]

    process = subprocess.run(
        [sys.executable, str(script_path)],
        input=f"{instance_name}\n",
        text=True,
    )

    if process.returncode != 0:
        print(f"[ERROR] {algorithm} failed on {instance_name}")


def main() -> None:

    print("Available algorithms:")
    for name in ALGORITHMS:
        print(f" - {name}")

    selected_algorithms = (
        input("\nAlgorithms to run (comma separated): ").strip().split(",")
    )

    selected_algorithms = [
        alg.strip() for alg in selected_algorithms if alg.strip() in ALGORITHMS
    ]

    instances_dir = PROJECT_ROOT / "max_cut_instances"

    available_instances = sorted(p.name for p in instances_dir.iterdir() if p.is_file())

    print("\nAvailable instances:")
    for name in available_instances:
        print(f" - {name}")

    selected_instances = input("\nInstances to run (comma separated or ALL): ").strip()

    if selected_instances.upper() == "ALL":
        selected_instances = available_instances
    else:
        selected_instances = [x.strip() for x in selected_instances.split(",")]

    print()

    for instance in selected_instances:
        print(f"================ {instance} ================")

        for algorithm in selected_algorithms:
            print(f"\nRunning {algorithm}...")

            run_algorithm(
                algorithm,
                instance,
            )


if __name__ == "__main__":
    main()
