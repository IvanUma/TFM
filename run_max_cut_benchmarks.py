from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MULTIOBJECTIVE_CONFIG = PROJECT_ROOT / "algorithms" / "multi_objective_max_cut" / "config.json"

ALGORITHMS = {
    "basic": PROJECT_ROOT / "algorithms" / "basic_max_cut" / "main.py",
    "qpu_less": PROJECT_ROOT / "algorithms" / "qpu_less" / "main.py",
    "multiobjective": PROJECT_ROOT / "algorithms" / "multi_objective_max_cut" / "main.py",
}

MULTIOBJECTIVE_VARIANTS = {
    "multiobjective_clifford": "clifford",
    "multiobjective_parametric": "parametric",
}


def run_algorithm(
    algorithm: str,
    instance_name: str,
    approach: str | None = None,
) -> None:
    script_path = ALGORITHMS[algorithm]
    original_config = None

    if approach is not None:
        if not MULTIOBJECTIVE_CONFIG.exists():
            raise FileNotFoundError(f"Missing config file: {MULTIOBJECTIVE_CONFIG}")

        original_config = MULTIOBJECTIVE_CONFIG.read_text(encoding="utf-8")
        config_data = json.loads(original_config)
        config_data["approach"] = approach
        MULTIOBJECTIVE_CONFIG.write_text(
            json.dumps(config_data, indent=4),
            encoding="utf-8",
        )

    try:
        process = subprocess.run(
            [sys.executable, str(script_path)],
            input=f"{instance_name}\n",
            text=True,
        )

        if process.returncode != 0:
            print(f"[ERROR] {algorithm} failed on {instance_name}")
    finally:
        if original_config is not None:
            MULTIOBJECTIVE_CONFIG.write_text(
                original_config,
                encoding="utf-8",
            )


def main() -> None:
    print("Available algorithms:")
    for name in ALGORITHMS:
        print(f" - {name}")
    for name in MULTIOBJECTIVE_VARIANTS:
        print(f" - {name}")

    selected_algorithms = input(
        "\nAlgorithms to run (comma separated, default: both): "
    ).strip()

    if not selected_algorithms:
        selected_algorithms = list(ALGORITHMS) + list(MULTIOBJECTIVE_VARIANTS)
    else:
        selected_algorithms = [
            alg.strip()
            for alg in selected_algorithms.split(",")
            if alg.strip() in ALGORITHMS or alg.strip() in MULTIOBJECTIVE_VARIANTS
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

            if algorithm in MULTIOBJECTIVE_VARIANTS:
                run_algorithm(
                    "multiobjective",
                    instance,
                    approach=MULTIOBJECTIVE_VARIANTS[algorithm],
                )
            else:
                run_algorithm(
                    algorithm,
                    instance,
                )


if __name__ == "__main__":
    main()
