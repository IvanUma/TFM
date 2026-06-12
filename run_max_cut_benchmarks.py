from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

INSTANCES = ["c5.txt", "k5.txt", "petersen.txt"]
MODULES = [
    ("basic", "basic_max_cut.main"),
    ("multiobjective", "multi_objective_max_cut.main"),
]


def run_solver(module: str, instance_name: str) -> None:
    command = [str(PYTHON), "-m", module]
    print(f"\n=== Running {module} on {instance_name} ===")
    result = subprocess.run(
        command,
        input=f"{instance_name}\n",
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[ERROR] {module} failed for {instance_name} with exit code {result.returncode}"
        )


def main() -> None:
    if not PYTHON.exists():
        raise SystemExit(f"[ERROR] Python interpreter not found at: {PYTHON}")

    for instance_name in INSTANCES:
        for _, module in MODULES:
            run_solver(module, instance_name)


if __name__ == "__main__":
    main()
