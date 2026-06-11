"""Run the V1 full-input binary classifier for minilm + mlp.

This folder represents exactly one experiment combination. Extra command-line
arguments are forwarded to the project training script. Example:
    python program.py --device cuda --skip-existing
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(root / "experiments/run_experiments.py"),
        "--encoders",
        "minilm",
        "--heads",
        "mlp",
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
