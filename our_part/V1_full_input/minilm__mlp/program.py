"""Run the V1 full-input binary classifier: minilm + mlp.

This file is the one-combination entry point for the submitted sourcecode
folder. It writes reproduced outputs under sourcecode/reproduced/ and does not
overwrite the tracked result files in this model folder.

Extra command-line arguments are forwarded to the project training script.
Example:
    python program.py --device cuda --skip-existing
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts/run_v1_experiments.py"),
        "--encoders",
        "minilm",
        "--heads",
        "mlp",
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
