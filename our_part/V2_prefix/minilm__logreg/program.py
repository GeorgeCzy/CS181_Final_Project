"""Run the V2 prefix early-decision classifier for minilm + logreg.

Extra command-line arguments are forwarded to the shared runner. Example:
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
        str(root / "version2_prefix/scripts/run_prefix_experiments.py"),
        "--encoders",
        "minilm",
        "--heads",
        "logreg",
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
