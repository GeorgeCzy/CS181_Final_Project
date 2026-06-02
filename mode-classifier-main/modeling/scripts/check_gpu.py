"""Print lightweight GPU availability information for shared servers."""

from __future__ import annotations

import shutil
import subprocess


def run_command(command: list[str]) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, check=False, text=True)
    print()
    return completed.returncode


def main() -> None:
    if shutil.which("gpustat"):
        run_command(["gpustat"])
    elif shutil.which("nvidia-smi"):
        run_command(["nvidia-smi"])
    else:
        print("Neither gpustat nor nvidia-smi is available on PATH.")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment.")
        return

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")


if __name__ == "__main__":
    main()

