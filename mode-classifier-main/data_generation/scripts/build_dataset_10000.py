"""Build a 10,000-row dataset from the existing 2,000 rows plus template expansion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXISTING = ROOT / "data" / "raw" / "deepseek_generated_2000.csv"
TEMPLATE_BATCH = ROOT / "data" / "raw" / "template_generated_8000.csv"
OUTPUT_STEM = "deepseek_generated_10000"
TARGET_TOTAL = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge data into a 10k dataset.")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        help="Generate the gap with DeepSeek API instead of templates.",
    )
    parser.add_argument("--deepseek-batch-size", type=int, default=50)
    return parser.parse_args()


def run_script(name: str, *cli_args: str) -> None:
    script = SCRIPTS / name
    command = [sys.executable, str(script), *cli_args]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def count_rows(path: Path) -> int:
    import csv

    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    args = parse_args()
    if not EXISTING.exists():
        raise FileNotFoundError(f"Missing base dataset: {EXISTING}")

    current = count_rows(EXISTING)
    if current >= args.target:
        raise ValueError(
            f"Base dataset already has {current} rows (target {args.target})."
        )

    gap = args.target - current
    print(f"Base rows: {current}; need {gap} more to reach {args.target}.")

    if args.use_deepseek:
        run_script(
            "generate_with_deepseek.py",
            "--total",
            str(gap),
            "--batch-size",
            str(args.deepseek_batch_size),
            "--output-stem",
            "deepseek_generated_extra",
            "--id-prefix",
            "deepseek-extra",
            "--exclude-path",
            str(EXISTING),
        )
        extra_path = ROOT / "data" / "raw" / "deepseek_generated_extra.csv"
    else:
        run_script(
            "expand_templates.py",
            "--count",
            str(gap),
            "--output-stem",
            "template_generated_8000",
            "--exclude",
            str(EXISTING),
            "--seed",
            str(args.seed),
        )
        extra_path = TEMPLATE_BATCH

    run_script(
        "merge_datasets.py",
        "--input",
        str(EXISTING),
        "--input",
        str(extra_path),
        "--output-stem",
        OUTPUT_STEM,
        "--id-prefix",
        "ds10k",
        "--shuffle",
        "--seed",
        str(args.seed),
    )

    output_csv = ROOT / "data" / "raw" / f"{OUTPUT_STEM}.csv"
    run_script(
        "validate_dataset.py",
        str(output_csv),
        "--expect-total",
        str(args.target),
        "--require-balanced",
    )
    print(f"Done. Main dataset: {output_csv}")


if __name__ == "__main__":
    main()
