"""Build prefix evaluation rows from full-sentence split files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from shared_utils import read_split, word_prefixes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT = ROOT / "modeling" / "data" / "prefixes" / "test_prefixes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create prefix evaluation dataset.")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ratios",
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated prefix ratios to keep (by word count).",
    )
    return parser.parse_args()


def pick_prefix(
    prefixes: list[tuple[str, float]],
    target_ratio: float,
) -> tuple[str, float] | None:
    eligible = [item for item in prefixes if item[1] + 1e-9 >= target_ratio]
    if not eligible:
        return None
    return min(eligible, key=lambda item: item[1])


def main() -> None:
    args = parse_args()
    ratio_targets = [float(value.strip()) for value in args.ratios.split(",") if value.strip()]
    rows = read_split(args.split_dir / f"{args.split}.csv")

    output_rows: list[dict[str, str]] = []
    for row in rows:
        utterance = row["utterance"]
        prefixes = word_prefixes(utterance)
        total_words = len(utterance.split())
        for target_ratio in ratio_targets:
            picked = pick_prefix(prefixes, target_ratio)
            if picked is None:
                continue
            prefix_text, prefix_ratio = picked
            prefix_words = len(prefix_text.split()) if prefix_text else 0
            output_rows.append(
                {
                    "id": f"{row['id']}-r{int(target_ratio * 100):03d}",
                    "source_id": row["id"],
                    "utterance": utterance,
                    "prefix_utterance": prefix_text,
                    "prefix_ratio": f"{prefix_ratio:.4f}",
                    "target_ratio": f"{target_ratio:.4f}",
                    "prefix_words": str(prefix_words),
                    "total_words": str(total_words),
                    "label": row["label"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source_id",
        "utterance",
        "prefix_utterance",
        "prefix_ratio",
        "target_ratio",
        "prefix_words",
        "total_words",
        "label",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} prefix rows to {args.output}")


if __name__ == "__main__":
    main()
