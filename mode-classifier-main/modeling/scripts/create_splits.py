"""Create stratified train/validation/test splits for classifier training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data_generation" / "data" / "raw" / "deepseek_generated_10000.csv"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "data" / "splits"
LABELS = {"chat", "motion_query"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified dataset splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, start=1):
        if set(row) != {"id", "utterance", "label"}:
            raise ValueError(f"Row {index} has unexpected fields: {sorted(row)}")
        if row["label"] not in LABELS:
            raise ValueError(f"Row {index} has invalid label: {row['label']}")
    return rows


def split_label_rows(
    rows: list[dict[str, str]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if round(train_ratio + val_ratio + test_ratio, 8) != 1:
        raise ValueError("Split ratios must sum to 1.")

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    for label, label_rows in by_label.items():
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_end = round(total * train_ratio)
        val_end = train_end + round(total * val_ratio)
        train.extend(label_rows[:train_end])
        val.extend(label_rows[train_end:val_end])
        test.extend(label_rows[val_end:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "utterance", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    output_dir: Path,
    *,
    input_path: Path,
    splits: dict[str, list[dict[str, str]]],
    seed: int,
) -> None:
    manifest = {
        "input": str(input_path.relative_to(ROOT)),
        "seed": seed,
        "splits": {
            name: {
                "rows": len(rows),
                "labels": dict(Counter(row["label"] for row in rows)),
            }
            for name, rows in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_rows(args.input)
    train, val, test = split_label_rows(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )
    splits = {"train": train, "val": val, "test": test}
    for name, split_rows in splits.items():
        write_csv(args.output_dir / f"{name}.csv", split_rows)
    write_manifest(args.output_dir, input_path=args.input, splits=splits, seed=args.seed)

    for name, split_rows in splits.items():
        labels = Counter(row["label"] for row in split_rows)
        print(f"{name}: {len(split_rows)} rows, labels={dict(labels)}")


if __name__ == "__main__":
    main()
