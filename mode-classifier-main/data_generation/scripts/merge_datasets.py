"""Merge classifier CSV/JSONL datasets and assign clean sequential ids."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LABELS = {"chat", "motion_query"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated classifier datasets.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--id-prefix", default="merged")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def read_dataset(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported dataset extension: {path}")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def merge(rows_by_file: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for rows in rows_by_file:
        for row in rows:
            utterance = row.get("utterance", "").strip()
            label = row.get("label", "").strip()
            if not utterance:
                raise ValueError("Found row with empty utterance.")
            if label not in LABELS:
                raise ValueError(f"Invalid label: {label}")
            normalized = normalize(utterance)
            if normalized in seen:
                raise ValueError(f"Duplicate utterance across inputs: {utterance}")
            seen.add(normalized)
            merged.append({"utterance": utterance, "label": label})
    return merged


def write_outputs(rows: list[dict[str, str]], output_stem: str) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"{output_stem}.csv"
    jsonl_path = RAW_DIR / f"{output_stem}.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "utterance", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")

    return csv_path, jsonl_path


def main() -> None:
    args = parse_args()
    rows_by_file = [read_dataset(path) for path in args.input]
    merged = merge(rows_by_file)
    if args.shuffle:
        random.Random(args.seed).shuffle(merged)
    for index, row in enumerate(merged, start=1):
        row["id"] = f"{args.id_prefix}-{index:05d}"

    csv_path, jsonl_path = write_outputs(merged, args.output_stem)
    print(f"Wrote {len(merged)} rows to {csv_path}")
    print(f"Wrote {len(merged)} rows to {jsonl_path}")
    print(f"Label distribution: {dict(Counter(row['label'] for row in merged))}")


if __name__ == "__main__":
    main()

