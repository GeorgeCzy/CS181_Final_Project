"""Validate exported response-mode classifier datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABELS = {"chat", "motion_query"}
FIELDS = {"id", "utterance", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CSV or JSONL dataset.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect-total", type=int, default=None)
    parser.add_argument("--require-balanced", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def validate_rows(
    rows: list[dict[str, str]],
    *,
    expect_total: int | None,
    require_balanced: bool,
) -> Counter[str]:
    if expect_total is not None and len(rows) != expect_total:
        raise ValueError(f"Expected {expect_total} rows, found {len(rows)}.")

    ids: set[str] = set()
    utterances: set[str] = set()
    labels: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        if set(row) != FIELDS:
            raise ValueError(f"Row {index} has unexpected fields: {sorted(row)}")

        row_id = row["id"].strip()
        utterance = row["utterance"].strip()
        label = row["label"].strip()

        if not row_id:
            raise ValueError(f"Row {index} has an empty id.")
        if row_id in ids:
            raise ValueError(f"Duplicate id: {row_id}")
        ids.add(row_id)

        if not utterance:
            raise ValueError(f"Row {index} has an empty utterance.")
        normalized = " ".join(utterance.lower().split())
        if normalized in utterances:
            raise ValueError(f"Duplicate utterance: {utterance}")
        utterances.add(normalized)

        if label not in LABELS:
            raise ValueError(f"Invalid label on row {index}: {label}")
        labels[label] += 1

    if require_balanced and len(set(labels.values())) != 1:
        raise ValueError(f"Label distribution is not balanced: {dict(labels)}")

    return labels


def main() -> None:
    args = parse_args()
    rows = read_rows(args.path)
    labels = validate_rows(
        rows,
        expect_total=args.expect_total,
        require_balanced=args.require_balanced,
    )
    print(f"Validated {len(rows)} rows from {args.path}")
    print(f"Label distribution: {dict(labels)}")


if __name__ == "__main__":
    main()

