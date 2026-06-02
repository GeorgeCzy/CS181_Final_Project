"""Train/evaluate the lexical keyword heuristic baseline (no fitting step)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from shared_utils import (
    compute_metrics,
    collect_predictions,
    make_keyword_predictor,
    read_split,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_HARD_TEST = ROOT / "modeling" / "data" / "hard_test.csv"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "keyword_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate keyword heuristic baseline.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--hard-test", type=Path, default=DEFAULT_HARD_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def evaluate_split(name: str, rows: list[dict[str, str]], predict_fn) -> dict[str, object]:
    gold, predictions, motion_probs = collect_predictions(rows, predict_fn)
    metrics = compute_metrics(gold, predictions, motion_probs)
    metrics["split"] = name
    return metrics


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    predict_fn,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "utterance", "label", "prediction", "p_motion_query"],
        )
        writer.writeheader()
        for row in rows:
            text = row["utterance"]
            prediction, p_motion = predict_fn(text)
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "utterance": text,
                    "label": row["label"],
                    "prediction": prediction,
                    "p_motion_query": f"{p_motion:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    predict_fn = make_keyword_predictor()
    splits = {
        "train": read_split(args.split_dir / "train.csv"),
        "val": read_split(args.split_dir / "val.csv"),
        "test": read_split(args.split_dir / "test.csv"),
    }
    if args.hard_test.exists():
        splits["hard"] = read_split(args.hard_test)

    metrics = {
        name: evaluate_split(name, rows, predict_fn) for name, rows in splits.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    for name, rows in splits.items():
        if name == "train":
            continue
        write_predictions(args.output_dir / f"{name}_predictions.csv", rows, predict_fn)

    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    if "hard" in metrics:
        print(f"Hard accuracy: {metrics['hard']['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
