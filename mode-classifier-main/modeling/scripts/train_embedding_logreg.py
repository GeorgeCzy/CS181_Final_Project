"""Train logistic regression on cached text embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from embedding_utils import load_embedding_cache
from wandb_utils import add_wandb_args, init_wandb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "embedding_logreg"
LABEL_TO_INDEX = {"chat": 0, "motion_query": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LogReg on cached embeddings.")
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260518)
    add_wandb_args(parser)
    return parser.parse_args()


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def split_to_arrays(
    rows: list[dict[str, str]],
    id_to_index: dict[str, int],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = [id_to_index[row["id"]] for row in rows]
    labels = np.array([LABEL_TO_INDEX[row["label"]] for row in rows], dtype=np.int64)
    features = embeddings[indices]
    return features, labels


def evaluate(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = model.predict(features)
    return {"accuracy": float(accuracy_score(labels, predictions))}


def main() -> None:
    args = parse_args()
    id_to_index, embeddings = load_embedding_cache(args.embedding_cache)
    train_rows = read_split(args.split_dir / "train.csv")
    val_rows = read_split(args.split_dir / "val.csv")
    test_rows = read_split(args.split_dir / "test.csv")

    x_train, y_train = split_to_arrays(train_rows, id_to_index, embeddings)
    x_val, y_val = split_to_arrays(val_rows, id_to_index, embeddings)
    x_test, y_test = split_to_arrays(test_rows, id_to_index, embeddings)

    run = init_wandb(
        args,
        config={
            "architecture": "CachedEmbedding + LogisticRegression",
            "embedding_cache": str(args.embedding_cache),
            "embedding_dim": int(embeddings.shape[1]),
            "c": args.c,
            "seed": args.seed,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
    )

    model = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib")

    model_config = {
        "architecture": "CachedEmbedding + LogisticRegression",
        "head": "logreg",
        "embedding_cache": str(args.embedding_cache),
        "input_dim": int(embeddings.shape[1]),
        "labels": {str(index): label for index, label in enumerate(("chat", "motion_query"))},
    }
    cache_metadata_path = args.embedding_cache.with_suffix(".json")
    if cache_metadata_path.exists():
        model_config["embedding_metadata"] = json.loads(
            cache_metadata_path.read_text(encoding="utf-8")
        )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "val": val_metrics,
                "test": test_metrics,
                "c": args.c,
                "embedding_dim": int(embeddings.shape[1]),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")
    if run:
        run.log(
            {
                "val/accuracy": val_metrics["accuracy"],
                "test/accuracy": test_metrics["accuracy"],
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
