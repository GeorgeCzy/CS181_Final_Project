"""Train a TF-IDF + logistic regression baseline classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from wandb_utils import add_wandb_args, init_wandb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "tfidf_logreg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF logistic baseline.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    add_wandb_args(parser)
    return parser.parse_args()


def read_split(path: Path) -> tuple[list[str], list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["id"] for row in rows]
    texts = [row["utterance"] for row in rows]
    labels = [row["label"] for row in rows]
    return ids, texts, labels


def evaluate(
    model: Pipeline,
    texts: list[str],
    labels: list[str],
    *,
    split_name: str,
) -> dict[str, object]:
    predictions = model.predict(texts)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=["chat", "motion_query"])
    return {
        "split": split_name,
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["chat", "motion_query"],
            "matrix": matrix.tolist(),
        },
    }


def write_predictions(
    path: Path,
    *,
    ids: list[str],
    texts: list[str],
    labels: list[str],
    predictions: list[str],
    motion_probabilities: list[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "utterance",
                "label",
                "prediction",
                "p_motion_query",
            ],
        )
        writer.writeheader()
        for row in zip(ids, texts, labels, predictions, motion_probabilities):
            writer.writerow(
                {
                    "id": row[0],
                    "utterance": row[1],
                    "label": row[2],
                    "prediction": row[3],
                    "p_motion_query": f"{row[4]:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    train_ids, train_texts, train_labels = read_split(args.split_dir / "train.csv")
    val_ids, val_texts, val_labels = read_split(args.split_dir / "val.csv")
    test_ids, test_texts, test_labels = read_split(args.split_dir / "test.csv")
    run = init_wandb(
        args,
        config={
            "architecture": "TF-IDF + LogisticRegression",
            "dataset": str(args.split_dir),
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "c": args.c,
            "train_rows": len(train_texts),
            "val_rows": len(val_texts),
            "test_rows": len(test_texts),
        },
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, args.ngram_max),
                    max_features=args.max_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=20260518,
                ),
            ),
        ]
    )
    model.fit(train_texts, train_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train": evaluate(model, train_texts, train_labels, split_name="train"),
        "val": evaluate(model, val_texts, val_labels, split_name="val"),
        "test": evaluate(model, test_texts, test_labels, split_name="test"),
    }
    if run:
        run.log(
            {
                "train/accuracy": metrics["train"]["accuracy"],
                "val/accuracy": metrics["val"]["accuracy"],
                "test/accuracy": metrics["test"]["accuracy"],
            }
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, args.output_dir / "model.joblib")

    class_index = list(model.classes_).index("motion_query")
    for split_name, ids, texts, labels in (
        ("val", val_ids, val_texts, val_labels),
        ("test", test_ids, test_texts, test_labels),
    ):
        predictions = list(model.predict(texts))
        probabilities = model.predict_proba(texts)[:, class_index].tolist()
        write_predictions(
            args.output_dir / f"{split_name}_predictions.csv",
            ids=ids,
            texts=texts,
            labels=labels,
            predictions=predictions,
            motion_probabilities=probabilities,
        )

    print(f"Validation accuracy: {metrics['val']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
