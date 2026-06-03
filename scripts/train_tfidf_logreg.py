"""Train a TF-IDF + logistic regression classifier from the top-level data folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "tfidf_logreg"
LABELS = ["chat", "motion_query"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a TF-IDF + logistic regression baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--val-file", default="val.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"id", "utterance", "label"}
    if not rows:
        raise ValueError(f"{path} is empty.")

    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    for index, row in enumerate(rows, start=1):
        if row["label"] not in LABELS:
            raise ValueError(
                f"{path} row {index} has invalid label {row['label']!r}; "
                f"expected one of {LABELS}."
            )
    return rows


def split_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    ids = [row["id"] for row in rows]
    texts = [row["utterance"] for row in rows]
    labels = [row["label"] for row in rows]
    return ids, texts, labels


def make_model(max_features: int, ngram_max: int, c: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=20260518,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, texts: list[str], labels: list[str]) -> dict[str, object]:
    predictions = model.predict(texts)
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            labels,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def write_predictions(
    path: Path,
    *,
    model: Pipeline,
    ids: list[str],
    texts: list[str],
    labels: list[str],
) -> None:
    predictions = list(model.predict(texts))
    class_names = list(model.classes_)
    motion_index = class_names.index("motion_query")
    motion_probabilities = model.predict_proba(texts)[:, motion_index]

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
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    train_rows = read_rows(data_dir / args.train_file)
    val_rows = read_rows(data_dir / args.val_file)
    test_rows = read_rows(data_dir / args.test_file)

    _, train_texts, train_labels = split_columns(train_rows)
    val_ids, val_texts, val_labels = split_columns(val_rows)
    test_ids, test_texts, test_labels = split_columns(test_rows)

    model = make_model(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        c=args.c,
    )
    model.fit(train_texts, train_labels)

    metrics = {
        "config": {
            "data_dir": str(data_dir),
            "train_file": args.train_file,
            "val_file": args.val_file,
            "test_file": args.test_file,
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "c": args.c,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
        "train": evaluate(model, train_texts, train_labels),
        "val": evaluate(model, val_texts, val_labels),
        "test": evaluate(model, test_texts, test_labels),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_predictions(
        output_dir / "val_predictions.csv",
        model=model,
        ids=val_ids,
        texts=val_texts,
        labels=val_labels,
    )
    write_predictions(
        output_dir / "test_predictions.csv",
        model=model,
        ids=test_ids,
        texts=test_texts,
        labels=test_labels,
    )

    print(f"train accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"val accuracy:   {metrics['val']['accuracy']:.4f}")
    print(f"test accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"artifacts:      {output_dir}")


if __name__ == "__main__":
    main()
