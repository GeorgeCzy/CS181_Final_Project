"""Shared helpers for training and evaluation scripts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABELS = ("chat", "motion_query")
LABEL_TO_INDEX = {"chat": 0, "motion_query": 1}


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, start=1):
        if "utterance" not in row or "label" not in row:
            raise ValueError(f"Row {index} in {path} is missing required fields.")
        if row["label"] not in LABEL_TO_INDEX:
            raise ValueError(f"Row {index} has invalid label: {row['label']}")
    return rows


def get_text(row: dict[str, str]) -> str:
    return row.get("prefix_utterance") or row["utterance"]


def word_prefixes(text: str) -> list[tuple[str, float]]:
    """Return (prefix_text, prefix_ratio) for each non-empty word prefix."""
    words = text.split()
    if not words:
        return [("", 0.0)]
    total = len(words)
    prefixes: list[tuple[str, float]] = []
    for end in range(1, total + 1):
        prefix = " ".join(words[:end])
        prefixes.append((prefix, end / total))
    return prefixes


def compute_metrics(
    gold_labels: list[str],
    predictions: list[str],
    motion_probabilities: list[float] | None = None,
) -> dict[str, object]:
    accuracy = accuracy_score(gold_labels, predictions)
    macro_f1 = f1_score(gold_labels, predictions, average="macro", zero_division=0)
    matrix = confusion_matrix(gold_labels, predictions, labels=list(LABELS))
    false_motion = sum(
        1
        for gold, pred in zip(gold_labels, predictions)
        if gold == "chat" and pred == "motion_query"
    )
    false_text = sum(
        1
        for gold, pred in zip(gold_labels, predictions)
        if gold == "motion_query" and pred == "chat"
    )
    result: dict[str, object] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "false_motion_rate": false_motion / max(len(gold_labels), 1),
        "false_text_rate": false_text / max(len(gold_labels), 1),
        "false_motion_count": false_motion,
        "false_text_count": false_text,
        "num_examples": len(gold_labels),
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            gold_labels,
            predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }
    if motion_probabilities is not None:
        result["mean_p_motion_query"] = float(np.mean(motion_probabilities))
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Keyword / rule baseline -------------------------------------------------

MOTION_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(point|wave|dance|turn|follow|bring|demonstrat|gesture|spin|imitat)\w*\b",
        r"\b(stop|stay|hold|freeze|halt)\b",
        r"\b(walk|run|step|move|locomot|approach|retreat|back away|come here)\b",
        r"\b(pick up|put down|grab|lift|lower|raise|push|pull|place|hand me)\b",
        r"\b(show me|act out|sign language|answer with a gesture)\b",
        r"\b(look at|gaze|orient|face toward|nod|shake your head)\b",
    )
]

NEGATION_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(don't|do not|dont|no motion|without moving|stay still and explain)\b",
        r"\b(just explain|only explain|words only|no movement|don't move)\b",
        r"\b(do not move|please do not move)\b",
    )
]


def keyword_predict(text: str) -> tuple[str, float]:
    """Lexical heuristic baseline with explicit negation handling."""
    normalized = text.strip()
    if not normalized:
        return "chat", 0.0

    if any(pattern.search(normalized) for pattern in NEGATION_REGEXES):
        return "chat", 0.1

    motion_hits = sum(1 for pattern in MOTION_REGEXES if pattern.search(normalized))
    if motion_hits > 0:
        confidence = min(0.55 + 0.15 * motion_hits, 0.95)
        return "motion_query", confidence

    return "chat", 0.05


def make_keyword_predictor() -> Callable[[str], tuple[str, float]]:
    return keyword_predict


def collect_predictions(
    rows: list[dict[str, str]],
    predict_fn: Callable[[str], tuple[str, float]],
) -> tuple[list[str], list[str], list[float]]:
    gold_labels = [row["label"] for row in rows]
    predictions: list[str] = []
    motion_probs: list[float] = []
    for row in rows:
        text = get_text(row)
        label, p_motion = predict_fn(text)
        predictions.append(label)
        motion_probs.append(p_motion)
    return gold_labels, predictions, motion_probs
