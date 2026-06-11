"""Evaluate all V1 encoder/head models on the curated hard test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_experiments import (  # noqa: E402
    ENCODER_MODEL_NAMES,
    LABELS,
    LABEL_TO_INDEX,
    make_head,
    load_sentence_transformer,
)


COMBOS = [
    ("tfidf", "logreg", "tfidf__logreg"),
    ("tfidf", "mlp", "tfidf__mlp"),
    ("minilm", "logreg", "minilm__logreg"),
    ("minilm", "mlp", "minilm__mlp"),
    ("clip", "logreg", "clip__logreg"),
    ("clip", "mlp", "clip__mlp"),
    ("qwen3_0_6b", "logreg", "qwen3_0_6b__logreg"),
    ("qwen3_0_6b", "mlp", "qwen3_0_6b__mlp"),
]


@dataclass(frozen=True)
class HardExample:
    row_id: str
    utterance: str
    label: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V1 models on hard test data.")
    parser.add_argument(
        "--hard-test-file",
        type=Path,
        default=ROOT / "our_part" / "data" / "hard_test" / "hard_test.csv",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts" / "ours",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "our_part" / "hard_test",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_hard_examples(path: Path) -> list[HardExample]:
    examples: list[HardExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"Unknown label in hard test file: {label}")
            examples.append(
                HardExample(
                    row_id=row["id"],
                    utterance=row["utterance"],
                    label=label,
                    category=row["hard_category"],
                )
            )
    return examples


def text_digest(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_or_encode_features(
    encoder: str,
    examples: list[HardExample],
    *,
    artifact_dir: Path,
    device: torch.device,
    batch_size: int,
    force_recompute: bool,
) -> np.ndarray:
    texts = [example.utterance for example in examples]
    cache_dir = artifact_dir / "hard_test" / "embeddings" / encoder
    cache_path = cache_dir / "hard_embeddings.npz"
    metadata_path = cache_dir / "metadata.json"
    metadata = {
        "encoder": encoder,
        "rows": len(texts),
        "text_digest": text_digest(texts),
    }

    if not force_recompute and cache_path.exists() and metadata_path.exists():
        if read_json(metadata_path) == metadata:
            return np.load(cache_path)["embeddings"].astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if encoder == "tfidf":
        vectorizer_path = artifact_dir / "embeddings" / "tfidf" / "vectorizer.joblib"
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Missing TF-IDF vectorizer: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        features = vectorizer.transform(texts).astype(np.float32).toarray()
    else:
        model_name = ENCODER_MODEL_NAMES[encoder]
        model = load_sentence_transformer(model_name, device)
        features = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    np.savez_compressed(cache_path, embeddings=features)
    write_json(metadata_path, metadata)
    return features


def predict(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], list[float], list[float], list[float]]:
    model.eval()
    predictions: list[str] = []
    p_chat: list[float] = []
    p_motion: list[float] = []
    confidence: list[float] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size].astype(np.float32)).to(device)
            probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()
            indexes = probabilities.argmax(axis=1)
            predictions.extend(LABELS[index] for index in indexes)
            p_chat.extend(probabilities[:, LABEL_TO_INDEX["chat"]].tolist())
            p_motion.extend(probabilities[:, LABEL_TO_INDEX["motion_query"]].tolist())
            confidence.extend(probabilities.max(axis=1).tolist())
    return predictions, p_chat, p_motion, confidence


def compute_metrics(labels: list[str], predictions: list[str]) -> dict[str, object]:
    total = len(labels)
    chat_total = sum(1 for label in labels if label == "chat")
    motion_total = sum(1 for label in labels if label == "motion_query")
    false_motion = sum(
        1 for gold, pred in zip(labels, predictions) if gold == "chat" and pred == "motion_query"
    )
    false_text = sum(
        1 for gold, pred in zip(labels, predictions) if gold == "motion_query" and pred == "chat"
    )
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "false_motion_count": false_motion,
        "false_text_count": false_text,
        "false_motion_rate_total": false_motion / total,
        "false_text_rate_total": false_text / total,
        "false_motion_rate_by_chat": false_motion / chat_total,
        "false_text_rate_by_motion": false_text / motion_total,
        "chat_accuracy": (chat_total - false_motion) / chat_total,
        "motion_query_accuracy": (motion_total - false_text) / motion_total,
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


def compute_category_metrics(
    examples: list[HardExample],
    predictions: list[str],
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions):
        grouped[example.category].append((example.label, prediction))

    rows = []
    for category in sorted(grouped):
        labels = [label for label, _ in grouped[category]]
        preds = [pred for _, pred in grouped[category]]
        rows.append(
            {
                "hard_category": category,
                "count": len(labels),
                "accuracy": f"{accuracy_score(labels, preds):.6f}",
                "false_motion_count": sum(
                    1 for label, pred in grouped[category] if label == "chat" and pred == "motion_query"
                ),
                "false_text_count": sum(
                    1 for label, pred in grouped[category] if label == "motion_query" and pred == "chat"
                ),
            }
        )
    return rows


def load_model(
    combo: str,
    encoder: str,
    head: str,
    *,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    run_dir = artifact_dir / "runs" / combo
    config_path = run_dir / "run_config.json"
    model_path = run_dir / "model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model artifacts for {combo}: {run_dir}")
    config = read_json(config_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = make_head(
        head,
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(config.get("mlp_hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Hard Test Summary",
        "",
        "The hard test contains 120 manually curated examples: 60 chat and 60 motion_query.",
        "FMR/FTR below are reported as rates over all hard-test examples, matching the report table.",
        "",
        "| run | accuracy | macro F1 | FMR | FTR | false motion | false text |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {accuracy:.4f} | {macro_f1:.4f} | {false_motion_rate_total:.4f} | "
            "{false_text_rate_total:.4f} | {false_motion_count} | {false_text_count} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [row["run_name"] for row in rows]
    values = [row["accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(names, values, color="#4c78a8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Hard test accuracy")
    ax.set_title("Hard Test Accuracy by Model")
    ax.tick_params(axis="x", labelrotation=35)
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    examples = load_hard_examples(args.hard_test_file)
    labels = [example.label for example in examples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    feature_cache: dict[str, np.ndarray] = {}
    for encoder, head, combo in COMBOS:
        if encoder not in feature_cache:
            feature_cache[encoder] = load_or_encode_features(
                encoder,
                examples,
                artifact_dir=args.artifact_dir,
                device=device,
                batch_size=args.batch_size,
                force_recompute=args.force_recompute_embeddings,
            )
        features = feature_cache[encoder]
        model, source_config = load_model(
            combo,
            encoder,
            head,
            artifact_dir=args.artifact_dir,
            device=device,
        )
        predictions, p_chat, p_motion, confidence = predict(
            model,
            features,
            device=device,
            batch_size=int(source_config.get("batch_size", args.batch_size)),
        )
        metrics = compute_metrics(labels, predictions)
        run_output = args.output_dir / combo
        write_json(
            run_output / "metrics.json",
            {
                "run_name": combo,
                "encoder": encoder,
                "head": head,
                "hard_test_file": str(args.hard_test_file),
                "metrics": metrics,
                "source_model_config": source_config,
            },
        )
        prediction_rows = []
        for example, prediction, chat_prob, motion_prob, conf in zip(
            examples,
            predictions,
            p_chat,
            p_motion,
            confidence,
        ):
            prediction_rows.append(
                {
                    "id": example.row_id,
                    "utterance": example.utterance,
                    "label": example.label,
                    "hard_category": example.category,
                    "prediction": prediction,
                    "p_chat": f"{chat_prob:.6f}",
                    "p_motion_query": f"{motion_prob:.6f}",
                    "confidence": f"{conf:.6f}",
                    "correct": str(example.label == prediction).lower(),
                }
            )
        write_csv(
            run_output / "predictions.csv",
            prediction_rows,
            [
                "id",
                "utterance",
                "label",
                "hard_category",
                "prediction",
                "p_chat",
                "p_motion_query",
                "confidence",
                "correct",
            ],
        )
        category_rows = compute_category_metrics(examples, predictions)
        write_csv(
            run_output / "category_metrics.csv",
            category_rows,
            ["hard_category", "count", "accuracy", "false_motion_count", "false_text_count"],
        )
        shutil.copy2(args.artifact_dir / "runs" / combo / "run_config.json", run_output / "source_run_config.json")

        row = {
            "run_name": combo,
            "encoder": encoder,
            "head": head,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "false_motion_count": int(metrics["false_motion_count"]),
            "false_text_count": int(metrics["false_text_count"]),
            "false_motion_rate_total": float(metrics["false_motion_rate_total"]),
            "false_text_rate_total": float(metrics["false_text_rate_total"]),
            "false_motion_rate_by_chat": float(metrics["false_motion_rate_by_chat"]),
            "false_text_rate_by_motion": float(metrics["false_text_rate_by_motion"]),
            "chat_accuracy": float(metrics["chat_accuracy"]),
            "motion_query_accuracy": float(metrics["motion_query_accuracy"]),
        }
        summary_rows.append(row)
        print(
            f"{combo}: acc={row['accuracy']:.4f}, "
            f"false_motion={row['false_motion_count']}, false_text={row['false_text_count']}"
        )

    summary_rows.sort(key=lambda row: (-row["accuracy"], row["false_motion_count"], row["false_text_count"], row["run_name"]))
    write_csv(
        args.output_dir / "summary.csv",
        [
            {
                key: (f"{value:.8f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in summary_rows
        ],
        [
            "run_name",
            "encoder",
            "head",
            "accuracy",
            "macro_f1",
            "false_motion_count",
            "false_text_count",
            "false_motion_rate_total",
            "false_text_rate_total",
            "false_motion_rate_by_chat",
            "false_text_rate_by_motion",
            "chat_accuracy",
            "motion_query_accuracy",
        ],
    )
    write_summary_markdown(args.output_dir / "summary.md", summary_rows)
    plot_summary(args.output_dir / "hard_accuracy.png", summary_rows)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "hard_test_file": str(args.hard_test_file),
            "artifact_dir": str(args.artifact_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "num_examples": len(examples),
            "labels": {label: labels.count(label) for label in LABELS},
            "models_evaluated": [combo for _, _, combo in COMBOS],
        },
    )


if __name__ == "__main__":
    main()
