"""Run encoder + classifier-head experiments for response-mode classification.

This is the self-contained V1 reproduction script for the submitted
sourcecode folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "v1_full_input"
DEFAULT_OUTPUT_DIR = ROOT / "reproduced" / "artifacts" / "v1"

LABELS = ["chat", "motion_query"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

ENCODER_MODEL_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
    "qwen3_0_6b": "Qwen/Qwen3-Embedding-0.6B",
}

ENCODER_CHOICES = ["tfidf", "minilm", "clip", "qwen3_0_6b"]
HEAD_CHOICES = ["logreg", "mlp"]


@dataclass(frozen=True)
class Split:
    name: str
    ids: list[str]
    texts: list[str]
    labels: list[str]

    @property
    def label_indices(self) -> np.ndarray:
        return np.array([LABEL_TO_INDEX[label] for label in self.labels], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4 encoders x 2 classifier heads for the top-level project data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["all"],
        help=f"Encoder names or 'all'. Choices: {', '.join(ENCODER_CHOICES)}",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["all"],
        help=f"Head names or 'all'. Choices: {', '.join(HEAD_CHOICES)}",
    )
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--val-file", default="val.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--tfidf-max-features", type=int, default=10000)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reproduced" / "V1_full_input",
        help="Tracked lightweight report snapshot directory. Use --no-report-copy to disable.",
    )
    parser.add_argument("--no-report-copy", action="store_true")
    return parser.parse_args()


def resolve_choices(values: Iterable[str], choices: list[str], field_name: str) -> list[str]:
    values = list(values)
    if values == ["all"] or "all" in values:
        return choices

    unknown = sorted(set(values) - set(choices))
    if unknown:
        raise ValueError(f"Unknown {field_name}: {unknown}. Valid choices: {choices}")
    return values


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path: Path, *, limit: int | None = None) -> Split:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{path} has no rows.")

    required_columns = {"id", "utterance", "label"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        label = row["label"]
        if label not in LABEL_TO_INDEX:
            raise ValueError(
                f"{path}:{row_number} has invalid label {label!r}; expected {LABELS}"
            )
        ids.append(row["id"])
        texts.append(row["utterance"])
        labels.append(label)

    return Split(path.stem, ids, texts, labels)


def load_splits(args: argparse.Namespace) -> dict[str, Split]:
    data_dir = args.data_dir.resolve()
    splits = {
        "train": read_split(data_dir / args.train_file, limit=args.limit_train),
        "val": read_split(data_dir / args.val_file, limit=args.limit_val),
        "test": read_split(data_dir / args.test_file, limit=args.limit_test),
    }
    if len(splits["train"].texts) < args.min_train_rows:
        raise ValueError(
            f"Training split has {len(splits['train'].texts)} rows, below "
            f"--min-train-rows={args.min_train_rows}. Generate or add more data first, "
            "or lower --min-train-rows for a smoke test."
        )
    return splits


def text_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_sentence_transformer(model_name: str, device: torch.device):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, object] = {"device": str(device)}
    if model_name.startswith("Qwen/"):
        kwargs["trust_remote_code"] = True

    try:
        return SentenceTransformer(model_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return SentenceTransformer(model_name, **kwargs)


def encode_tfidf(
    splits: dict[str, Split],
    *,
    cache_dir: Path,
    max_features: int,
    ngram_max: int,
    force_recompute: bool,
) -> dict[str, np.ndarray]:
    metadata_path = cache_dir / "metadata.json"
    expected_metadata = {
        "encoder": "tfidf",
        "max_features": max_features,
        "ngram_range": [1, ngram_max],
        "train_digest": text_digest(splits["train"].texts),
        "split_sizes": {name: len(split.texts) for name, split in splits.items()},
    }
    embedding_paths = {
        split_name: cache_dir / f"{split_name}_embeddings.npz"
        for split_name in splits
    }
    can_reuse = (
        not force_recompute
        and metadata_path.exists()
        and all(path.exists() for path in embedding_paths.values())
        and read_json(metadata_path) == expected_metadata
    )
    if can_reuse:
        return {
            split_name: np.load(path)["embeddings"].astype(np.float32)
            for split_name, path in embedding_paths.items()
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        max_features=max_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(splits["train"].texts)
    matrices = {
        "train": train_matrix,
        "val": vectorizer.transform(splits["val"].texts),
        "test": vectorizer.transform(splits["test"].texts),
    }
    embeddings = {
        name: matrix.astype(np.float32).toarray()
        for name, matrix in matrices.items()
    }
    for split_name, array in embeddings.items():
        np.savez_compressed(embedding_paths[split_name], embeddings=array)
    joblib.dump(vectorizer, cache_dir / "vectorizer.joblib")
    write_json(metadata_path, expected_metadata)
    return embeddings


def encode_sentence_model(
    encoder_name: str,
    splits: dict[str, Split],
    *,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    force_recompute: bool,
) -> dict[str, np.ndarray]:
    model_name = ENCODER_MODEL_NAMES[encoder_name]
    metadata_path = cache_dir / "metadata.json"
    expected_metadata = {
        "encoder": encoder_name,
        "model_name": model_name,
        "normalize_embeddings": True,
        "split_digests": {
            split_name: text_digest(split.texts)
            for split_name, split in splits.items()
        },
        "split_sizes": {name: len(split.texts) for name, split in splits.items()},
    }
    embedding_paths = {
        split_name: cache_dir / f"{split_name}_embeddings.npz"
        for split_name in splits
    }
    can_reuse = (
        not force_recompute
        and metadata_path.exists()
        and all(path.exists() for path in embedding_paths.values())
        and read_json(metadata_path) == expected_metadata
    )
    if can_reuse:
        return {
            split_name: np.load(path)["embeddings"].astype(np.float32)
            for split_name, path in embedding_paths.items()
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = load_sentence_transformer(model_name, device)
    embeddings: dict[str, np.ndarray] = {}
    for split_name, split in splits.items():
        encoded = model.encode(
            split.texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings[split_name] = encoded.astype(np.float32)
        np.savez_compressed(embedding_paths[split_name], embeddings=embeddings[split_name])
    write_json(metadata_path, expected_metadata)
    return embeddings


def get_embeddings(
    encoder_name: str,
    splits: dict[str, Split],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    cache_dir = args.output_dir.resolve() / "embeddings" / encoder_name
    if encoder_name == "tfidf":
        return encode_tfidf(
            splits,
            cache_dir=cache_dir,
            max_features=args.tfidf_max_features,
            ngram_max=args.tfidf_ngram_max,
            force_recompute=args.force_recompute_embeddings,
        )
    return encode_sentence_model(
        encoder_name,
        splits,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.embedding_batch_size,
        force_recompute=args.force_recompute_embeddings,
    )


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def make_head(
    head_name: str,
    *,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if head_name == "logreg":
        return LinearHead(input_dim, len(LABELS))
    if head_name == "mlp":
        return MLPHead(input_dim, hidden_dim, len(LABELS), dropout)
    raise ValueError(f"Unknown head: {head_name}")


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(train_mode):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def predict(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], list[float]]:
    loader = make_loader(
        features,
        np.zeros(features.shape[0], dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions: list[str] = []
    motion_probabilities: list[float] = []
    with torch.no_grad():
        for batch_features, _ in loader:
            logits = model(batch_features.to(device))
            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = probabilities.argmax(dim=1).cpu().numpy().tolist()
            predictions.extend(INDEX_TO_LABEL[index] for index in batch_predictions)
            motion_probabilities.extend(probabilities[:, LABEL_TO_INDEX["motion_query"]].cpu().numpy().tolist())
    return predictions, motion_probabilities


def compute_metrics(labels: list[str], predictions: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    false_motion = sum(
        1
        for gold, pred in zip(labels, predictions)
        if gold == "chat" and pred == "motion_query"
    )
    false_text = sum(
        1
        for gold, pred in zip(labels, predictions)
        if gold == "motion_query" and pred == "chat"
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "false_motion_count": false_motion,
        "false_text_count": false_text,
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
    split: Split,
    predictions: list[str],
    motion_probabilities: list[float],
) -> None:
    rows = []
    for row_id, text, label, prediction, probability in zip(
        split.ids,
        split.texts,
        split.labels,
        predictions,
        motion_probabilities,
    ):
        rows.append(
            {
                "id": row_id,
                "utterance": text,
                "label": label,
                "prediction": prediction,
                "p_motion_query": f"{probability:.6f}",
            }
        )
    write_csv(
        path,
        rows,
        ["id", "utterance", "label", "prediction", "p_motion_query"],
    )


def plot_history(history_path: Path, image_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_acc: list[float] = []
    val_acc: list[float] = []

    with history_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            val_acc.append(float(row["val_accuracy"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=160)
    plt.close(fig)


def train_one_run(
    encoder_name: str,
    head_name: str,
    splits: dict[str, Split],
    embeddings: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    run_name = f"{encoder_name}__{head_name}"
    run_dir = args.output_dir.resolve() / "runs" / run_name
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        stored_summary = read_json(metrics_path)["summary"]
        return {
            "run_name": run_name,
            "encoder": encoder_name,
            "head": head_name,
            **stored_summary,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    input_dim = int(embeddings["train"].shape[1])
    model = make_head(
        head_name,
        input_dim=input_dim,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(
        embeddings["train"],
        splits["train"].label_indices,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        embeddings["val"],
        splits["val"].label_indices,
        batch_size=args.batch_size,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            device=device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "train_accuracy": f"{train_accuracy:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "val_accuracy": f"{val_accuracy:.8f}",
                "learning_rate": f"{args.lr:.8f}",
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed_seconds = time.perf_counter() - start_time

    history_path = run_dir / "history.csv"
    write_csv(
        history_path,
        history_rows,
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
        ],
    )
    plot_history(history_path, run_dir / "loss_curve.png")

    split_metrics: dict[str, object] = {}
    for split_name, split in splits.items():
        predictions, probabilities = predict(
            model,
            embeddings[split_name],
            device=device,
            batch_size=args.batch_size,
        )
        split_metrics[split_name] = compute_metrics(split.labels, predictions)
        if split_name in {"val", "test"}:
            write_predictions(
                run_dir / f"{split_name}_predictions.csv",
                split,
                predictions,
                probabilities,
            )

    torch.save(
        {
            "encoder": encoder_name,
            "head": head_name,
            "input_dim": input_dim,
            "labels": LABELS,
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
        },
        run_dir / "model.pt",
    )
    run_config = {
        "run_name": run_name,
        "encoder": encoder_name,
        "head": head_name,
        "device": str(device),
        "input_dim": input_dim,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "dropout": args.dropout,
    }
    write_json(run_dir / "run_config.json", run_config)

    summary = {
        "run_name": run_name,
        "encoder": encoder_name,
        "head": head_name,
        "input_dim": input_dim,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "train_accuracy": split_metrics["train"]["accuracy"],
        "val_accuracy": split_metrics["val"]["accuracy"],
        "test_accuracy": split_metrics["test"]["accuracy"],
        "test_macro_f1": split_metrics["test"]["macro_f1"],
        "test_false_motion_count": split_metrics["test"]["false_motion_count"],
        "test_false_text_count": split_metrics["test"]["false_text_count"],
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(
        metrics_path,
        {
            "summary": summary,
            "config": run_config,
            "metrics": split_metrics,
        },
    )
    print(
        f"{run_name}: val={summary['val_accuracy']:.4f} "
        f"test={summary['test_accuracy']:.4f} "
        f"epochs={summary['epochs_completed']}"
    )
    return summary


def write_summary(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return

    fieldnames = [
        "run_name",
        "encoder",
        "head",
        "input_dim",
        "epochs_completed",
        "best_epoch",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_false_motion_count",
        "test_false_text_count",
        "elapsed_seconds",
    ]
    write_csv(output_dir / "summary.csv", summaries, fieldnames)

    sorted_summaries = sorted(
        summaries,
        key=lambda row: float(row.get("test_accuracy", 0.0)),
        reverse=True,
    )
    lines = [
        "# Experiment Summary",
        "",
        "| run | val acc | test acc | test macro F1 | epochs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted_summaries:
        lines.append(
            "| {run_name} | {val_accuracy:.4f} | {test_accuracy:.4f} | "
            "{test_macro_f1:.4f} | {epochs_completed} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_snapshot(
    *,
    output_dir: Path,
    report_dir: Path,
    summaries: list[dict[str, object]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("experiment_plan.json", "summary.csv", "summary.md"):
        source = output_dir / filename
        if source.exists():
            shutil.copy2(source, report_dir / filename)

    for row in summaries:
        run_name = str(row["run_name"])
        source_run_dir = output_dir / "runs" / run_name
        target_run_dir = report_dir / "runs" / run_name
        target_run_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("run_config.json", "history.csv", "metrics.json", "loss_curve.png"):
            source = source_run_dir / filename
            if source.exists():
                shutil.copy2(source, target_run_dir / filename)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    encoders = resolve_choices(args.encoders, ENCODER_CHOICES, "encoders")
    heads = resolve_choices(args.heads, HEAD_CHOICES, "heads")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")

    splits = load_splits(args)
    print(
        "rows: "
        + ", ".join(f"{name}={len(split.texts)}" for name, split in splits.items())
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "experiment_plan.json",
        {
            "encoders": encoders,
            "heads": heads,
            "data_dir": str(args.data_dir),
            "device": str(device),
            "seed": args.seed,
            "artifact_layout": {
                "embedding_cache": "reproduced/artifacts/v1/embeddings/{encoder}/",
                "run_output": "reproduced/artifacts/v1/runs/{encoder}__{head}/",
                "summary": "reproduced/artifacts/v1/summary.csv and summary.md",
            },
        },
    )

    summaries: list[dict[str, object]] = []
    for encoder_name in encoders:
        embeddings = get_embeddings(encoder_name, splits, args, device)
        for head_name in heads:
            summaries.append(
                train_one_run(
                    encoder_name,
                    head_name,
                    splits,
                    embeddings,
                    args,
                    device,
                )
            )
    write_summary(args.output_dir, summaries)
    if not args.no_report_copy:
        write_report_snapshot(
            output_dir=args.output_dir,
            report_dir=args.report_dir.resolve(),
            summaries=summaries,
        )
        print(f"report snapshot: {args.report_dir.resolve()}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
