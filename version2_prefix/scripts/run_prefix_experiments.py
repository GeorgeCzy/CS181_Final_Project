"""Run version2 prefix-based three-class experiments.

Labels:
- chat
- motion_query
- wait

This implementation is isolated from both `experiments/` and
`mode-classifier-main/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "version2_prefix" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "version2_prefix"
DEFAULT_REPORT_DIR = ROOT / "version2_prefix" / "results" / "latest"

LABELS = ["chat", "motion_query", "wait"]
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
    full_labels: list[str]
    target_prefix_ratios: list[str]
    visible_ratios: list[float]
    prefix_tokens: list[int]
    total_tokens: list[int]
    full_utterances: list[str]

    @property
    def label_indices(self) -> np.ndarray:
        return np.array([LABEL_TO_INDEX[label] for label in self.labels], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run version2 prefix-based three-class experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--text-column", default="prefix_utterance")
    parser.add_argument("--train-file", default="prefix_train.csv")
    parser.add_argument("--val-file", default="prefix_val.csv")
    parser.add_argument("--test-file", default="prefix_test.csv")
    parser.add_argument("--encoders", nargs="+", default=["all"])
    parser.add_argument("--heads", nargs="+", default=["all"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--tfidf-max-features", type=int, default=15000)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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


def read_split(path: Path, *, text_column: str, limit: int | None = None) -> Split:
    if not path.exists():
        raise FileNotFoundError(f"Missing prefix data file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{path} has no rows.")

    required = {
        "id",
        text_column,
        "label",
        "full_label",
        "target_prefix_ratio",
        "visible_ratio",
        "prefix_tokens",
        "total_tokens",
        "full_utterance",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    full_labels: list[str] = []
    target_prefix_ratios: list[str] = []
    visible_ratios: list[float] = []
    prefix_tokens: list[int] = []
    total_tokens: list[int] = []
    full_utterances: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        label = row["label"]
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"{path}:{line_number} has invalid label {label!r}")
        ids.append(row["id"])
        texts.append(row[text_column])
        labels.append(label)
        full_labels.append(row["full_label"])
        target_prefix_ratios.append(row["target_prefix_ratio"])
        visible_ratios.append(float(row["visible_ratio"]))
        prefix_tokens.append(int(row["prefix_tokens"]))
        total_tokens.append(int(row["total_tokens"]))
        full_utterances.append(row["full_utterance"])

    return Split(
        name=path.stem,
        ids=ids,
        texts=texts,
        labels=labels,
        full_labels=full_labels,
        target_prefix_ratios=target_prefix_ratios,
        visible_ratios=visible_ratios,
        prefix_tokens=prefix_tokens,
        total_tokens=total_tokens,
        full_utterances=full_utterances,
    )


def load_splits(args: argparse.Namespace) -> dict[str, Split]:
    data_dir = args.data_dir.resolve()
    splits = {
        "train": read_split(data_dir / args.train_file, text_column=args.text_column, limit=args.limit_train),
        "val": read_split(data_dir / args.val_file, text_column=args.text_column, limit=args.limit_val),
        "test": read_split(data_dir / args.test_file, text_column=args.text_column, limit=args.limit_test),
    }
    if len(splits["train"].texts) < args.min_train_rows:
        raise ValueError(
            f"Training split has {len(splits['train'].texts)} rows, below "
            f"--min-train-rows={args.min_train_rows}."
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
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def make_head(head_name: str, *, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if head_name == "logreg":
        return LinearHead(input_dim, len(LABELS))
    if head_name == "mlp":
        return MLPHead(input_dim, hidden_dim, len(LABELS), dropout)
    raise ValueError(f"Unknown head: {head_name}")


def make_loader(features: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
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
) -> tuple[list[str], list[float], list[float], list[float]]:
    loader = make_loader(
        features,
        np.zeros(features.shape[0], dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions: list[str] = []
    p_chat: list[float] = []
    p_motion: list[float] = []
    p_wait: list[float] = []
    with torch.no_grad():
        for batch_features, _ in loader:
            probabilities = torch.softmax(model(batch_features.to(device)), dim=1).cpu().numpy()
            indices = probabilities.argmax(axis=1).tolist()
            predictions.extend(INDEX_TO_LABEL[index] for index in indices)
            p_chat.extend(probabilities[:, LABEL_TO_INDEX["chat"]].tolist())
            p_motion.extend(probabilities[:, LABEL_TO_INDEX["motion_query"]].tolist())
            p_wait.extend(probabilities[:, LABEL_TO_INDEX["wait"]].tolist())
    return predictions, p_chat, p_motion, p_wait


def compute_metrics(labels: list[str], predictions: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0),
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


def metrics_by_prefix_ratio(split: Split, predictions: list[str]) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, ratio in enumerate(split.target_prefix_ratios):
        grouped[ratio].append(index)

    rows: list[dict[str, object]] = []
    for ratio in sorted(grouped, key=lambda value: float(value)):
        indices = grouped[ratio]
        gold = [split.labels[index] for index in indices]
        pred = [predictions[index] for index in indices]
        non_wait_indices = [index for index, label in enumerate(gold) if label != "wait"]
        decision_indices = [index for index, label in enumerate(pred) if label != "wait"]
        non_wait_accuracy = ""
        if non_wait_indices:
            non_wait_accuracy = accuracy_score(
                [gold[index] for index in non_wait_indices],
                [pred[index] for index in non_wait_indices],
            )
        decision_accuracy = ""
        if decision_indices:
            decision_accuracy = accuracy_score(
                [gold[index] for index in decision_indices],
                [pred[index] for index in decision_indices],
            )
        rows.append(
            {
                "target_prefix_ratio": ratio,
                "num_examples": len(indices),
                "accuracy": accuracy_score(gold, pred),
                "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
                "gold_wait_rate": sum(label == "wait" for label in gold) / len(gold),
                "pred_wait_rate": sum(label == "wait" for label in pred) / len(pred),
                "pred_decision_rate": sum(label != "wait" for label in pred) / len(pred),
                "non_wait_gold_accuracy": non_wait_accuracy,
                "decision_accuracy": decision_accuracy,
            }
        )
    return rows


def write_predictions(
    path: Path,
    split: Split,
    predictions: list[str],
    p_chat: list[float],
    p_motion: list[float],
    p_wait: list[float],
) -> None:
    rows = []
    for index in range(len(split.ids)):
        rows.append(
            {
                "id": split.ids[index],
                "text": split.texts[index],
                "label": split.labels[index],
                "prediction": predictions[index],
                "full_label": split.full_labels[index],
                "target_prefix_ratio": split.target_prefix_ratios[index],
                "visible_ratio": f"{split.visible_ratios[index]:.6f}",
                "prefix_tokens": split.prefix_tokens[index],
                "total_tokens": split.total_tokens[index],
                "p_chat": f"{p_chat[index]:.6f}",
                "p_motion_query": f"{p_motion[index]:.6f}",
                "p_wait": f"{p_wait[index]:.6f}",
                "full_utterance": split.full_utterances[index],
            }
        )
    write_csv(
        path,
        rows,
        [
            "id",
            "text",
            "label",
            "prediction",
            "full_label",
            "target_prefix_ratio",
            "visible_ratio",
            "prefix_tokens",
            "total_tokens",
            "p_chat",
            "p_motion_query",
            "p_wait",
            "full_utterance",
        ],
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


def plot_prefix_metrics(prefix_metrics_path: Path, image_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ratios: list[float] = []
    accuracy: list[float] = []
    decision_rate: list[float] = []
    pred_wait_rate: list[float] = []
    with prefix_metrics_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            ratios.append(float(row["target_prefix_ratio"]))
            accuracy.append(float(row["accuracy"]))
            decision_rate.append(float(row["pred_decision_rate"]))
            pred_wait_rate.append(float(row["pred_wait_rate"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, accuracy, marker="o", label="accuracy")
    ax.plot(ratios, decision_rate, marker="o", label="decision rate")
    ax.plot(ratios, pred_wait_rate, marker="o", label="wait rate")
    ax.set_xlabel("target prefix ratio")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
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
        return read_json(metrics_path)["summary"]

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion=criterion, device=device)
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
        ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"],
    )
    plot_history(history_path, run_dir / "loss_curve.png")

    split_metrics: dict[str, object] = {}
    prefix_ratio_metrics: dict[str, list[dict[str, object]]] = {}
    for split_name, split in splits.items():
        predictions, p_chat, p_motion, p_wait = predict(
            model,
            embeddings[split_name],
            device=device,
            batch_size=args.batch_size,
        )
        split_metrics[split_name] = compute_metrics(split.labels, predictions)
        prefix_ratio_metrics[split_name] = metrics_by_prefix_ratio(split, predictions)
        if split_name in {"val", "test"}:
            write_predictions(
                run_dir / f"{split_name}_predictions.csv",
                split,
                predictions,
                p_chat,
                p_motion,
                p_wait,
            )
            prefix_metrics_path = run_dir / f"{split_name}_metrics_by_prefix_ratio.csv"
            write_csv(
                prefix_metrics_path,
                prefix_ratio_metrics[split_name],
                [
                    "target_prefix_ratio",
                    "num_examples",
                    "accuracy",
                    "macro_f1",
                    "gold_wait_rate",
                    "pred_wait_rate",
                    "pred_decision_rate",
                    "non_wait_gold_accuracy",
                    "decision_accuracy",
                ],
            )
            plot_prefix_metrics(
                prefix_metrics_path,
                run_dir / f"{split_name}_prefix_metrics.png",
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
        "task": "version2_prefix_three_class",
        "text_column": args.text_column,
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

    test_ratio_rows = {
        row["target_prefix_ratio"]: row
        for row in prefix_ratio_metrics["test"]
    }
    final_ratio = test_ratio_rows.get("1.00", {})
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
        "test_final_ratio_accuracy": final_ratio.get("accuracy", ""),
        "test_avg_pred_decision_rate": float(np.mean([row["pred_decision_rate"] for row in prefix_ratio_metrics["test"]])),
        "test_avg_pred_wait_rate": float(np.mean([row["pred_wait_rate"] for row in prefix_ratio_metrics["test"]])),
        "elapsed_seconds": elapsed_seconds,
    }

    write_json(run_dir / "run_config.json", run_config)
    write_json(
        metrics_path,
        {
            "summary": summary,
            "config": run_config,
            "metrics": split_metrics,
            "metrics_by_prefix_ratio": prefix_ratio_metrics,
        },
    )
    print(
        f"{run_name}: val={summary['val_accuracy']:.4f} "
        f"test={summary['test_accuracy']:.4f} "
        f"final={summary['test_final_ratio_accuracy']:.4f} "
        f"epochs={summary['epochs_completed']}"
    )
    return summary


def write_summary(output_dir: Path, summaries: list[dict[str, object]]) -> None:
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
        "test_final_ratio_accuracy",
        "test_avg_pred_decision_rate",
        "test_avg_pred_wait_rate",
        "elapsed_seconds",
    ]
    write_csv(output_dir / "summary.csv", summaries, fieldnames)
    sorted_summaries = sorted(
        summaries,
        key=lambda row: float(row.get("test_accuracy", 0.0)),
        reverse=True,
    )
    lines = [
        "# Version2 Prefix Experiment Summary",
        "",
        "| run | val acc | test acc | final-prefix acc | avg decision rate | epochs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_summaries:
        lines.append(
            "| {run_name} | {val_accuracy:.4f} | {test_accuracy:.4f} | "
            "{test_final_ratio_accuracy:.4f} | {test_avg_pred_decision_rate:.4f} | "
            "{epochs_completed} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_snapshot(output_dir: Path, report_dir: Path, summaries: list[dict[str, object]]) -> None:
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
        for filename in (
            "run_config.json",
            "history.csv",
            "loss_curve.png",
            "metrics.json",
            "test_metrics_by_prefix_ratio.csv",
            "test_prefix_metrics.png",
            "val_metrics_by_prefix_ratio.csv",
            "val_prefix_metrics.png",
        ):
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
    print("rows: " + ", ".join(f"{name}={len(split.texts)}" for name, split in splits.items()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "experiment_plan.json",
        {
            "version": "version2_prefix",
            "labels": LABELS,
            "input_column": args.text_column,
            "encoders": encoders,
            "heads": heads,
            "data_dir": str(args.data_dir),
            "device": str(device),
            "artifact_layout": {
                "embedding_cache": "artifacts/version2_prefix/embeddings/{encoder}/",
                "run_output": "artifacts/version2_prefix/runs/{encoder}__{head}/",
                "report_snapshot": "version2_prefix/results/latest/",
            },
        },
    )

    summaries: list[dict[str, object]] = []
    for encoder_name in encoders:
        embeddings = get_embeddings(encoder_name, splits, args, device)
        for head_name in heads:
            summaries.append(
                train_one_run(encoder_name, head_name, splits, embeddings, args, device)
            )
    write_summary(args.output_dir, summaries)
    if not args.no_report_copy:
        write_report_snapshot(args.output_dir, args.report_dir.resolve(), summaries)
        print(f"report snapshot: {args.report_dir.resolve()}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
