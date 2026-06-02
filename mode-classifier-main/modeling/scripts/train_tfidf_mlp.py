"""Train an MLP head on TF-IDF features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlp_head import MLPHead, evaluate as evaluate_mlp
from wandb_utils import add_wandb_args, init_wandb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "tfidf_mlp"
LABEL_TO_INDEX = {"chat": 0, "motion_query": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + MLP classifier.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    add_wandb_args(parser)
    return parser.parse_args()


def read_split(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    texts = [row["utterance"] for row in rows]
    labels = [LABEL_TO_INDEX[row["label"]] for row in rows]
    return texts, labels


def texts_to_dataset(
    vectorizer: TfidfVectorizer,
    texts: list[str],
    labels: list[int],
) -> TensorDataset:
    features = vectorizer.transform(texts).toarray().astype("float32")
    x = torch.from_numpy(features)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_texts, train_labels = read_split(args.split_dir / "train.csv")
    val_texts, val_labels = read_split(args.split_dir / "val.csv")
    test_texts, test_labels = read_split(args.split_dir / "test.csv")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
    )
    vectorizer.fit(train_texts)

    train_dataset = texts_to_dataset(vectorizer, train_texts, train_labels)
    val_dataset = texts_to_dataset(vectorizer, val_texts, val_labels)
    test_dataset = texts_to_dataset(vectorizer, test_texts, test_labels)
    input_dim = int(train_dataset.tensors[0].shape[1])

    run = init_wandb(
        args,
        config={
            "architecture": "TF-IDF + MLPHead",
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(device),
        },
    )

    model = MLPHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_val_accuracy = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_mlp(model, train_dataset, device)
        val_metrics = evaluate_mlp(model, val_dataset, device)
        print(
            f"epoch={epoch} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if run:
            run.log(
                {
                    "epoch": epoch,
                    "train/accuracy": train_metrics["accuracy"],
                    "val/accuracy": val_metrics["accuracy"],
                },
                step=epoch,
            )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_mlp(model, test_dataset, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.output_dir / "vectorizer.joblib")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "architecture": "TF-IDF + MLPHead",
                "encoder": "tfidf",
                "head": "mlp",
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "max_features": args.max_features,
                "ngram_max": args.ngram_max,
                "labels": {"0": "chat", "1": "motion_query"},
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_accuracy": best_val_accuracy,
                "test": test_metrics,
                "input_dim": input_dim,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")
    if run:
        run.log(
            {
                "best_val/accuracy": best_val_accuracy,
                "test/accuracy": test_metrics["accuracy"],
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
