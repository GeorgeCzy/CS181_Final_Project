"""Train a small MLP head on cached text embeddings.

Expected embedding cache format: an .npz file with arrays:

- ids: string sample ids
- embeddings: float matrix with shape [num_examples, embedding_dim]

The labels and split membership are read from the CSV split files created by
create_splits.py. This keeps expensive embedding generation separate from MLP
training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embedding_utils import load_embedding_cache
from mlp_head import MLPHead, evaluate
from wandb_utils import add_wandb_args, init_wandb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "modeling" / "artifacts" / "embedding_mlp"
LABEL_TO_INDEX = {"chat": 0, "motion_query": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on cached embeddings.")
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Log batch loss to W&B every N optimizer steps.",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def split_to_tensors(
    rows: list[dict[str, str]],
    id_to_index: dict[str, int],
    embeddings: np.ndarray,
) -> TensorDataset:
    indices = [id_to_index[row["id"]] for row in rows]
    labels = [LABEL_TO_INDEX[row["label"]] for row in rows]
    x = torch.from_numpy(embeddings[indices])
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_index, embeddings = load_embedding_cache(args.embedding_cache)
    train_rows = read_split(args.split_dir / "train.csv")
    val_rows = read_split(args.split_dir / "val.csv")
    test_rows = read_split(args.split_dir / "test.csv")

    train_dataset = split_to_tensors(train_rows, id_to_index, embeddings)
    val_dataset = split_to_tensors(val_rows, id_to_index, embeddings)
    test_dataset = split_to_tensors(test_rows, id_to_index, embeddings)
    run = init_wandb(
        args,
        config={
            "architecture": "CachedEmbedding + MLPHead",
            "embedding_cache": str(args.embedding_cache),
            "embedding_dim": int(embeddings.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "log_every_n_steps": args.log_every_n_steps,
            "device": str(device),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
    )

    model = MLPHead(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_val_accuracy = -1.0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            if run and args.log_every_n_steps > 0 and global_step % args.log_every_n_steps == 0:
                run.log(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "train/batch_loss": float(loss.item()),
                    },
                    step=global_step,
                )

        train_metrics = evaluate(model, train_dataset, device)
        val_metrics = evaluate(model, val_dataset, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(
            f"epoch={epoch} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if run:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                },
                step=global_step,
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

    test_metrics = evaluate(model, test_dataset, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    model_config = {
        "architecture": "CachedEmbedding + MLPHead",
        "head": "mlp",
        "embedding_cache": str(args.embedding_cache),
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "labels": INDEX_TO_LABEL,
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
                "best_val_accuracy": best_val_accuracy,
                "test": test_metrics,
                "history": history,
                "labels": INDEX_TO_LABEL,
                "embedding_dim": int(embeddings.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
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
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
            },
            step=global_step + 1,
        )
        run.finish()


if __name__ == "__main__":
    main()
