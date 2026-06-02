"""Optional Weights & Biases helpers for training scripts."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_ENTITY = "chengzy2023-shanghaitech-university"
DEFAULT_WANDB_PROJECT = "mode-classifier"


def add_wandb_args(parser: ArgumentParser) -> None:
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )


def init_wandb(args: Namespace, *, config: dict[str, Any]):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    wandb_dir = ROOT / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_dir / "cache"))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_dir / "data"))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(wandb_dir / "artifacts"))

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=config,
    )
