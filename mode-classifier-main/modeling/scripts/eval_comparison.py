"""Compare 4 text encoders × 2 classifier heads on standard and hard splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embedding_utils import (
    make_embedding_predictor,
    make_tfidf_logreg_predictor,
    make_tfidf_mlp_predictor,
)
from shared_utils import (
    collect_predictions,
    compute_metrics,
    make_keyword_predictor,
    read_split,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "modeling" / "data" / "splits"
DEFAULT_HARD_TEST = ROOT / "modeling" / "data" / "hard_test.csv"
DEFAULT_REPORT = ROOT / "modeling" / "reports" / "comparison_summary.json"
DEFAULT_MODELS = [
    "tfidf+logreg:modeling/artifacts/tfidf_logreg",
    "tfidf+mlp:modeling/artifacts/tfidf_mlp",
    "clip+logreg:modeling/artifacts/clip_logreg",
    "clip+mlp:modeling/artifacts/clip_mlp",
    "qwen+logreg:modeling/artifacts/qwen_logreg",
    "qwen+mlp:modeling/artifacts/qwen_mlp",
    "minilm+logreg:modeling/artifacts/minilm_logreg",
    "minilm+mlp:modeling/artifacts/minilm_mlp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate encoder × head combinations."
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--hard-test", type=Path, default=DEFAULT_HARD_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Entries like name:relative/path/to/model_dir",
    )
    parser.add_argument(
        "--include-keyword",
        action="store_true",
        help="Also evaluate the keyword rule baseline.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid model spec: {spec}")
        name, rel_path = spec.split(":", 1)
        parsed.append((name.strip(), (ROOT / rel_path.strip()).resolve()))
    return parsed


def resolve_predictor(name: str, model_dir: Path, device: torch.device):
    if name == "tfidf+logreg":
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return make_tfidf_logreg_predictor(model_path)

    if name == "tfidf+mlp":
        return make_tfidf_mlp_predictor(model_dir, device)

    return make_embedding_predictor(model_dir, device)


def evaluate_model(name: str, predict_fn, splits: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    results: dict[str, object] = {"model": name}
    for split_name, rows in splits.items():
        gold, predictions, motion_probs = collect_predictions(rows, predict_fn)
        results[split_name] = compute_metrics(gold, predictions, motion_probs)
    return results


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Model Comparison Summary",
        "",
        "Encoders: **TF-IDF**, **CLIP**, **Qwen3-Embedding-0.6B**, **MiniLM**.",
        "Heads: **Logistic Regression**, **MLP**.",
        "",
    ]

    models = summary["models"]
    matrix_rows = [
        entry for entry in models if "+" in entry["model"]
    ]
    if matrix_rows:
        lines.append("## Test accuracy (4 × 2)")
        lines.append("")
        lines.append("| Encoder | LogReg | MLP |")
        lines.append("| --- | ---: | ---: |")
        for encoder in ("tfidf", "clip", "qwen", "minilm"):
            logreg = next(
                (entry["test"]["accuracy"] for entry in matrix_rows if entry["model"] == f"{encoder}+logreg"),
                None,
            )
            mlp = next(
                (entry["test"]["accuracy"] for entry in matrix_rows if entry["model"] == f"{encoder}+mlp"),
                None,
            )
            logreg_cell = f"{logreg:.4f}" if logreg is not None else "—"
            mlp_cell = f"{mlp:.4f}" if mlp is not None else "—"
            lines.append(f"| {encoder} | {logreg_cell} | {mlp_cell} |")
        lines.append("")

    for model_entry in models:
        name = model_entry["model"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for split_name in ("test", "hard", "val"):
            if split_name not in model_entry:
                continue
            metrics = model_entry[split_name]
            lines.append(
                f"| {split_name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['false_motion_rate']:.4f} | {metrics['false_text_rate']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    splits: dict[str, list[dict[str, str]]] = {
        "val": read_split(args.split_dir / "val.csv"),
        "test": read_split(args.split_dir / "test.csv"),
    }
    if args.hard_test.exists():
        splits["hard"] = read_split(args.hard_test)

    model_results: list[dict[str, object]] = []

    if args.include_keyword:
        model_results.append(
            evaluate_model("keyword", make_keyword_predictor(), splits)
        )

    for name, model_dir in parse_model_specs(args.models):
        if name == "tfidf+logreg":
            model_path = model_dir / "model.joblib"
            if not model_path.exists():
                print(f"Skipping '{name}' (missing): {model_path}")
                continue
        elif not model_dir.exists():
            print(f"Skipping '{name}' (missing): {model_dir}")
            continue

        try:
            predict_fn = resolve_predictor(name, model_dir, device)
        except FileNotFoundError as error:
            print(f"Skipping '{name}' (missing artifact): {error}")
            continue

        model_results.append(
            evaluate_model(name, predict_fn, splits)
        )

    summary = {"models": model_results}
    write_json(args.output, summary)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote comparison report to {args.output}")
    print(f"Wrote markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
