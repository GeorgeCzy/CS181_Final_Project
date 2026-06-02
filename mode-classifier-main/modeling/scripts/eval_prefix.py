"""Evaluate prefix (early) intent prediction with latency and error-tradeoff metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import torch
from mlp_head import MLPHead

from shared_utils import (
    compute_metrics,
    make_keyword_predictor,
    read_split,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIXES = ROOT / "modeling" / "data" / "prefixes" / "test_prefixes.csv"
DEFAULT_OUTPUT = ROOT / "modeling" / "reports" / "prefix_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on prefix inputs.")
    parser.add_argument("--prefix-file", type=Path, default=DEFAULT_PREFIXES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=("keyword", "tfidf", "mlp"),
        required=True,
    )
    parser.add_argument(
        "--mlp-model-dir",
        type=Path,
        default=ROOT / "modeling" / "artifacts" / "embedding_mlp",
    )
    parser.add_argument(
        "--tfidf-model",
        type=Path,
        default=ROOT / "modeling" / "artifacts" / "tfidf_logreg" / "model.joblib",
    )
    parser.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated confidence thresholds for early-decision analysis.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_predictor(args: argparse.Namespace, device: torch.device):
    if args.model == "keyword":
        return make_keyword_predictor()

    if args.model == "tfidf":
        model = joblib.load(args.tfidf_model)
        class_index = list(model.classes_).index("motion_query")

        def predict(text: str) -> tuple[str, float]:
            label = str(model.predict([text])[0])
            p_motion = float(model.predict_proba([text])[0][class_index])
            return label, p_motion

        return predict

    config = json.loads((args.mlp_model_dir / "model_config.json").read_text(encoding="utf-8"))
    metadata = config.get("embedding_metadata", {})
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("MLP config is missing embedding_metadata.model_name")

    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(model_name, device=str(device))
    max_seq_length = metadata.get("max_seq_length")
    if max_seq_length:
        sentence_model.max_seq_length = int(max_seq_length)

    classifier = MLPHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    classifier.load_state_dict(
        torch.load(args.mlp_model_dir / "model.pt", map_location=device)
    )
    classifier.eval()
    prefix = metadata.get("text_prefix", "")
    normalize = bool(metadata.get("normalize_embeddings", True))

    def predict(text: str) -> tuple[str, float]:
        model_input = f"{prefix}{text}" if prefix else text
        embedding = sentence_model.encode(
            [model_input],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype("float32")
        inputs = torch.from_numpy(embedding).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(classifier(inputs), dim=1)[0]
        p_motion = float(probabilities[1].item())
        label = "motion_query" if p_motion >= 0.5 else "chat"
        return label, p_motion

    return predict


def group_by_source(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    for source_rows in grouped.values():
        source_rows.sort(key=lambda row: float(row["prefix_ratio"]))
    return grouped


def evaluate_by_ratio(rows: list[dict[str, str]], predict_fn) -> dict[str, object]:
    by_ratio: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ratio[row["target_ratio"]].append(row)

    ratio_metrics: dict[str, object] = {}
    for ratio, ratio_rows in sorted(by_ratio.items(), key=lambda item: float(item[0])):
        gold = [row["label"] for row in ratio_rows]
        predictions: list[str] = []
        probs: list[float] = []
        for row in ratio_rows:
            text = row["prefix_utterance"]
            label, p_motion = predict_fn(text)
            predictions.append(label)
            probs.append(p_motion)
        ratio_metrics[ratio] = compute_metrics(gold, predictions, probs)
    return ratio_metrics


def early_decision_stats(
    grouped_rows: dict[str, list[dict[str, str]]],
    predict_fn,
    thresholds: list[float],
) -> dict[str, object]:
    stats: dict[str, object] = {}
    for threshold in thresholds:
        correct_latencies: list[float] = []
        uncertain_count = 0
        false_motion = 0
        false_text = 0
        total = len(grouped_rows)

        for source_rows in grouped_rows.values():
            gold_label = source_rows[0]["label"]
            decided = False
            for row in source_rows:
                text = row["prefix_utterance"]
                prediction, p_motion = predict_fn(text)
                confidence = p_motion if prediction == "motion_query" else 1.0 - p_motion
                if confidence < threshold:
                    continue
                decided = True
                prefix_ratio = float(row["prefix_ratio"])
                if prediction == gold_label:
                    correct_latencies.append(prefix_ratio)
                elif prediction == "motion_query" and gold_label == "chat":
                    false_motion += 1
                elif prediction == "chat" and gold_label == "motion_query":
                    false_text += 1
                break
            if not decided:
                uncertain_count += 1

        stats[str(threshold)] = {
            "threshold": threshold,
            "num_sources": total,
            "uncertain_rate": uncertain_count / max(total, 1),
            "mean_correct_decision_latency": (
                sum(correct_latencies) / len(correct_latencies) if correct_latencies else None
            ),
            "median_correct_decision_latency": (
                sorted(correct_latencies)[len(correct_latencies) // 2]
                if correct_latencies
                else None
            ),
            "false_motion_rate": false_motion / max(total, 1),
            "false_text_rate": false_text / max(total, 1),
        }
    return stats


def write_curve_csv(path: Path, ratio_metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_ratio",
                "accuracy",
                "macro_f1",
                "false_motion_rate",
                "false_text_rate",
            ],
        )
        writer.writeheader()
        for ratio, metrics in sorted(ratio_metrics.items(), key=lambda item: float(item[0])):
            writer.writerow(
                {
                    "target_ratio": ratio,
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "macro_f1": f"{metrics['macro_f1']:.6f}",
                    "false_motion_rate": f"{metrics['false_motion_rate']:.6f}",
                    "false_text_rate": f"{metrics['false_text_rate']:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    if not args.prefix_file.exists():
        raise FileNotFoundError(
            f"Prefix file not found: {args.prefix_file}. "
            "Run build_prefix_dataset.py first."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predict_fn = load_predictor(args, device)
    rows = read_split(args.prefix_file)
    grouped = group_by_source(rows)
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]

    ratio_metrics = evaluate_by_ratio(rows, predict_fn)
    overall_gold = [row["label"] for row in rows]
    overall_predictions: list[str] = []
    overall_probs: list[float] = []
    for row in rows:
        prediction, p_motion = predict_fn(row["prefix_utterance"])
        overall_predictions.append(prediction)
        overall_probs.append(p_motion)

    payload = {
        "model": args.model,
        "mlp_model_dir": str(args.mlp_model_dir) if args.model == "mlp" else None,
        "prefix_file": str(args.prefix_file),
        "overall": compute_metrics(overall_gold, overall_predictions, overall_probs),
        "by_target_ratio": ratio_metrics,
        "early_decision": early_decision_stats(grouped, predict_fn, thresholds),
    }
    write_json(args.output, payload)
    curve_path = args.output.parent / f"prefix_{args.model}_curve.csv"
    write_curve_csv(curve_path, ratio_metrics)
    print(f"Wrote prefix evaluation to {args.output}")
    print(f"Wrote prefix curve CSV to {curve_path}")


if __name__ == "__main__":
    main()
