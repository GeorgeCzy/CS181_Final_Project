"""Demonstrate label/confidence changes as an utterance grows word by word."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
from mlp_head import MLPHead

from shared_utils import word_prefixes


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefix growth demo for one utterance.")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--model",
        choices=("keyword", "tfidf", "mlp"),
        default="mlp",
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
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_predictor(args: argparse.Namespace, device: torch.device):
    if args.model == "keyword":
        from shared_utils import keyword_predict

        return keyword_predict

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
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(metadata["model_name"], device=str(device))
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predict_fn = load_predictor(args, device)

    print(f"model={args.model}")
    print(f"threshold={args.threshold}")
    print(f"full_text={args.text}")
    print("")
    print("prefix_ratio | prefix | label | p_motion | confident? | early_decision")
    print("--- | --- | --- | --- | --- | ---")

    early_decision: str | None = None
    for prefix_text, prefix_ratio in word_prefixes(args.text):
        label, p_motion = predict_fn(prefix_text)
        confidence = p_motion if label == "motion_query" else 1.0 - p_motion
        confident = confidence >= args.threshold
        if early_decision is None and confident:
            early_decision = label
        print(
            f"{prefix_ratio:>11.2f} | {prefix_text!r} | {label} | {p_motion:.4f} | "
            f"{str(confident):>5} | {early_decision or '-'}"
        )

    print("")
    print(
        json.dumps(
            {
                "label": label,
                "p_motion_query": p_motion,
                "early_decision_at_threshold": early_decision,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
