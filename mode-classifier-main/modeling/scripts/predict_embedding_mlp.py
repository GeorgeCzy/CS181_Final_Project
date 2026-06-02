"""Run inference with a cached-embedding MLP classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mlp_head import MLPHead


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "modeling" / "artifacts" / "embedding_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict response mode with embedding + MLP.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text", default=None)
    return parser.parse_args()


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing model config: {config_path}. Run train_embedding_mlp.py first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_sentence_model(config: dict, embedding_model: str | None, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
        ) from error

    metadata = config.get("embedding_metadata", {})
    model_name = embedding_model or metadata.get("model_name")
    if not model_name:
        raise ValueError("Embedding model name is missing. Pass --embedding-model.")
    model = SentenceTransformer(model_name, device=device)
    max_seq_length = metadata.get("max_seq_length")
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
    return model


def encode_text(sentence_model, config: dict, text: str) -> torch.Tensor:
    metadata = config.get("embedding_metadata", {})
    prefix = metadata.get("text_prefix", "")
    normalize = bool(metadata.get("normalize_embeddings", True))
    model_input = f"{prefix}{text}" if prefix else text
    embedding = sentence_model.encode(
        [model_input],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype("float32")
    return torch.from_numpy(embedding)


def print_prediction(classifier: MLPHead, sentence_model, config: dict, text: str, device: torch.device) -> None:
    classifier.eval()
    inputs = encode_text(sentence_model, config, text).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier(inputs), dim=1)[0]
    p_motion = float(probabilities[1].item())
    label = "motion_query" if p_motion >= 0.5 else "chat"
    print(f"text: {text}")
    print(f"label: {label}")
    print(f"p_motion_query: {p_motion:.4f}")


def interactive_loop(classifier: MLPHead, sentence_model, config: dict, device: torch.device) -> None:
    print("Enter a human utterance. Press Ctrl+C or submit an empty line to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        print_prediction(classifier, sentence_model, config, text, device)


def main() -> None:
    args = parse_args()
    config = load_config(args.model_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier = MLPHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    classifier.load_state_dict(
        torch.load(args.model_dir / "model.pt", map_location=device)
    )
    sentence_model = load_sentence_model(config, args.embedding_model, args.device)

    if args.text:
        print_prediction(classifier, sentence_model, config, args.text, device)
    else:
        interactive_loop(classifier, sentence_model, config, device)


if __name__ == "__main__":
    main()

