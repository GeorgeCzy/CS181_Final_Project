"""Run inference with the trained TF-IDF logistic regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "modeling" / "artifacts" / "tfidf_logreg" / "model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict response mode for text.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text", default=None, help="Single utterance to classify.")
    return parser.parse_args()


def predict(model, text: str) -> tuple[str, float]:
    label = str(model.predict([text])[0])
    class_index = list(model.classes_).index("motion_query")
    p_motion = float(model.predict_proba([text])[0][class_index])
    return label, p_motion


def print_prediction(model, text: str) -> None:
    label, p_motion = predict(model, text)
    print(f"text: {text}")
    print(f"label: {label}")
    print(f"p_motion_query: {p_motion:.4f}")


def run_interactive(model) -> None:
    print("Enter a human utterance. Press Ctrl+C or submit an empty line to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        print_prediction(model, text)


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. "
            "Run modeling/scripts/train_tfidf_logreg.py first."
        )
    model = joblib.load(args.model_path)

    if args.text:
        print_prediction(model, args.text)
    else:
        run_interactive(model)


if __name__ == "__main__":
    main()

