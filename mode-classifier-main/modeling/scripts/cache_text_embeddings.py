"""Cache text embeddings for the response-mode dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data_generation" / "data" / "raw" / "deepseek_generated_10000.csv"
DEFAULT_OUTPUT = ROOT / "modeling" / "data" / "embeddings" / "qwen3_0_6b_deepseek_10000.npz"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TEXT_PREFIX = "Represent this human-robot utterance for response mode classification: "
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached text embeddings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--encoder-backend",
        choices=("sentence-transformers", "clip"),
        default="sentence-transformers",
        help="How to load and encode text. Use clip for CLIP text encoders.",
    )
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--text-prefix", default=DEFAULT_TEXT_PREFIX)
    parser.add_argument(
        "--no-text-prefix",
        action="store_true",
        help="Encode utterances without a task prefix.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write metadata for an existing .npz cache.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, start=1):
        if set(row) != {"id", "utterance", "label"}:
            raise ValueError(f"Row {index} has unexpected fields: {sorted(row)}")
    return rows


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def default_max_seq_length(encoder_backend: str) -> int:
    return 77 if encoder_backend == "clip" else 512


def encode_with_sentence_transformer(
    texts: list[str],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
    max_seq_length: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype("float32")


def encode_with_clip(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "text_embeds") and features.text_embeds is not None:
                    features = features.text_embeds
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                else:
                    raise TypeError(f"Unexpected CLIP text feature type: {type(features)}")
            if normalize_embeddings:
                features = features / features.norm(dim=-1, keepdim=True)
        chunks.append(features.cpu().numpy().astype("float32"))
    return np.vstack(chunks)


def write_metadata(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    encoder_backend: str,
    device: str | None,
    batch_size: int,
    max_seq_length: int,
    text_prefix: str,
    normalize_embeddings: bool,
    rows: int,
    embedding_dim: int,
) -> None:
    metadata = {
        "input": display_path(input_path),
        "output": display_path(output_path),
        "model_name": model_name,
        "encoder_backend": encoder_backend,
        "device": device,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "text_prefix": text_prefix,
        "normalize_embeddings": normalize_embeddings,
        "rows": rows,
        "embedding_dim": embedding_dim,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote metadata: {output_path.with_suffix('.json')}")


def resolve_device(requested: str | None, encoder_backend: str) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def main() -> None:
    args = parse_args()
    if args.no_text_prefix:
        args.text_prefix = ""
    if args.encoder_backend == "clip" and args.model_name == DEFAULT_MODEL:
        args.model_name = DEFAULT_CLIP_MODEL
    max_seq_length = args.max_seq_length or default_max_seq_length(args.encoder_backend)
    device = resolve_device(args.device, args.encoder_backend)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.metadata_only:
        if not args.output.exists():
            raise FileNotFoundError(f"Embedding cache does not exist: {args.output}")
        cache = np.load(args.output, allow_pickle=False)
        write_metadata(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model_name,
            encoder_backend=args.encoder_backend,
            device=device,
            batch_size=args.batch_size,
            max_seq_length=max_seq_length,
            text_prefix=args.text_prefix,
            normalize_embeddings=not args.no_normalize,
            rows=int(cache["embeddings"].shape[0]),
            embedding_dim=int(cache["embeddings"].shape[1]),
        )
        return

    rows = read_rows(args.input)
    ids = np.array([row["id"] for row in rows], dtype=str)
    labels = np.array([row["label"] for row in rows], dtype=str)
    texts = [
        f"{args.text_prefix}{row['utterance']}" if args.text_prefix else row["utterance"]
        for row in rows
    ]

    normalize_embeddings = not args.no_normalize
    if args.encoder_backend == "clip":
        embeddings = encode_with_clip(
            texts,
            model_name=args.model_name,
            device=device,
            batch_size=args.batch_size,
            max_seq_length=max_seq_length,
            normalize_embeddings=normalize_embeddings,
        )
    else:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from error
        embeddings = encode_with_sentence_transformer(
            texts,
            model_name=args.model_name,
            device=device,
            batch_size=args.batch_size,
            max_seq_length=max_seq_length,
            normalize_embeddings=normalize_embeddings,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=ids,
        labels=labels,
        embeddings=embeddings,
    )

    write_metadata(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        encoder_backend=args.encoder_backend,
        device=device,
        batch_size=args.batch_size,
        max_seq_length=max_seq_length,
        text_prefix=args.text_prefix,
        normalize_embeddings=normalize_embeddings,
        rows=len(rows),
        embedding_dim=int(embeddings.shape[1]),
    )
    print(f"Wrote embeddings: {args.output}")
    print(f"shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
