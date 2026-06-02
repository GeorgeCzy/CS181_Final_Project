"""Shared helpers for cached embedding models (LogReg and MLP heads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from mlp_head import MLPHead

INDEX_TO_LABEL = {0: "chat", 1: "motion_query"}


def load_embedding_cache(path: Path) -> tuple[dict[str, int], np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    ids = cache["ids"].astype(str).tolist()
    embeddings = cache["embeddings"].astype("float32")
    id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}
    return id_to_index, embeddings


def load_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_text_encoder(metadata: dict, device: torch.device):
    encoder_backend = metadata.get("encoder_backend", "sentence-transformers")
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("Embedding metadata is missing model_name")

    max_seq_length = metadata.get("max_seq_length")
    if encoder_backend == "clip":
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        if max_seq_length:
            processor.tokenizer.model_max_length = int(max_seq_length)
        return ("clip", processor, model)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=str(device))
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
    return ("sentence-transformers", model, None)


def encode_text(
    text: str,
    metadata: dict,
    encoder_bundle,
    device: torch.device,
) -> np.ndarray:
    backend, primary, secondary = encoder_bundle
    prefix = metadata.get("text_prefix", "")
    normalize = bool(metadata.get("normalize_embeddings", True))
    model_input = f"{prefix}{text}" if prefix else text
    max_seq_length = int(metadata.get("max_seq_length") or (77 if backend == "clip" else 512))

    if backend == "clip":
        processor = primary
        model = secondary
        inputs = processor(
            text=[model_input],
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
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")

    embedding = primary.encode(
        [model_input],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embedding.astype("float32")


def make_embedding_logreg_predictor(model_dir: Path, device: torch.device):
    config = load_model_config(model_dir)
    metadata = config.get("embedding_metadata", {})
    classifier = joblib.load(model_dir / "model.joblib")
    encoder_bundle = load_text_encoder(metadata, device)

    def predict(text: str) -> tuple[str, float]:
        features = encode_text(text, metadata, encoder_bundle, device)
        probabilities = classifier.predict_proba(features)[0]
        classes = list(classifier.classes_)
        if "motion_query" in classes:
            class_index = classes.index("motion_query")
        else:
            class_index = classes.index(1)
        p_motion = float(probabilities[class_index])
        label = "motion_query" if p_motion >= 0.5 else "chat"
        return label, p_motion

    return predict


def make_embedding_mlp_predictor(model_dir: Path, device: torch.device):
    config = load_model_config(model_dir)
    metadata = config.get("embedding_metadata", {})
    encoder_bundle = load_text_encoder(metadata, device)
    classifier = MLPHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    classifier.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=device)
    )
    classifier.eval()

    def predict(text: str) -> tuple[str, float]:
        features = encode_text(text, metadata, encoder_bundle, device)
        inputs = torch.from_numpy(features).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(classifier(inputs), dim=1)[0]
        p_motion = float(probabilities[1].item())
        label = "motion_query" if p_motion >= 0.5 else "chat"
        return label, p_motion

    return predict


def make_embedding_predictor(model_dir: Path, device: torch.device) -> Callable[[str], tuple[str, float]]:
    config = load_model_config(model_dir)
    head = config.get("head", "mlp")
    if head == "logreg":
        return make_embedding_logreg_predictor(model_dir, device)
    if head == "mlp":
        return make_embedding_mlp_predictor(model_dir, device)
    raise ValueError(f"Unsupported head in {model_dir}/model_config.json: {head}")


def make_tfidf_logreg_predictor(model_path: Path):
    model = joblib.load(model_path)
    class_index = list(model.classes_).index("motion_query")

    def predict(text: str) -> tuple[str, float]:
        label = str(model.predict([text])[0])
        p_motion = float(model.predict_proba([text])[0][class_index])
        return label, p_motion

    return predict


def make_tfidf_mlp_predictor(model_dir: Path, device: torch.device):
    config = load_model_config(model_dir)
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    classifier = MLPHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    classifier.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=device)
    )
    classifier.eval()

    def predict(text: str) -> tuple[str, float]:
        features = vectorizer.transform([text]).toarray().astype("float32")
        inputs = torch.from_numpy(features).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(classifier(inputs), dim=1)[0]
        p_motion = float(probabilities[1].item())
        label = "motion_query" if p_motion >= 0.5 else "chat"
        return label, p_motion

    return predict
