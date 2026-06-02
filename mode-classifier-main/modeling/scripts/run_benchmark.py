"""Run full 4×2 benchmark: train, evaluate, measure latency, write report."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "modeling" / "scripts"
REPORTS = ROOT / "modeling" / "reports"
SPLITS = ROOT / "modeling" / "data" / "splits"
HARD_TEST = ROOT / "modeling" / "data" / "hard_test.csv"

TIMINGS_PATH = REPORTS / "benchmark_timings.json"

EMBEDDING_CACHES = {
    "clip": {
        "path": ROOT / "modeling/data/embeddings/clip_deepseek_10000.npz",
        "cmd": [
            sys.executable,
            str(SCRIPTS / "cache_text_embeddings.py"),
            "--encoder-backend",
            "clip",
            "--model-name",
            "openai/clip-vit-base-patch32",
            "--no-text-prefix",
            "--output",
            "modeling/data/embeddings/clip_deepseek_10000.npz",
        ],
    },
    "qwen": {
        "path": ROOT / "modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz",
        "cmd": [
            sys.executable,
            str(SCRIPTS / "cache_text_embeddings.py"),
            "--model-name",
            "Qwen/Qwen3-Embedding-0.6B",
            "--output",
            "modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz",
        ],
    },
    "minilm": {
        "path": ROOT / "modeling/data/embeddings/minilm_deepseek_10000.npz",
        "cmd": [
            sys.executable,
            str(SCRIPTS / "cache_text_embeddings.py"),
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--no-text-prefix",
            "--output",
            "modeling/data/embeddings/minilm_deepseek_10000.npz",
        ],
    },
}

MODELS = [
    {
        "name": "tfidf+logreg",
        "encoder": "tfidf",
        "head": "logreg",
        "artifact_dir": ROOT / "modeling/artifacts/tfidf_logreg",
        "train_cmd": [sys.executable, str(SCRIPTS / "train_tfidf_logreg.py")],
    },
    {
        "name": "tfidf+mlp",
        "encoder": "tfidf",
        "head": "mlp",
        "artifact_dir": ROOT / "modeling/artifacts/tfidf_mlp",
        "train_cmd": [sys.executable, str(SCRIPTS / "train_tfidf_mlp.py")],
    },
    {
        "name": "clip+logreg",
        "encoder": "clip",
        "head": "logreg",
        "artifact_dir": ROOT / "modeling/artifacts/clip_logreg",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_logreg.py"),
            "--embedding-cache",
            "modeling/data/embeddings/clip_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/clip_logreg",
        ],
    },
    {
        "name": "clip+mlp",
        "encoder": "clip",
        "head": "mlp",
        "artifact_dir": ROOT / "modeling/artifacts/clip_mlp",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_mlp.py"),
            "--embedding-cache",
            "modeling/data/embeddings/clip_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/clip_mlp",
        ],
    },
    {
        "name": "qwen+logreg",
        "encoder": "qwen",
        "head": "logreg",
        "artifact_dir": ROOT / "modeling/artifacts/qwen_logreg",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_logreg.py"),
            "--embedding-cache",
            "modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/qwen_logreg",
        ],
    },
    {
        "name": "qwen+mlp",
        "encoder": "qwen",
        "head": "mlp",
        "artifact_dir": ROOT / "modeling/artifacts/qwen_mlp",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_mlp.py"),
            "--embedding-cache",
            "modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/qwen_mlp",
        ],
    },
    {
        "name": "minilm+logreg",
        "encoder": "minilm",
        "head": "logreg",
        "artifact_dir": ROOT / "modeling/artifacts/minilm_logreg",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_logreg.py"),
            "--embedding-cache",
            "modeling/data/embeddings/minilm_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/minilm_logreg",
        ],
    },
    {
        "name": "minilm+mlp",
        "encoder": "minilm",
        "head": "mlp",
        "artifact_dir": ROOT / "modeling/artifacts/minilm_mlp",
        "train_cmd": [
            sys.executable,
            str(SCRIPTS / "train_embedding_mlp.py"),
            "--embedding-cache",
            "modeling/data/embeddings/minilm_deepseek_10000.npz",
            "--output-dir",
            "modeling/artifacts/minilm_mlp",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark and write report.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-cache", action="store_true", help="Reuse existing .npz caches.")
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "benchmark_report.md",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def run_step(label: str, cmd: list[str]) -> float:
    print(f"\n{'=' * 60}\n==> {label}\n{'=' * 60}", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    print(f"[{label}] finished in {elapsed:.1f}s", flush=True)
    return elapsed


def load_test_utterances(limit: int) -> list[str]:
    from shared_utils import read_split

    rows = read_split(SPLITS / "test.csv")
    texts = [row["utterance"] for row in rows]
    if limit >= len(texts):
        return texts
    return texts[:limit]


def benchmark_latency(
    name: str,
    artifact_dir: Path,
    texts: list[str],
    device: torch.device,
    warmup: int,
) -> dict[str, float]:
    sys.path.insert(0, str(SCRIPTS))
    from embedding_utils import (
        make_embedding_predictor,
        make_tfidf_logreg_predictor,
        make_tfidf_mlp_predictor,
    )

    if name == "tfidf+logreg":
        predict = make_tfidf_logreg_predictor(artifact_dir / "model.joblib")
    elif name == "tfidf+mlp":
        predict = make_tfidf_mlp_predictor(artifact_dir, device)
    else:
        predict = make_embedding_predictor(artifact_dir, device)

    for _ in range(warmup):
        predict(texts[0])

    latencies_ms: list[float] = []
    for text in texts:
        start = time.perf_counter()
        predict(text)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    p50_index = max(int(len(latencies_ms) * 0.50) - 1, 0)
    p95_index = max(int(len(latencies_ms) * 0.95) - 1, 0)
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": latencies_ms[p50_index],
        "p95_ms": latencies_ms[p95_index],
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "num_samples": len(latencies_ms),
    }


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.0f}s"


def render_report(payload: dict) -> str:
    meta = payload["meta"]
    rows = payload["models"]
    lines = [
        "# Mode Classifier Benchmark Report",
        "",
        f"- **Generated**: {meta['generated_at']}",
        f"- **Dataset**: {meta['dataset']} ({meta['split']})",
        f"- **Device**: {meta['device']}",
        f"- **Task**: binary classification (`chat` vs `motion_query`)",
        "",
        "## Summary",
        "",
        "Comparison of **4 text encoders** × **2 classifier heads**.",
        "Latency = end-to-end single-utterance inference (encode + classify).",
        "",
        "| Encoder | Head | Test Acc | Hard Acc | Train Time | Latency (mean) | Latency (p95) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        if row.get("status") != "ok":
            continue
        acc = row["accuracy"]
        latency = row["latency"]
        timing = row["timing"]
        lines.append(
            f"| {row['encoder']} | {row['head']} | "
            f"{acc.get('test', float('nan')):.4f} | "
            f"{acc.get('hard', float('nan')):.4f} | "
            f"{format_seconds(timing['total_train_s'])} | "
            f"{latency['mean_ms']:.1f} ms | "
            f"{latency['p95_ms']:.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Accuracy (4 × 2 matrix, test split)",
            "",
            "| Encoder | LogReg | MLP |",
            "| --- | ---: | ---: |",
        ]
    )
    for encoder in ("tfidf", "clip", "qwen", "minilm"):
        logreg = next(
            (r for r in rows if r["encoder"] == encoder and r["head"] == "logreg" and r.get("status") == "ok"),
            None,
        )
        mlp = next(
            (r for r in rows if r["encoder"] == encoder and r["head"] == "mlp" and r.get("status") == "ok"),
            None,
        )
        logreg_cell = f"{logreg['accuracy']['test']:.4f}" if logreg else "—"
        mlp_cell = f"{mlp['accuracy']['test']:.4f}" if mlp else "—"
        lines.append(f"| {encoder} | {logreg_cell} | {mlp_cell} |")

    lines.extend(["", "## Training Time", ""])
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"- **{row['name']}**: skipped ({row.get('error', 'unknown')})")
            continue
        timing = row["timing"]
        cache_part = (
            f"cache {format_seconds(timing['cache_s'])} + "
            if timing.get("cache_s", 0) > 0
            else ""
        )
        lines.append(
            f"- **{row['name']}**: {cache_part}head {format_seconds(timing['head_train_s'])} "
            f"= **{format_seconds(timing['total_train_s'])}**"
        )

    lines.extend(
        [
            "",
            "## Inference Latency (single utterance)",
            "",
            f"Measured on {meta['latency_samples']} test utterances "
            f"({meta['warmup']} warmup runs). Unit: milliseconds.",
            "",
            "| Model | Mean | p50 | p95 | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row.get("status") != "ok":
            continue
        lat = row["latency"]
        lines.append(
            f"| {row['name']} | {lat['mean_ms']:.1f} | {lat['p50_ms']:.1f} | "
            f"{lat['p95_ms']:.1f} | {lat['min_ms']:.1f} | {lat['max_ms']:.1f} |"
        )

    lines.extend(["", "## Hard-set Accuracy (boundary cases)", ""])
    lines.append("| Model | Accuracy | Macro F1 | False Motion | False Text |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        if row.get("status") != "ok":
            continue
        hard = row["accuracy"].get("hard_details", {})
        if not hard:
            continue
        lines.append(
            f"| {row['name']} | {hard.get('accuracy', 0):.4f} | "
            f"{hard.get('macro_f1', 0):.4f} | "
            f"{hard.get('false_motion_rate', 0):.4f} | "
            f"{hard.get('false_text_rate', 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Train time** for embedding models includes one-time encoder caching on the full 10k dataset, "
            "amortized across both LogReg and MLP heads of the same encoder.",
            "- **Latency** includes model loading is excluded; numbers reflect steady-state per-query inference.",
            "- TF-IDF models are CPU-friendly; embedding models ran on the configured device above.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(SCRIPTS))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_times: dict[str, float] = {}
    head_train_times: dict[str, float] = {}
    encoder_cache_assigned: dict[str, bool] = {}

    if not args.skip_train:
        for encoder, spec in EMBEDDING_CACHES.items():
            if args.skip_cache and spec["path"].exists():
                print(f"Skipping cache for {encoder} (exists): {spec['path']}", flush=True)
                cache_times[encoder] = 0.0
                continue
            cache_times[encoder] = run_step(f"Cache {encoder} embeddings", spec["cmd"])

        for model in MODELS:
            head_train_times[model["name"]] = run_step(
                f"Train {model['name']}", model["train_cmd"]
            )

        TIMINGS_PATH.write_text(
            json.dumps(
                {"cache_times": cache_times, "head_train_times": head_train_times},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    elif TIMINGS_PATH.exists():
        timing_data = json.loads(TIMINGS_PATH.read_text(encoding="utf-8"))
        cache_times = timing_data.get("cache_times", {})
        head_train_times = timing_data.get("head_train_times", {})

    run_step(
        "Evaluate all models",
        [sys.executable, str(SCRIPTS / "eval_comparison.py")],
    )

    comparison_path = REPORTS / "comparison_summary.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    acc_by_name = {entry["model"]: entry for entry in comparison["models"]}

    texts = load_test_utterances(args.latency_samples)
    results: list[dict] = []

    for model in MODELS:
        name = model["name"]
        entry: dict = {
            "name": name,
            "encoder": model["encoder"],
            "head": model["head"],
        }
        artifact_dir = model["artifact_dir"]
        if not artifact_dir.exists() or (
            name == "tfidf+logreg" and not (artifact_dir / "model.joblib").exists()
        ):
            entry["status"] = "missing"
            entry["error"] = f"artifact not found: {artifact_dir}"
            results.append(entry)
            continue

        acc_entry = acc_by_name.get(name, {})
        entry["accuracy"] = {
            "test": acc_entry.get("test", {}).get("accuracy"),
            "val": acc_entry.get("val", {}).get("accuracy"),
            "hard": acc_entry.get("hard", {}).get("accuracy"),
            "hard_details": acc_entry.get("hard", {}),
        }

        encoder = model["encoder"]
        cache_s = 0.0
        if encoder in EMBEDDING_CACHES and encoder not in encoder_cache_assigned:
            cache_s = cache_times.get(encoder, 0.0)
            encoder_cache_assigned[encoder] = True
            # Split cache cost across the two heads of this encoder.
            cache_s /= 2.0

        head_s = head_train_times.get(name, 0.0)
        entry["timing"] = {
            "cache_s": cache_s,
            "head_train_s": head_s,
            "total_train_s": cache_s + head_s,
        }

        try:
            entry["latency"] = benchmark_latency(
                name, artifact_dir, texts, device, args.warmup
            )
            entry["status"] = "ok"
        except Exception as error:
            entry["status"] = "error"
            entry["error"] = str(error)

        results.append(entry)
        print(
            f"[latency] {name}: mean={entry.get('latency', {}).get('mean_ms', 'n/a')} ms",
            flush=True,
        )

    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "dataset": manifest.get("input", "deepseek_generated_10000.csv"),
            "split": "80/10/10 stratified",
            "device": str(device),
            "latency_samples": len(texts),
            "warmup": args.warmup,
        },
        "models": results,
    }

    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md = render_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_md, encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
