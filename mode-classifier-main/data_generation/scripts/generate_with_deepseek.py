"""Generate classifier data with the DeepSeek chat completions API.

The script uses reference examples as few-shot guidance, but the generated
training export contains only: id, utterance, label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
REFERENCE_CSV = DATA_DIR / "reference" / "manual_seed_500.csv"
RAW_DIR = DATA_DIR / "raw"
SYSTEM_PROMPT_PATH = PROMPT_DIR / "response_mode_generation_system.md"
USER_PROMPT_PATH = PROMPT_DIR / "response_mode_generation_user.md"

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_ENV_NAMES = (
    "DEEPSEEK-mode-classifier-apikey",
    "DEEPSEEK_MODE_CLASSIFIER_APIKEY",
    "DEEPSEEK_API_KEY",
)
LABELS = {"chat", "motion_query"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-robot response-mode data with DeepSeek."
    )
    parser.add_argument("--total", type=int, default=500, help="Total examples to export.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Examples requested per API call. Use an even number for balanced batches.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat completions URL.")
    parser.add_argument(
        "--env-name",
        default=None,
        help="Environment variable containing the API key. Defaults to known names.",
    )
    parser.add_argument(
        "--output-stem",
        default="deepseek_generated_500",
        help="Output filename stem under data/raw.",
    )
    parser.add_argument(
        "--id-prefix",
        default="deepseek",
        help="Prefix for generated row ids.",
    )
    parser.add_argument(
        "--exclude-path",
        type=Path,
        action="append",
        default=[],
        help="Existing CSV/JSONL dataset to avoid duplicating. Can be repeated.",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=24,
        help="Reference examples sampled into each generation prompt.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--sleep-seconds", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def get_api_key(env_name: str | None) -> str:
    names = (env_name,) if env_name else DEFAULT_ENV_NAMES
    for name in names:
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value
        value = get_windows_environment_value(name)
        if value:
            return value
    expected = ", ".join(name for name in names if name)
    raise RuntimeError(f"DeepSeek API key not found. Expected one of: {expected}")


def get_windows_environment_value(name: str) -> str | None:
    """Read persisted Windows environment variables when not inherited."""

    if os.name != "nt":
        return None

    try:
        import winreg
    except ImportError:
        return None

    registry_locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )

    for root, subkey in registry_locations:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _value_type = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return str(value)
    return None


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_reference_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    examples: list[dict[str, str]] = []
    for row in rows:
        utterance = (row.get("utterance") or "").strip()
        label = (row.get("label") or "").strip()
        if utterance and label in LABELS:
            examples.append({"utterance": utterance, "label": label})
    if not examples:
        raise RuntimeError(f"No usable reference examples found in {path}")
    return examples


def read_dataset(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported dataset extension for exclude path: {path}")


def normalized_utterances(rows: list[dict[str, Any]]) -> set[str]:
    normalized: set[str] = set()
    for row in rows:
        utterance = str(row.get("utterance", "")).strip()
        if utterance:
            normalized.add(" ".join(utterance.lower().split()))
    return normalized


def format_reference_examples(
    examples: list[dict[str, str]], count: int, rng: random.Random
) -> str:
    sample = rng.sample(examples, k=min(count, len(examples)))
    return "\n".join(
        f'- "{example["utterance"]}" -> {example["label"]}' for example in sample
    )


def build_user_prompt(
    template: str,
    reference_examples: list[dict[str, str]],
    chat_count: int,
    motion_count: int,
    reference_count: int,
    rng: random.Random,
) -> str:
    batch_size = chat_count + motion_count
    return template.format(
        batch_size=batch_size,
        chat_count=chat_count,
        motion_count=motion_count,
        reference_examples=format_reference_examples(reference_examples, reference_count, rng),
    )


def post_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def parse_examples(content: str) -> list[dict[str, str]]:
    data = json.loads(content)
    raw_examples = data.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("Response JSON must contain an examples list.")

    examples: list[dict[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        utterance = str(item.get("utterance", "")).strip()
        label = str(item.get("label", "")).strip()
        if utterance and label in LABELS:
            examples.append({"utterance": utterance, "label": label})
    return examples


def is_valid_utterance(utterance: str) -> bool:
    if len(utterance) < 8 or len(utterance) > 180:
        return False
    if any(ord(char) > 127 for char in utterance):
        return False
    return True


def add_unique_examples(
    output: list[dict[str, str]],
    candidates: list[dict[str, str]],
    seen: set[str],
    target_counts: Counter[str],
) -> int:
    added = 0
    current_counts = Counter(example["label"] for example in output)
    for candidate in candidates:
        utterance = candidate["utterance"].strip()
        label = candidate["label"]
        normalized = " ".join(utterance.lower().split())
        if normalized in seen:
            continue
        if not is_valid_utterance(utterance):
            continue
        if current_counts[label] >= target_counts[label]:
            continue
        output.append({"utterance": utterance, "label": label})
        seen.add(normalized)
        current_counts[label] += 1
        added += 1
    return added


def write_outputs(examples: list[dict[str, str]], output_stem: str) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RAW_DIR / f"{output_stem}.jsonl"
    csv_path = RAW_DIR / f"{output_stem}.csv"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=True) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "utterance", "label"])
        writer.writeheader()
        writer.writerows(examples)

    return jsonl_path, csv_path


def main() -> None:
    args = parse_args()
    if args.total <= 0:
        raise ValueError("--total must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    rng = random.Random(args.seed)
    api_key = get_api_key(args.env_name)
    system_prompt = read_prompt(SYSTEM_PROMPT_PATH)
    user_template = read_prompt(USER_PROMPT_PATH)
    reference_examples = read_reference_examples(REFERENCE_CSV)

    target_counts = Counter(
        {
            "chat": args.total // 2,
            "motion_query": args.total - (args.total // 2),
        }
    )
    generated: list[dict[str, str]] = []
    seen = normalized_utterances(reference_examples)
    for exclude_path in args.exclude_path:
        excluded_rows = read_dataset(exclude_path)
        seen.update(normalized_utterances(excluded_rows))
        print(f"Loaded {len(excluded_rows)} exclusion rows from {exclude_path}")

    attempt = 0
    while sum(Counter(example["label"] for example in generated).values()) < args.total:
        current_counts = Counter(example["label"] for example in generated)
        remaining_chat = target_counts["chat"] - current_counts["chat"]
        remaining_motion = target_counts["motion_query"] - current_counts["motion_query"]
        remaining = remaining_chat + remaining_motion
        batch_size = min(args.batch_size, remaining)
        if batch_size <= 0:
            break

        chat_count = min(remaining_chat, batch_size // 2)
        motion_count = min(remaining_motion, batch_size - chat_count)
        unused_slots = batch_size - chat_count - motion_count
        if unused_slots and remaining_chat > chat_count:
            extra_chat = min(unused_slots, remaining_chat - chat_count)
            chat_count += extra_chat
            unused_slots -= extra_chat
        if unused_slots and remaining_motion > motion_count:
            motion_count += min(unused_slots, remaining_motion - motion_count)

        attempt += 1
        prompt = build_user_prompt(
            user_template,
            reference_examples,
            chat_count,
            motion_count,
            args.reference_count,
            rng,
        )

        last_error: Exception | None = None
        for retry in range(1, args.max_retries + 1):
            try:
                content = post_chat_completion(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                candidates = parse_examples(content)
                added = add_unique_examples(
                    generated, candidates, seen, target_counts
                )
                counts = Counter(example["label"] for example in generated)
                print(
                    f"Batch {attempt}, retry {retry}: added {added}; "
                    f"counts={dict(counts)}"
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, json.JSONDecodeError) as error:
                last_error = error
                print(f"Batch {attempt}, retry {retry} failed: {error}")
                time.sleep(args.sleep_seconds * retry)
        else:
            raise RuntimeError(f"Failed to generate batch {attempt}") from last_error

        time.sleep(args.sleep_seconds)

    final_counts = Counter(example["label"] for example in generated)
    if final_counts != target_counts:
        raise RuntimeError(f"Could not satisfy target counts: {dict(final_counts)}")

    rng.shuffle(generated)
    for index, example in enumerate(generated, start=1):
        example["id"] = f"{args.id_prefix}-{index:05d}"

    jsonl_path, csv_path = write_outputs(generated, args.output_stem)
    print(f"Wrote {len(generated)} examples to {jsonl_path}")
    print(f"Wrote {len(generated)} examples to {csv_path}")
    print(f"Label distribution: {dict(final_counts)}")


if __name__ == "__main__":
    main()
