"""Build version2 prefix datasets with a third `wait` label.

The model input is `prefix_utterance`. `masked_utterance` is included only for
inspection, not as the default training text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "version2_prefix" / "data"
SOURCE_LABELS = {"chat", "motion_query"}
TARGET_LABELS = {"chat", "motion_query", "wait"}

MOTION_PATTERNS = [
    r"\b(point|wave|dance|turn|follow|bring|demonstrat|gesture|spin|imitat)\w*\b",
    r"\b(stop|stay|hold|freeze|halt)\b",
    r"\b(walk|run|step|move|approach|retreat|back away|come here)\b",
    r"\b(pick up|put down|grab|lift|lower|raise|push|pull|place|hand me)\b",
    r"\b(show me|act out|sign language|answer with a gesture)\b",
    r"\b(look at|gaze|orient|face toward|nod|shake your head)\b",
]

TEXT_PATTERNS = [
    r"\b(what|why|how|when|where|who|which)\b",
    r"\b(explain|tell me|describe|define|summarize|list|answer|say|speak)\b",
    r"\b(mean|meaning|information|history|reason|opinion)\b",
    r"\b(without moving|do not move|don't move|no movement|words only)\b",
]

MOTION_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in MOTION_PATTERNS]
TEXT_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in TEXT_PATTERNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build version2 prefix datasets.")
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Target prefix ratios to generate for each utterance.",
    )
    parser.add_argument(
        "--chat-decision-ratio",
        type=float,
        default=0.8,
        help="Chat prefixes at or above this visible ratio are labeled chat if no motion cue appears.",
    )
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--val-file", default="val.csv")
    parser.add_argument("--test-file", default="test.csv")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source data file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"id", "utterance", "label"}
    if not rows:
        raise ValueError(f"{path} has no rows.")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for line_number, row in enumerate(rows, start=2):
        if row["label"] not in SOURCE_LABELS:
            raise ValueError(f"{path}:{line_number} has invalid label {row['label']!r}")
    return rows


def has_match(text: str, regexes: list[re.Pattern[str]]) -> bool:
    return any(regex.search(text) for regex in regexes)


def tokenize(text: str) -> list[str]:
    return text.strip().split()


def label_prefix(
    *,
    full_label: str,
    prefix_text: str,
    visible_ratio: float,
    is_complete: bool,
    chat_decision_ratio: float,
) -> tuple[str, str]:
    if is_complete:
        return full_label, "complete_input"

    has_motion = has_match(prefix_text, MOTION_REGEXES)
    has_text = has_match(prefix_text, TEXT_REGEXES)

    if full_label == "motion_query":
        if has_motion:
            return "motion_query", "motion_cue_seen"
        return "wait", "motion_cue_not_seen"

    if has_motion:
        return "wait", "potential_motion_ambiguity"
    if has_text:
        return "chat", "text_cue_seen"
    if visible_ratio >= chat_decision_ratio:
        return "chat", "high_ratio_no_motion_cue"
    return "wait", "chat_prefix_too_ambiguous"


def build_rows(
    rows: list[dict[str, str]],
    *,
    split_name: str,
    ratios: list[float],
    chat_decision_ratio: float,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in rows:
        tokens = tokenize(row["utterance"])
        if not tokens:
            continue

        total_tokens = len(tokens)
        for ratio in ratios:
            prefix_tokens = max(1, min(total_tokens, math.ceil(total_tokens * ratio)))
            prefix_text = " ".join(tokens[:prefix_tokens])
            hidden_count = total_tokens - prefix_tokens
            masked_text = " ".join(tokens[:prefix_tokens] + ["<MASK>"] * hidden_count)
            visible_ratio = prefix_tokens / total_tokens
            is_complete = prefix_tokens == total_tokens
            target_label, label_source = label_prefix(
                full_label=row["label"],
                prefix_text=prefix_text,
                visible_ratio=visible_ratio,
                is_complete=is_complete,
                chat_decision_ratio=chat_decision_ratio,
            )
            output_rows.append(
                {
                    "id": f"{row['id']}__p{int(round(ratio * 100)):03d}",
                    "original_id": row["id"],
                    "split": split_name,
                    "target_prefix_ratio": f"{ratio:.2f}",
                    "visible_ratio": f"{visible_ratio:.6f}",
                    "prefix_tokens": prefix_tokens,
                    "total_tokens": total_tokens,
                    "hidden_tokens": hidden_count,
                    "prefix_utterance": prefix_text,
                    "masked_utterance": masked_text,
                    "full_utterance": row["utterance"],
                    "full_label": row["label"],
                    "label": target_label,
                    "label_source": label_source,
                }
            )
    return output_rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "original_id",
        "split",
        "target_prefix_ratio",
        "visible_ratio",
        "prefix_tokens",
        "total_tokens",
        "hidden_tokens",
        "prefix_utterance",
        "masked_utterance",
        "full_utterance",
        "full_label",
        "label",
        "label_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["label"]) for row in rows)
    return {label: counts.get(label, 0) for label in sorted(TARGET_LABELS)}


def ratio_counts(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for row in rows:
        ratio = str(row["target_prefix_ratio"])
        grouped.setdefault(ratio, Counter()).update([str(row["label"])])
    return {
        ratio: {label: counter.get(label, 0) for label in sorted(TARGET_LABELS)}
        for ratio, counter in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    ratios = sorted(set(args.ratios))
    if not ratios or ratios[-1] != 1.0:
        raise ValueError("--ratios must include 1.0 so complete utterances are present.")
    if any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise ValueError("--ratios must be in the interval (0, 1].")

    source_data_dir = args.source_data_dir.resolve()
    output_dir = args.output_dir.resolve()

    source_files = {
        "train": args.train_file,
        "val": args.val_file,
        "test": args.test_file,
    }
    manifest: dict[str, object] = {
        "version": "version2_prefix",
        "task": "three-class early classification",
        "input_column": "prefix_utterance",
        "inspection_column": "masked_utterance",
        "labels": sorted(TARGET_LABELS),
        "source_data_dir": str(source_data_dir),
        "ratios": ratios,
        "chat_decision_ratio": args.chat_decision_ratio,
        "splits": {},
        "labeling_policy": {
            "complete_input": "Use the original chat/motion_query label at 100% prefix.",
            "motion_query_prefix": "Use motion_query only after an explicit motion cue appears; otherwise wait.",
            "chat_prefix": "Use chat after text cues, or at high visible ratio without motion cues; otherwise wait.",
        },
    }

    for split_name, filename in source_files.items():
        source_rows = read_rows(source_data_dir / filename)
        prefix_rows = build_rows(
            source_rows,
            split_name=split_name,
            ratios=ratios,
            chat_decision_ratio=args.chat_decision_ratio,
        )
        write_rows(output_dir / f"prefix_{split_name}.csv", prefix_rows)
        manifest["splits"][split_name] = {
            "source_rows": len(source_rows),
            "prefix_rows": len(prefix_rows),
            "label_counts": label_counts(prefix_rows),
            "ratio_label_counts": ratio_counts(prefix_rows),
        }
        print(
            f"{split_name}: source={len(source_rows)} "
            f"prefix={len(prefix_rows)} labels={label_counts(prefix_rows)}"
        )

    (output_dir / "prefix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
