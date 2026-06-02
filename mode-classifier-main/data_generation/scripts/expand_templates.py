"""Expand the dataset with rule-based template utterances (no API required)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_CSV = ROOT / "data" / "reference" / "manual_seed_500.csv"
LABELS = {"chat", "motion_query"}

CHAT_TEMPLATES = [
    "What is {topic}?",
    "How does {topic} work?",
    "Can you explain {topic} in simple terms?",
    "Why is {topic} important?",
    "Tell me about {topic}.",
    "What are the benefits of {topic}?",
    "How can I learn {topic}?",
    "Give me advice about {topic}.",
    "What is the difference between {a} and {b}?",
    "Who invented {thing}?",
    "When was {thing} discovered?",
    "Where can I find information about {topic}?",
    "Summarize the history of {topic}.",
    "What causes {problem}?",
    "How do I fix {problem}?",
    "Recommend resources for {topic}.",
    "What are common mistakes in {topic}?",
    "Compare {a} and {b} for beginners.",
    "I need help with {topic} for my homework.",
    "Describe the main ideas behind {topic}.",
    "Explain {topic} {modifier}.",
    "What should I know about {topic} {modifier}?",
    "Help me understand {topic} {modifier}.",
]

MOTION_TEMPLATES = [
    "Point to the {object}.",
    "Please point at the {object}.",
    "Wave your {body_part}.",
    "Can you wave at me?",
    "Turn toward the {object}.",
    "Turn to face the {object}.",
    "Look at the {object}.",
    "Follow me to the {place}.",
    "Walk to the {place}.",
    "Move to the {place}.",
    "Bring me the {object}.",
    "Pick up the {object}.",
    "Put the {object} on the table.",
    "Place the {object} near the door.",
    "Hand me the {object}.",
    "Demonstrate how to {action}.",
    "Show me how to {action}.",
    "Can you do a short {motion}?",
    "Perform a {motion} for me.",
    "Stop moving and hold your position.",
    "Stay where you are.",
    "Back away from the {object}.",
    "Step closer to the {object}.",
    "Lower your {body_part} slowly.",
    "Raise your {body_part}.",
    "Nod to confirm you understand.",
    "Shake your head if you disagree.",
    "Act out {emotion}.",
    "Make a gesture that means {signal}.",
    "Guide me through the {place}.",
    "Lead me to the {place}.",
    "Rotate toward the audience.",
    "Grab the {object} with your right hand.",
    "{polite} point to the {object}.",
    "{polite} move the {object} to the {place}.",
    "{polite} show me the {object} in the {place}.",
]

CHAT_SLOTS = {
    "topic": [
        "machine learning",
        "photosynthesis",
        "blockchain",
        "public speaking",
        "time management",
        "nutrition",
        "sleep hygiene",
        "linear algebra",
        "climate change",
        "renewable energy",
        "meditation",
        "personal finance",
        "resume writing",
        "stress management",
        "Python programming",
        "database design",
        "classical music",
        "world history",
        "calculus",
        "probability",
        "organic chemistry",
        "microeconomics",
        "astrophysics",
        "linguistics",
        "robotics ethics",
        "computer vision",
        "natural language processing",
        "supply chains",
        "project management",
        "critical thinking",
        "digital privacy",
        "vaccines",
        "epidemiology",
        "urban planning",
        "game theory",
        "cryptography",
        "operating systems",
        "compilers",
        "human anatomy",
        "evolution",
        "thermodynamics",
    ],
    "problem": [
        "insomnia",
        "procrastination",
        "network latency",
        "overfitting",
        "back pain",
        "stage fright",
        "memory leaks",
        "gradient explosion",
        "writer's block",
        "decision fatigue",
        "eye strain",
        "budgeting mistakes",
    ],
    "thing": [
        "the telephone",
        "the internet",
        "CRISPR",
        "the printing press",
        "GPS",
        "the microscope",
        "the steam engine",
        "the transistor",
        "the bicycle",
        "the airplane",
        "the camera",
        "the refrigerator",
    ],
    "a": [
        "TCP",
        "UDP",
        "SQL",
        "NoSQL",
        "supervised learning",
        "reinforcement learning",
        "gradient descent",
        "k-means",
        "decision trees",
        "random forests",
    ],
    "b": [
        "HTTP",
        "FTP",
        "Python",
        "R",
        "unsupervised learning",
        "imitation learning",
        "momentum",
        "Adam",
        "logistic regression",
        "SVM",
    ],
    "modifier": [
        "",
        "briefly",
        "in detail",
        "for a beginner",
        "step by step",
        "with examples",
        "in one paragraph",
    ],
}

MOTION_SLOTS = {
    "object": [
        "red block",
        "blue cup",
        "door",
        "window",
        "chair",
        "laptop",
        "book",
        "plant",
        "exit sign",
        "toolbox",
        "marker",
        "whiteboard",
        "trash bin",
        "power switch",
        "charging dock",
        "first aid kit",
        "fire extinguisher",
        "elevator button",
        "shelf",
        "drawer",
    ],
    "body_part": [
        "arm",
        "hand",
        "head",
        "right arm",
        "left hand",
        "both hands",
        "wrist",
    ],
    "place": [
        "lab",
        "kitchen",
        "doorway",
        "charging station",
        "workbench",
        "hallway",
        "classroom",
        "lobby",
        "parking area",
        "conference room",
    ],
    "action": [
        "tie a knot",
        "open a jar",
        "fold a towel",
        "press the button",
        "scan the QR code",
        "stack the boxes",
        "sort the tools",
        "wipe the table",
    ],
    "motion": [
        "dance",
        "bow",
        "spin",
        "salute",
        "stretch",
        "wave",
        "clap",
        "kneel",
    ],
    "emotion": ["happiness", "surprise", "confusion", "excitement", "tiredness", "curiosity"],
    "signal": ["hello", "stop", "come here", "thank you", "yes", "no", "wait", "help"],
    "polite": ["", "Please", "Could you", "Can you"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate template-based utterances.")
    parser.add_argument("--count", type=int, required=True, help="How many new rows to add.")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="CSV/JSONL files whose utterances must not be duplicated.",
    )
    parser.add_argument("--output-stem", default="template_generated")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_seen(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_rows(path):
            utterance = row.get("utterance", "").strip()
            if utterance:
                seen.add(normalize(utterance))
    return seen


def fill_template(template: str, rng: random.Random) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        pool = CHAT_SLOTS.get(key) or MOTION_SLOTS.get(key)
        if not pool:
            return key
        return rng.choice(pool)

    text = re.sub(r"\{(\w+)\}", replacer, template)
    return " ".join(text.split())


def generate_candidates(
    *,
    label: str,
    count: int,
    seen: set[str],
    rng: random.Random,
) -> list[dict[str, str]]:
    templates = CHAT_TEMPLATES if label == "chat" else MOTION_TEMPLATES
    rows: list[dict[str, str]] = []
    max_attempts = count * 200

    # Pass 1: random sampling with rich templates.
    attempts = 0
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        template = rng.choice(templates)
        utterance = fill_template(template, rng).strip()
        if utterance and not utterance.endswith("?"):
            utterance = utterance.rstrip(".") + "."
        key = normalize(utterance)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"utterance": utterance, "label": label})

    # Pass 2: numbered variants if still short.
    suffix = 1
    while len(rows) < count:
        template = templates[suffix % len(templates)]
        utterance = fill_template(template, rng).strip()
        utterance = f"{utterance} (variant {suffix})"
        key = normalize(utterance)
        suffix += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append({"utterance": utterance, "label": label})

    return rows


def write_outputs(rows: list[dict[str, str]], stem: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"{stem}.csv"
    for index, row in enumerate(rows, start=1):
        row["id"] = f"{stem}-{index:05d}"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "utterance", "label"])
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = RAW_DIR / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")
    return csv_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    exclude_paths = list(args.exclude) + [REFERENCE_CSV]
    seen = load_seen(exclude_paths)

    half = args.count // 2
    chat_rows = generate_candidates(label="chat", count=half, seen=seen, rng=rng)
    motion_rows = generate_candidates(
        label="motion_query",
        count=args.count - half,
        seen=seen,
        rng=rng,
    )
    rows = chat_rows + motion_rows
    rng.shuffle(rows)
    csv_path = write_outputs(rows, args.output_stem)
    print(f"Wrote {len(rows)} template rows to {csv_path}")


if __name__ == "__main__":
    main()
