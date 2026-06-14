"""Build a clean, presentation-friendly view of our project files.

The original runnable code and report snapshots stay in place. This script
copies the data splits, per-combination programs, training curves, and test
reports into `our_part/` so the project contribution is easier to inspect.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUR_PART = ROOT / "our_part"

COMBOS = [
    ("tfidf", "logreg", "tfidf__logreg"),
    ("tfidf", "mlp", "tfidf__mlp"),
    ("minilm", "logreg", "minilm__logreg"),
    ("minilm", "mlp", "minilm__mlp"),
    ("clip", "logreg", "clip__logreg"),
    ("clip", "mlp", "clip__mlp"),
    ("qwen3_0_6b", "logreg", "qwen3_0_6b__logreg"),
    ("qwen3_0_6b", "mlp", "qwen3_0_6b__mlp"),
]

V2_EXTRA_REPORTS = [
    "test_threshold_sweep.csv",
    "test_selective_metrics_by_prefix_ratio_t070.csv",
    "test_first_decisions_t070.csv",
    "test_metrics_by_prefix_ratio.csv",
    "test_prefix_metrics.png",
    "val_threshold_sweep.csv",
    "val_selective_metrics_by_prefix_ratio_t070.csv",
    "val_first_decisions_t070.csv",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8", newline="\n")


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_repro_script(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"Expected text not found in {src}: {old}")
        text = text.replace(old, new)
    write_text(dst, text)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def program_launcher(version: str, encoder: str, head: str) -> str:
    if version == "v1":
        script_path = "scripts/run_v1_experiments.py"
        description = "V1 full-input binary classifier"
    elif version == "v2":
        script_path = "scripts/run_v2_prefix_experiments.py"
        description = "V2 prefix early-decision classifier"
    else:
        raise ValueError(f"Unknown version: {version}")

    return f'''"""Run the {description}: {encoder} + {head}.

This file is the one-combination entry point for the submitted sourcecode
folder. It writes reproduced outputs under sourcecode/reproduced/ and does not
overwrite the tracked result files in this model folder.

Extra command-line arguments are forwarded to the project training script.
Example:
    python program.py --device cuda --skip-existing
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "{script_path}"),
        "--encoders",
        "{encoder}",
        "--heads",
        "{head}",
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
'''


def copy_data_splits() -> None:
    v1_data = OUR_PART / "data" / "v1_full_input"
    copy_file(ROOT / "data" / "train.csv", v1_data / "train.csv")
    copy_file(ROOT / "data" / "val.csv", v1_data / "val.csv")
    copy_file(ROOT / "data" / "test.csv", v1_data / "test.csv")
    copy_file(ROOT / "data" / "split_manifest.json", v1_data / "manifest.json")

    v2_data = OUR_PART / "data" / "v2_prefix_input"
    copy_file(ROOT / "version2_prefix" / "data" / "prefix_train.csv", v2_data / "train.csv")
    copy_file(ROOT / "version2_prefix" / "data" / "prefix_val.csv", v2_data / "val.csv")
    copy_file(ROOT / "version2_prefix" / "data" / "prefix_test.csv", v2_data / "test.csv")
    copy_file(ROOT / "version2_prefix" / "data" / "prefix_manifest.json", v2_data / "manifest.json")


def copy_v1_combo(encoder: str, head: str, combo: str) -> None:
    src = ROOT / "experiments" / "results" / "latest" / "runs" / combo
    dst = OUR_PART / "V1_full_input" / combo
    write_text(dst / "program.py", program_launcher("v1", encoder, head))
    remove_file_if_exists(dst / "shared_program_source.py")
    copy_file(src / "loss_curve.png", dst / "training_loss.png")
    copy_file(src / "history.csv", dst / "training_history.csv")
    copy_file(src / "metrics.json", dst / "test_results.json")
    copy_file(src / "run_config.json", dst / "run_config.json")


def copy_v2_combo(encoder: str, head: str, combo: str) -> None:
    src = ROOT / "version2_prefix" / "results" / "latest" / "runs" / combo
    dst = OUR_PART / "V2_prefix" / combo
    write_text(dst / "program.py", program_launcher("v2", encoder, head))
    remove_file_if_exists(dst / "shared_program_source.py")
    copy_file(src / "loss_curve.png", dst / "training_loss.png")
    copy_file(src / "history.csv", dst / "training_history.csv")
    copy_file(src / "metrics.json", dst / "test_results.json")
    copy_file(src / "run_config.json", dst / "run_config.json")
    for filename in V2_EXTRA_REPORTS:
        report = src / filename
        if report.exists():
            copy_file(report, dst / filename)


def copy_summaries() -> None:
    v1_dst = OUR_PART / "V1_full_input"
    copy_file(ROOT / "experiments" / "results" / "latest" / "summary.csv", v1_dst / "summary.csv")
    copy_file(ROOT / "experiments" / "results" / "latest" / "summary.md", v1_dst / "summary.md")
    copy_file(ROOT / "experiments" / "results" / "latest" / "experiment_plan.json", v1_dst / "experiment_plan.json")

    v2_dst = OUR_PART / "V2_prefix"
    copy_file(ROOT / "version2_prefix" / "results" / "latest" / "summary.csv", v2_dst / "summary.csv")
    copy_file(ROOT / "version2_prefix" / "results" / "latest" / "summary.md", v2_dst / "summary.md")
    copy_file(ROOT / "version2_prefix" / "results" / "latest" / "experiment_plan.json", v2_dst / "experiment_plan.json")


def copy_reproduction_files() -> None:
    copy_file(ROOT / "requirements.txt", OUR_PART / "requirements.txt")
    write_text(
        OUR_PART / ".gitignore",
        "\n".join(["reproduced/", ".venv/", "__pycache__/", "*.pyc", ""]),
    )
    write_repro_script(
        ROOT / "experiments" / "run_experiments.py",
        OUR_PART / "scripts" / "run_v1_experiments.py",
        {
            "This is the top-level project implementation, intentionally separate from the\nimported `mode-classifier-main/` folder.": "This is the self-contained V1 reproduction script for the submitted\nsourcecode folder.",
            'DEFAULT_DATA_DIR = ROOT / "data"': 'DEFAULT_DATA_DIR = ROOT / "data" / "v1_full_input"',
            'DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "ours"': 'DEFAULT_OUTPUT_DIR = ROOT / "reproduced" / "artifacts" / "v1"',
            'default=ROOT / "experiments" / "results" / "latest",': 'default=ROOT / "reproduced" / "V1_full_input",',
            '"embedding_cache": "artifacts/ours/embeddings/{encoder}/",': '"embedding_cache": "reproduced/artifacts/v1/embeddings/{encoder}/",',
            '"run_output": "artifacts/ours/runs/{encoder}__{head}/",': '"run_output": "reproduced/artifacts/v1/runs/{encoder}__{head}/",',
            '"summary": "artifacts/ours/summary.csv and summary.md",': '"summary": "reproduced/artifacts/v1/summary.csv and summary.md",',
        },
    )
    write_repro_script(
        ROOT / "version2_prefix" / "scripts" / "run_prefix_experiments.py",
        OUR_PART / "scripts" / "run_v2_prefix_experiments.py",
        {
            "This implementation is isolated from both `experiments/` and\n`mode-classifier-main/`.": "This is the self-contained V2 reproduction script for the submitted\nsourcecode folder.",
            "ROOT = Path(__file__).resolve().parents[2]": "ROOT = Path(__file__).resolve().parents[1]",
            'DEFAULT_DATA_DIR = ROOT / "version2_prefix" / "data"': 'DEFAULT_DATA_DIR = ROOT / "data" / "v2_prefix_input"',
            'DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "version2_prefix"': 'DEFAULT_OUTPUT_DIR = ROOT / "reproduced" / "artifacts" / "v2"',
            'DEFAULT_REPORT_DIR = ROOT / "version2_prefix" / "results" / "latest"': 'DEFAULT_REPORT_DIR = ROOT / "reproduced" / "V2_prefix"',
            'parser.add_argument("--train-file", default="prefix_train.csv")': 'parser.add_argument("--train-file", default="train.csv")',
            'parser.add_argument("--val-file", default="prefix_val.csv")': 'parser.add_argument("--val-file", default="val.csv")',
            'parser.add_argument("--test-file", default="prefix_test.csv")': 'parser.add_argument("--test-file", default="test.csv")',
            '"embedding_cache": "artifacts/version2_prefix/embeddings/{encoder}/",': '"embedding_cache": "reproduced/artifacts/v2/embeddings/{encoder}/",',
            '"run_output": "artifacts/version2_prefix/runs/{encoder}__{head}/",': '"run_output": "reproduced/artifacts/v2/runs/{encoder}__{head}/",',
            '"report_snapshot": "version2_prefix/results/latest/",': '"report_snapshot": "reproduced/V2_prefix/",',
        },
    )
    write_repro_script(
        ROOT / "scripts" / "evaluate_hard_test.py",
        OUR_PART / "scripts" / "evaluate_hard_test.py",
        {
            "from experiments.run_experiments import": "from scripts.run_v1_experiments import",
            'default=ROOT / "our_part" / "data" / "hard_test" / "hard_test.csv",': 'default=ROOT / "data" / "hard_test" / "hard_test.csv",',
            'default=ROOT / "artifacts" / "ours",': 'default=ROOT / "reproduced" / "artifacts" / "v1",',
            'default=ROOT / "our_part" / "hard_test",': 'default=ROOT / "reproduced" / "hard_test",',
            'default=ROOT / "our_part" / "V1_full_input",': 'default=ROOT / "reproduced" / "V1_full_input",',
        },
    )
    write_repro_script(
        ROOT / "scripts" / "build_hard_test_dataset.py",
        OUR_PART / "scripts" / "build_hard_test_dataset.py",
        {
            'OUTPUT_DIR = ROOT / "our_part" / "data" / "hard_test"': 'OUTPUT_DIR = ROOT / "data" / "hard_test"',
            '"our_part/data/hard_test/provided_hard_test.csv"': '"data/hard_test/provided_hard_test.csv"',
        },
    )


def write_readme() -> None:
    v1_data = OUR_PART / "data" / "v1_full_input"
    v2_data = OUR_PART / "data" / "v2_prefix_input"
    hard_data = OUR_PART / "data" / "hard_test"
    readme = rf"""# Our Project Part

This folder is the submitted package for our part of the project. In the final
submission, this folder may be renamed to `sourcecode/`; all paths below are
relative to that submitted folder. It contains the data, reproduction scripts,
tracked outputs, and report source for our experiments.

The tracked CSV, JSON, PNG, and TeX files can be inspected directly. To rerun
the experiments, follow the setup and reproduction commands below.

## Data

- `data/v1_full_input/train.csv`, `val.csv`, `test.csv`: full-utterance binary
  classification data.
- `data/v1_full_input/manifest.json`: source and split metadata for V1.
- `data/v2_prefix_input/train.csv`, `val.csv`, `test.csv`: prefix-based data
  for early-decision classification.
- `data/v2_prefix_input/manifest.json`: prefix-generation and split metadata
  for V2.
- `data/hard_test/provided_hard_test.csv`: the 30-example hard-test extension
  provided externally.
- `data/hard_test/hard_test.csv`: {count_csv_rows(hard_data / "hard_test.csv")}-example combined hard test for
  full-utterance binary classification.
- `data/hard_test/manifest.json`: hard-test design metadata.

| Version | Train | Validation | Test |
|---------|-------|------------|------|
| V1 full input | {count_csv_rows(v1_data / "train.csv")} | {count_csv_rows(v1_data / "val.csv")} | {count_csv_rows(v1_data / "test.csv")} |
| V2 prefix input | {count_csv_rows(v2_data / "train.csv")} | {count_csv_rows(v2_data / "val.csv")} | {count_csv_rows(v2_data / "test.csv")} |

## V1_full_input

Each subfolder is one encoder/head combination, named as:

```text
{{encoder}}__{{head}}
```

Each combination folder contains:

- `program.py`: source/provenance file for the corresponding encoder/head
  combination. It can be run from this submitted folder to reproduce only that
  encoder/head combination; reproduced outputs are written under `reproduced/`.
- `training_loss.png`: training/validation loss curve.
- `training_history.csv`: epoch-by-epoch training history.
- `test_results.json`: metrics and classification report.
- `hard_test_metrics.json`, `hard_test_predictions.csv`,
  `hard_test_category_metrics.csv`: results on the combined hard test.
- `run_config.json`: exact run configuration.

## V2_prefix

The V2 folders use the same `{{encoder}}__{{head}}` naming. Each folder contains
the same core files as V1, plus prefix-specific evaluation reports:

- `test_threshold_sweep.csv`: accuracy/coverage/latency by confidence threshold.
- `test_selective_metrics_by_prefix_ratio_t070.csv`: selective accuracy and
  coverage at threshold 0.70 by prefix ratio.
- `test_first_decisions_t070.csv`: first non-wait decision for each test
  example at threshold 0.70.
- `test_metrics_by_prefix_ratio.csv` and `test_prefix_metrics.png`: diagnostic
  prefix-ratio reports.

Top-level `summary.csv` and `summary.md` files in both `V1_full_input/` and
`V2_prefix/` compare all eight combinations.

## How To Read This Submission

Start with these files:

1. `report/overleaf_report_source.tex`: report source aligned with the tracked
   results in this folder.
2. `V1_full_input/summary.csv`: full-utterance V1 comparison across all eight
   encoder/head combinations.
3. `V2_prefix/summary.csv`: V2 early-decision comparison at threshold 0.70.
4. `hard_test/summary.csv`: V1 hard-test comparison on 150 boundary examples.

For any individual model combination, open its folder:

```text
V1_full_input/{{encoder}}__{{head}}/
V2_prefix/{{encoder}}__{{head}}/
```

The most useful files inside each model folder are:

- `training_loss.png`: loss curve for convergence checking.
- `training_history.csv`: epoch-level train/validation loss and accuracy.
- `test_results.json`: final test metrics.
- `run_config.json`: encoder, head, device, and training configuration.

V1 folders also include `hard_test_metrics.json` and
`hard_test_predictions.csv`, which show how the same full-input model behaves
on the hard test.

## Setup and Running Instructions

Run all commands from the submitted folder root:

```bash
cd sourcecode
```

Recommended environment: Python 3.10 or 3.11.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run of MiniLM, CLIP, or Qwen3 may download pretrained model weights
through `sentence-transformers`/`transformers`.

To verify whether PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Use `--device cuda` when this prints `True`; otherwise use `--device cpu`. If a
CUDA-capable machine prints `False`, reinstall PyTorch with the CUDA wheel that
matches that machine, then rerun `python -m pip install -r requirements.txt`.

### Quick Smoke Tests

These commands run tiny TF-IDF jobs to check that the environment is working:

```bash
python scripts/run_v1_experiments.py --encoders tfidf --heads logreg --device cpu --limit-train 128 --limit-val 64 --limit-test 64 --min-train-rows 1 --epochs 3 --output-dir reproduced/smoke/v1_artifacts --report-dir reproduced/smoke/V1_full_input
python scripts/run_v2_prefix_experiments.py --encoders tfidf --heads logreg --device cpu --limit-train 256 --limit-val 128 --limit-test 128 --min-train-rows 1 --epochs 3 --output-dir reproduced/smoke/v2_artifacts --report-dir reproduced/smoke/V2_prefix
```

### Reproduce One Model Combination

Each model folder has a `program.py` entry point for that exact
encoder/head combination:

```bash
python V1_full_input/tfidf__logreg/program.py --device cuda
python V2_prefix/tfidf__logreg/program.py --device cuda
```

Change the folder name to rerun a different combination, for example
`V1_full_input/qwen3_0_6b__mlp/program.py`.

### Reproduce All V1 and V2 Experiments

```bash
python scripts/run_v1_experiments.py --device cuda
python scripts/run_v2_prefix_experiments.py --device cuda
```

By default, reproduced artifacts and report snapshots are written under:

```text
reproduced/artifacts/v1/
reproduced/artifacts/v2/
reproduced/V1_full_input/
reproduced/V2_prefix/
```

These reproduced outputs are separate from the tracked result files included in
this submission.

### Reproduce the Hard-Test Evaluation

After reproducing the V1 models, run:

```bash
python scripts/evaluate_hard_test.py --device cuda
```

This writes hard-test reproduction outputs to:

```text
reproduced/hard_test/
```

The hard-test CSV is already included at `data/hard_test/hard_test.csv`. To
regenerate it from the included script:

```bash
python scripts/build_hard_test_dataset.py
```

## Hard Test

- `hard_test/summary.csv` and `summary.md`: hard-test results for all eight V1
  encoder/head combinations.
- `hard_test/hard_accuracy.png`: accuracy bar chart.
- `hard_test/{{encoder}}__{{head}}/`: per-combination predictions, metrics,
  category metrics, and source run configuration.

## Report

- `report/overleaf_report_source.tex`: current Overleaf report source snapshot
  used for consistency checking against the organized data and results above.
- `report/content_consistency_audit.md`: line-by-line consistency notes between
  the report source and the current organized experiment evidence.
"""
    write_text(OUR_PART / "README.md", readme)


def main() -> None:
    copy_data_splits()
    copy_summaries()
    copy_reproduction_files()
    for encoder, head, combo in COMBOS:
        copy_v1_combo(encoder, head, combo)
        copy_v2_combo(encoder, head, combo)
    write_readme()
    print(f"Wrote organized project view to {OUR_PART}")


if __name__ == "__main__":
    main()
