# Our Project Part

This folder is the submitted package for our part of the project. It is
organized so that a reader can inspect the data, experiment outputs, and report
source without needing to browse the rest of the repository.

If only this `our_part/` folder is submitted, no setup is required for reading
the results. The CSV, JSON, PNG, and TeX files here are the tracked outputs used
in the report.

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
- `data/hard_test/hard_test.csv`: 150-example combined hard test for
  full-utterance binary classification.
- `data/hard_test/manifest.json`: hard-test design metadata.

| Version | Train | Validation | Test |
|---------|-------|------------|------|
| V1 full input | 2400 | 300 | 300 |
| V2 prefix input | 12000 | 1500 | 1500 |

## V1_full_input

Each subfolder is one encoder/head combination, named as:

```text
{encoder}__{head}
```

Each combination folder contains:

- `program.py`: one-combination launcher used when the complete repository is
  available. It records how to rerun the encoder/head combination named by the
  folder.
- `training_loss.png`: training/validation loss curve.
- `training_history.csv`: epoch-by-epoch training history.
- `test_results.json`: metrics and classification report.
- `hard_test_metrics.json`, `hard_test_predictions.csv`,
  `hard_test_category_metrics.csv`: results on the combined hard test.
- `run_config.json`: exact run configuration.

## V2_prefix

The V2 folders use the same `{encoder}__{head}` naming. Each folder contains
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
V1_full_input/{encoder}__{head}/
V2_prefix/{encoder}__{head}/
```

The most useful files inside each model folder are:

- `training_loss.png`: loss curve for convergence checking.
- `training_history.csv`: epoch-level train/validation loss and accuracy.
- `test_results.json`: final test metrics.
- `run_config.json`: encoder, head, device, and training configuration.

V1 folders also include `hard_test_metrics.json` and
`hard_test_predictions.csv`, which show how the same full-input model behaves
on the hard test.

## Optional Reproduction Instructions

The submitted `our_part/` folder is self-contained for inspection, but the
`program.py` launchers and full retraining commands require the complete
repository because they call scripts in `experiments/`, `version2_prefix/`, and
`scripts/`.

If the complete repository is available, run commands from the repository root:

```powershell
cd <full-repo-root>
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run one organized V1 model folder:

```powershell
python our_part/V1_full_input/tfidf__logreg/program.py --device cuda --skip-existing
```

Run one organized V2 model folder:

```powershell
python our_part/V2_prefix/tfidf__logreg/program.py --device cuda --skip-existing
```

Each `program.py` runs only the encoder/head combination named by its folder.
Use `--device cpu` if CUDA is unavailable. Use `--skip-existing` to reuse
already trained artifacts.

Re-run all V1 full-input experiments:

```powershell
python experiments/run_experiments.py --device cuda
```

Re-run all V2 prefix experiments:

```powershell
python version2_prefix/scripts/run_prefix_experiments.py --device cuda
```

Rebuild the organized `our_part/` view after re-running experiments:

```powershell
python scripts/organize_our_part.py
```

Rebuild and evaluate the hard test:

```powershell
python scripts/build_hard_test_dataset.py
python scripts/evaluate_hard_test.py --device cuda
```

Use `--device cpu` instead of `--device cuda` if GPU/CUDA is unavailable.

## Hard Test

- `hard_test/summary.csv` and `summary.md`: hard-test results for all eight V1
  encoder/head combinations.
- `hard_test/hard_accuracy.png`: accuracy bar chart.
- `hard_test/{encoder}__{head}/`: per-combination predictions, metrics,
  category metrics, and source run configuration.

## Report

- `report/overleaf_report_source.tex`: current Overleaf report source snapshot
  used for consistency checking against the organized data and results above.
- `report/content_consistency_audit.md`: line-by-line consistency notes between
  the report source and the current organized experiment evidence.
