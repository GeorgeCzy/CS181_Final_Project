# Our Project Part

This folder is the submitted package for our part of the project. In the final
submission, this folder may be renamed to `sourcecode/`; all paths below are
relative to that submitted folder.

No setup is required to inspect the submitted results. The CSV, JSON, PNG, and
TeX files here are the tracked outputs used in the report.

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

- `program.py`: source/provenance file for the corresponding encoder/head
  combination. It records the original combination-specific training entry
  used to produce the tracked outputs in this folder; running it is not
  required to inspect the submitted results.
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

## Setup and Running Instructions

For the submitted `sourcecode/` folder itself, there is no setup step. Reviewers
can inspect all data, results, plots, and report source directly from the files
listed above.

The tracked outputs have already been generated. The `program.py` files in the
model folders are included to show the source entry point for each
encoder/head combination, but this submitted folder is intended primarily as a
readable result package rather than a standalone retraining environment.

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
