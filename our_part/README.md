# Our Project Part

This directory is a clean view of our contribution. It does not replace the
original runnable folders; it collects the data splits and experiment outputs
in a structure that is easier to read for the final report/presentation.

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

- `program.py`: launcher that runs only this combination.
- `shared_program_source.py`: source snapshot of the shared V1 experiment
  runner.
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
