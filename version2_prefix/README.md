# Version2 Prefix Early Classification

This is an isolated version2 implementation. It does not depend on or modify
the original top-level `experiments/` implementation, and it does not use the
imported `mode-classifier-main/` folder.

## Task

Version1 predicts after seeing the full utterance:

```text
chat
motion_query
```

Version2 predicts while seeing only a prefix of the utterance:

```text
chat
motion_query
wait
```

`wait` means the visible prefix is not informative enough yet, so the system
should listen for more input.

## Data Layout

Generated version2 data is stored here:

```text
version2_prefix/data/
  prefix_train.csv
  prefix_val.csv
  prefix_test.csv
  prefix_manifest.json
```

Important columns:

- `prefix_utterance`: model input.
- `masked_utterance`: human-readable inspection field showing the hidden suffix.
- `full_utterance`: original complete utterance.
- `full_label`: original `chat` or `motion_query` label.
- `label`: version2 three-class target label.
- `target_prefix_ratio`: requested prefix ratio, such as `0.20` or `1.00`.
- `visible_ratio`: actual token ratio after rounding to whole tokens.

Build the data from the top-level `data/` splits:

```powershell
& "D:\anaconda3\python.exe" version2_prefix\scripts\build_prefix_dataset.py
```

## Experiment Layout

Full generated artifacts are ignored by git:

```text
artifacts/version2_prefix/
  embeddings/{encoder}/
  runs/{encoder}__{head}/
```

Lightweight report files are committed here:

```text
version2_prefix/results/latest/
  summary.csv
  summary.md
  runs/{encoder}__{head}/
    history.csv
    loss_curve.png
    metrics.json
    test_metrics_by_prefix_ratio.csv
    test_prefix_metrics.png
```

Run all 8 combinations:

```powershell
& "D:\anaconda3\python.exe" version2_prefix\scripts\run_prefix_experiments.py
```

Run one combination:

```powershell
& "D:\anaconda3\python.exe" version2_prefix\scripts\run_prefix_experiments.py --encoders tfidf --heads logreg
```

Run a smoke test:

```powershell
& "D:\anaconda3\python.exe" version2_prefix\scripts\run_prefix_experiments.py `
  --encoders tfidf --heads logreg `
  --limit-train 256 --limit-val 128 --limit-test 128 --min-train-rows 1 --epochs 3
```

## Run Names

```text
tfidf__logreg
tfidf__mlp
minilm__logreg
minilm__mlp
clip__logreg
clip__mlp
qwen3_0_6b__logreg
qwen3_0_6b__mlp
```
