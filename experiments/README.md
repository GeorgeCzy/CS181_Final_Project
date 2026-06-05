# Our Encoder/Classifier Experiments

This folder is the top-level project implementation and is intentionally
separate from `mode-classifier-main/`.

## Combinations

Encoders:

- `tfidf`
- `minilm` (`sentence-transformers/all-MiniLM-L6-v2`)
- `clip` (`sentence-transformers/clip-ViT-B-32`)
- `qwen3_0_6b` (`Qwen/Qwen3-Embedding-0.6B`)

Classifier heads:

- `logreg`: a single linear softmax layer, equivalent to multinomial logistic
  regression, trained with cross-entropy.
- `mlp`: a two-layer MLP classifier.

The full grid has 8 runs:

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

## Artifact Layout

All generated files are stored outside git under `artifacts/ours/`:

```text
artifacts/ours/
  experiment_plan.json
  summary.csv
  summary.md
  embeddings/
    {encoder}/
      metadata.json
      train_embeddings.npz
      val_embeddings.npz
      test_embeddings.npz
      vectorizer.joblib        # TF-IDF only
  runs/
    {encoder}__{head}/
      run_config.json
      history.csv              # epoch train/val loss and accuracy
      loss_curve.png
      metrics.json             # train/val/test metrics
      model.pt
      val_predictions.csv
      test_predictions.csv
experiments/results/latest/
  summary.csv
  summary.md
  runs/
    {encoder}__{head}/
      history.csv
      loss_curve.png
      metrics.json
```

`artifacts/ours/` is ignored by git because it contains embedding caches and
model weights. `experiments/results/latest/` is lightweight and can be committed
for reports.

## Commands

Run all 8 combinations with GPU when available:

```bash
python experiments/run_experiments.py
```

On the local Windows machine:

```powershell
& "D:\anaconda3\python.exe" experiments\run_experiments.py
```

Run only one combination:

```powershell
& "D:\anaconda3\python.exe" experiments\run_experiments.py --encoders tfidf --heads logreg
```

Run a quick smoke test:

```powershell
& "D:\anaconda3\python.exe" experiments\run_experiments.py `
  --encoders tfidf --heads logreg `
  --limit-train 128 --limit-val 64 --limit-test 64 --min-train-rows 1 --epochs 3
```
