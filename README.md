# CS182 Project — Humanoid Intent Classification

**Topic:** `chat` vs `motion_query` for humanoid robot dialogue, with prefix-based early prediction.

| Path | Purpose |
|------|---------|
| [Proposal.md](Proposal.md) | Project proposal (motivation, stages, metrics) |
| [neurips_2026.tex](neurips_2026.tex) | Final report (NeurIPS template) |
| [neurips_2026.sty](neurips_2026.sty) | Report style file |
| [mode-classifier-main/](mode-classifier-main/) | **Code + data + experiments** — see [README](mode-classifier-main/README.md) |

## Quick start

```bash
cd mode-classifier-main
pip install -r requirements.txt
```

See **mode-classifier-main/README.md** for training, comparison, and prefix evaluation.

## Top-level TF-IDF baseline

The simple TF-IDF + logistic regression baseline reads the top-level `data/`
folder:

```bash
pip install -r requirements.txt
python scripts/train_tfidf_logreg.py
```

On the local Windows machine, the Anaconda Python environment can run it with:

```powershell
& "D:\anaconda3\python.exe" scripts\train_tfidf_logreg.py
```

The script writes the trained model, metrics, and validation/test predictions to
`artifacts/tfidf_logreg/`.

## Our 4x2 Encoder/Head Experiments

Our implementation is in `experiments/`, separate from the imported
`mode-classifier-main/` folder.

It runs 4 encoders (`tfidf`, `minilm`, `clip`, `qwen3_0_6b`) with 2 classifier
heads (`logreg`, `mlp`), for 8 total combinations. The code uses CUDA for neural
encoders and classifier-head training when a GPU is available.

```powershell
& "D:\anaconda3\python.exe" experiments\run_experiments.py
```

Full outputs are written to `artifacts/ours/`, including embedding caches,
models, predictions, per-run `history.csv`, `loss_curve.png`, `metrics.json`,
and global `summary.csv` / `summary.md`. A lightweight report snapshot is also
written to `experiments/results/latest/` so training-loss curves and test
metrics can be committed without model weights. See `experiments/README.md` for
the exact file structure and run names.

## Version2 Prefix Early Classification

Version2 is isolated under `version2_prefix/`. It changes the task from
full-utterance binary classification to prefix-based three-class prediction:

```text
chat
motion_query
wait
```

Build the prefix dataset and run the version2 experiment grid with:

```powershell
& "D:\anaconda3\python.exe" version2_prefix\scripts\build_prefix_dataset.py
& "D:\anaconda3\python.exe" version2_prefix\scripts\run_prefix_experiments.py
```

See `version2_prefix/README.md` for the version2 data layout, artifact naming,
and report files.

## Submission zip

Include: PDF report, `mode-classifier-main/` (with its README), and this file if desired.
