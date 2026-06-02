# Run keyword baseline, prefix dataset build, and evaluations (Windows PowerShell).
# Requires: pip install -r requirements.txt
# Optional: train TF-IDF / MLP artifacts before running comparison on those models.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

Write-Host "==> Keyword baseline"
python modeling/scripts/train_keyword_baseline.py

Write-Host "==> Build prefix dataset"
python modeling/scripts/build_prefix_dataset.py

Write-Host "==> Prefix eval: keyword"
python modeling/scripts/eval_prefix.py --model keyword --output modeling/reports/prefix_eval_keyword.json

if (Test-Path "modeling/artifacts/tfidf_logreg/model.joblib") {
    Write-Host "==> Prefix eval: tfidf"
    python modeling/scripts/eval_prefix.py --model tfidf --output modeling/reports/prefix_eval_tfidf.json
} else {
    Write-Host "Skipping TF-IDF prefix eval (model not found)."
}

if (Test-Path "modeling/artifacts/embedding_mlp/model.pt") {
    Write-Host "==> Prefix eval: qwen mlp"
    python modeling/scripts/eval_prefix.py --model mlp --mlp-model-dir modeling/artifacts/embedding_mlp --output modeling/reports/prefix_eval_qwen_mlp.json
} else {
    Write-Host "Skipping MLP prefix eval (model not found)."
}

if (Test-Path "modeling/artifacts/tfidf_logreg/model.joblib") {
    Write-Host "==> Full comparison"
    python modeling/scripts/eval_comparison.py
} else {
    Write-Host "Skipping eval_comparison (TF-IDF model not found)."
}

Write-Host "Done. See modeling/reports/"
