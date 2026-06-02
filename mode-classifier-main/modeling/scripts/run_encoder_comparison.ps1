# Cache embeddings, train all 4 encoders × 2 heads, and compare (Windows PowerShell).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

Write-Host "==> TF-IDF + LogReg"
python modeling/scripts/train_tfidf_logreg.py

Write-Host "==> TF-IDF + MLP"
python modeling/scripts/train_tfidf_mlp.py

Write-Host "==> Cache CLIP embeddings"
python modeling/scripts/cache_text_embeddings.py `
  --encoder-backend clip `
  --model-name openai/clip-vit-base-patch32 `
  --no-text-prefix `
  --output modeling/data/embeddings/clip_deepseek_10000.npz

Write-Host "==> CLIP + LogReg / MLP"
python modeling/scripts/train_embedding_logreg.py `
  --embedding-cache modeling/data/embeddings/clip_deepseek_10000.npz `
  --output-dir modeling/artifacts/clip_logreg
python modeling/scripts/train_embedding_mlp.py `
  --embedding-cache modeling/data/embeddings/clip_deepseek_10000.npz `
  --output-dir modeling/artifacts/clip_mlp

Write-Host "==> Cache Qwen embeddings"
python modeling/scripts/cache_text_embeddings.py `
  --model-name Qwen/Qwen3-Embedding-0.6B `
  --output modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz

Write-Host "==> Qwen + LogReg / MLP"
python modeling/scripts/train_embedding_logreg.py `
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz `
  --output-dir modeling/artifacts/qwen_logreg
python modeling/scripts/train_embedding_mlp.py `
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz `
  --output-dir modeling/artifacts/qwen_mlp

Write-Host "==> Cache MiniLM embeddings"
python modeling/scripts/cache_text_embeddings.py `
  --model-name sentence-transformers/all-MiniLM-L6-v2 `
  --no-text-prefix `
  --output modeling/data/embeddings/minilm_deepseek_10000.npz

Write-Host "==> MiniLM + LogReg / MLP"
python modeling/scripts/train_embedding_logreg.py `
  --embedding-cache modeling/data/embeddings/minilm_deepseek_10000.npz `
  --output-dir modeling/artifacts/minilm_logreg
python modeling/scripts/train_embedding_mlp.py `
  --embedding-cache modeling/data/embeddings/minilm_deepseek_10000.npz `
  --output-dir modeling/artifacts/minilm_mlp

Write-Host "==> Compare all 8 models"
python modeling/scripts/eval_comparison.py

Write-Host "Done. See modeling/reports/comparison_summary.md"
