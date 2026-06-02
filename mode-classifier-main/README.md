# Mode Classifier

Binary classifier: human utterance → **`chat`** (verbal reply) or **`motion_query`** (physical action).

Includes full-sentence model comparison and **prefix / early prediction** (word-level masking of later tokens).

## Setup

```bash
pip install -r requirements.txt
```

GPU optional (TF-IDF / keyword need CPU only; embedding cache + MLP benefit from GPU). Server notes: `modeling/server_setup.md`, conda: `modeling/environment.yml`.

## Data

| File | Description |
|------|-------------|
| `data_generation/data/raw/deepseek_generated_10000.csv` | Main dataset (10000 rows) |
| `data_generation/data/raw/deepseek_generated_2000.csv` | Original subset (kept for reference) |
| `data_generation/data/reference/manual_seed_500.csv` | Reference examples for regeneration |
| `modeling/data/splits/{train,val,test}.csv` | Stratified 80/10/10 split |
| `modeling/data/hard_test.csv` | 30 boundary cases for analysis |

Build or refresh the 10k dataset:

```bash
# Default: merge existing 2000 + 8000 template-generated rows
python data_generation/scripts/build_dataset_10000.py

# Optional: fill the gap with DeepSeek API instead of templates
python data_generation/scripts/build_dataset_10000.py --use-deepseek

python modeling/scripts/create_splits.py
```

## Workflow

### 1. Baselines (optional)

```bash
python modeling/scripts/train_keyword_baseline.py
python modeling/scripts/train_tfidf_logreg.py
```

### 2. Encoder × head comparison (4 × 2)

Text encoders: **TF-IDF**, **CLIP** (`openai/clip-vit-base-patch32`), **Qwen3-Embedding-0.6B**, **MiniLM**.  
Classifier heads: **Logistic Regression**, **MLP** (same 2-layer head for all encoders).

|  | LogReg | MLP |
| --- | --- | --- |
| TF-IDF | `train_tfidf_logreg.py` | `train_tfidf_mlp.py` |
| CLIP | cache + `train_embedding_logreg.py` | cache + `train_embedding_mlp.py` |
| Qwen3 | cache + `train_embedding_logreg.py` | cache + `train_embedding_mlp.py` |
| MiniLM | cache + `train_embedding_logreg.py` | cache + `train_embedding_mlp.py` |

```bash
# One-shot (train/cache all 8 + compare)
.\modeling\scripts\run_encoder_comparison.ps1
```

Or step by step:

```bash
# --- TF-IDF ---
python modeling/scripts/train_tfidf_logreg.py
python modeling/scripts/train_tfidf_mlp.py

# --- CLIP ---
python modeling/scripts/cache_text_embeddings.py \
  --encoder-backend clip \
  --model-name openai/clip-vit-base-patch32 \
  --no-text-prefix \
  --output modeling/data/embeddings/clip_deepseek_10000.npz

python modeling/scripts/train_embedding_logreg.py \
  --embedding-cache modeling/data/embeddings/clip_deepseek_10000.npz \
  --output-dir modeling/artifacts/clip_logreg

python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/clip_deepseek_10000.npz \
  --output-dir modeling/artifacts/clip_mlp

# --- Qwen3-Embedding-0.6B ---
python modeling/scripts/cache_text_embeddings.py \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --output modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz

python modeling/scripts/train_embedding_logreg.py \
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz \
  --output-dir modeling/artifacts/qwen_logreg

python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/qwen3_0_6b_deepseek_10000.npz \
  --output-dir modeling/artifacts/qwen_mlp

# --- MiniLM ---
python modeling/scripts/cache_text_embeddings.py \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --no-text-prefix \
  --output modeling/data/embeddings/minilm_deepseek_10000.npz

python modeling/scripts/train_embedding_logreg.py \
  --embedding-cache modeling/data/embeddings/minilm_deepseek_10000.npz \
  --output-dir modeling/artifacts/minilm_logreg

python modeling/scripts/train_embedding_mlp.py \
  --embedding-cache modeling/data/embeddings/minilm_deepseek_10000.npz \
  --output-dir modeling/artifacts/minilm_mlp

# Compare all 8 on test + hard set
python modeling/scripts/eval_comparison.py
```

Writes `modeling/reports/comparison_summary.md` (and `.json`).

Optional keyword rule baseline in the same report:

```bash
python modeling/scripts/eval_comparison.py --include-keyword
```

### 3. Prefix / early prediction

```bash
python modeling/scripts/build_prefix_dataset.py

python modeling/scripts/eval_prefix.py --model keyword --output modeling/reports/prefix_eval_keyword.json
python modeling/scripts/eval_prefix.py --model tfidf --output modeling/reports/prefix_eval_tfidf.json
python modeling/scripts/eval_prefix.py --model mlp --mlp-model-dir modeling/artifacts/embedding_mlp \
  --output modeling/reports/prefix_eval_qwen_mlp.json
```

Presentation demo:

```bash
python modeling/scripts/run_prefix_demo.py --model tfidf --text "Can you dance for me?" --threshold 0.7
```

Windows batch:

```powershell
.\modeling\scripts\run_experiments.ps1
```

## Layout

```text
mode-classifier-main/
  requirements.txt
  data_generation/
    data/raw/deepseek_generated_2000.csv
    data/reference/manual_seed_500.csv
    prompts/                    # LLM generation templates
    scripts/                    # generate, merge, validate (optional)
  modeling/
    data/splits/, hard_test.csv
    scripts/                    # train, eval, demo
    reports/                    # experiment notes (*.md); regenerable *.json
    artifacts/                  # trained weights (gitignored except local runs)
    server_setup.md
```

## Scripts

| Script | Role |
|--------|------|
| `train_keyword_baseline.py` | Rule-based baseline |
| `train_tfidf_logreg.py` | TF-IDF + logistic regression |
| `train_tfidf_mlp.py` | TF-IDF + MLP |
| `cache_text_embeddings.py` | Cache sentence embeddings (Qwen / MiniLM / CLIP) |
| `train_embedding_logreg.py` | Logistic regression on cached embeddings |
| `train_embedding_mlp.py` | MLP head on cached embeddings |
| `eval_comparison.py` | 4 encoders × 2 heads comparison |
| `run_encoder_comparison.ps1` | Train/cache and compare all 8 models |
| `build_prefix_dataset.py` | Prefix rows at 25/50/75/100% |
| `eval_prefix.py` | Early-prediction metrics |
| `run_prefix_demo.py` | Word-by-word demo |
| `predict_*.py` | Single-utterance inference |

## Reference results

Documented in `modeling/reports/tfidf_logreg_baseline.md` and `qwen3_0_6b_mlp.md`.

## Extend dataset (optional)

Requires DeepSeek API key; see `data_generation/scripts/generate_with_deepseek.py`.
