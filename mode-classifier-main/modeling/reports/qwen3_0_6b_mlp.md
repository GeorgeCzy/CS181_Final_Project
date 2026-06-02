# Qwen3-Embedding-0.6B + MLP

Run date: 2026-05-20

Embedding cache:

```text
modeling/data/embeddings/qwen3_0_6b_deepseek_2000.npz
```

Embedding metadata:

```text
rows: 2000
embedding_dim: 1024
model_name: Qwen/Qwen3-Embedding-0.6B
normalize_embeddings: true
```

Classifier:

```text
LayerNorm -> Dropout -> Linear(1024, 256) -> GELU -> Dropout -> Linear(256, 2)
```

Training:

```text
epochs: 20
patience: 5
best_epoch: 2
```

Results:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.9900 |
| Test accuracy | 0.9750 |
| Test loss | 0.0554 |

Spot checks:

| Text | Prediction | p_motion_query |
| --- | --- | ---: |
| Can you do a short dance? | motion_query | 0.9287 |
| Can you dance? | motion_query | 0.7131 |
| Please do not move, just explain the answer. | motion_query | 0.6293 |

Notes:

- The embedding model correctly fixes the TF-IDF miss on "Can you do a short dance?"
- It still over-predicts `motion_query` on some negated or capability-style boundary cases.
- The next evaluation step should be a curated hard test set focused on these boundary patterns.

