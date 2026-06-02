# Model Comparison Summary

Encoders: **TF-IDF**, **CLIP**, **Qwen3-Embedding-0.6B**, **MiniLM**.
Heads: **Logistic Regression**, **MLP**.

## Test accuracy (4 × 2)

| Encoder | LogReg | MLP |
| --- | ---: | ---: |
| tfidf | 0.9940 | 0.9970 |
| clip | 0.9980 | 0.9980 |
| qwen | 0.9980 | 0.9960 |
| minilm | 0.9940 | 0.9980 |

## tfidf+logreg

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9940 | 0.9940 | 0.0040 | 0.0020 |
| hard | 0.8667 | 0.8661 | 0.0333 | 0.1000 |
| val | 0.9970 | 0.9970 | 0.0010 | 0.0020 |

## tfidf+mlp

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9970 | 0.9970 | 0.0020 | 0.0010 |
| hard | 0.8333 | 0.8316 | 0.0667 | 0.1000 |
| val | 0.9990 | 0.9990 | 0.0000 | 0.0010 |

## clip+logreg

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9980 | 0.9980 | 0.0020 | 0.0000 |
| hard | 0.7667 | 0.7436 | 0.2000 | 0.0333 |
| val | 0.9950 | 0.9950 | 0.0050 | 0.0000 |

## clip+mlp

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9980 | 0.9980 | 0.0010 | 0.0010 |
| hard | 0.8333 | 0.8316 | 0.0667 | 0.1000 |
| val | 0.9980 | 0.9980 | 0.0010 | 0.0010 |

## qwen+logreg

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9980 | 0.9980 | 0.0010 | 0.0010 |
| hard | 0.7667 | 0.7436 | 0.2000 | 0.0333 |
| val | 0.9950 | 0.9950 | 0.0050 | 0.0000 |

## qwen+mlp

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9960 | 0.9960 | 0.0010 | 0.0030 |
| hard | 0.7000 | 0.6970 | 0.1333 | 0.1667 |
| val | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## minilm+logreg

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9940 | 0.9940 | 0.0050 | 0.0010 |
| hard | 0.9000 | 0.8942 | 0.1000 | 0.0000 |
| val | 0.9950 | 0.9950 | 0.0050 | 0.0000 |

## minilm+mlp

| Split | Accuracy | Macro F1 | False Motion Rate | False Text Rate |
| --- | ---: | ---: | ---: | ---: |
| test | 0.9980 | 0.9980 | 0.0020 | 0.0000 |
| hard | 0.8667 | 0.8643 | 0.0667 | 0.0667 |
| val | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

