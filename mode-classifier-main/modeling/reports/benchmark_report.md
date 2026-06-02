# Mode Classifier Benchmark Report

- **Generated**: 2026-06-02 05:45 UTC
- **Dataset**: data_generation\data\raw\deepseek_generated_10000.csv (80/10/10 stratified)
- **Device**: cuda
- **Task**: binary classification (`chat` vs `motion_query`)

## Summary

Comparison of **4 text encoders** × **2 classifier heads**.
Latency = end-to-end single-utterance inference (encode + classify).

| Encoder | Head | Test Acc | Hard Acc | Train Time | Latency (mean) | Latency (p95) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | logreg | 0.9940 | 0.8667 | 4.8s | 1.6 ms | 2.3 ms |
| tfidf | mlp | 0.9970 | 0.8333 | 25.7s | 1.7 ms | 2.8 ms |
| clip | logreg | 0.9980 | 0.7667 | 37.0s | 24.2 ms | 34.0 ms |
| clip | mlp | 0.9980 | 0.8333 | 19.4s | 23.5 ms | 30.3 ms |
| qwen | logreg | 0.9980 | 0.7667 | 55.1s | 132.8 ms | 152.6 ms |
| qwen | mlp | 0.9960 | 0.7000 | 19.9s | 130.5 ms | 151.7 ms |
| minilm | logreg | 0.9940 | 0.9000 | 33.1s | 15.3 ms | 21.2 ms |
| minilm | mlp | 0.9980 | 0.8667 | 30.4s | 15.1 ms | 19.7 ms |

## Accuracy (4 × 2 matrix, test split)

| Encoder | LogReg | MLP |
| --- | ---: | ---: |
| tfidf | 0.9940 | 0.9970 |
| clip | 0.9980 | 0.9980 |
| qwen | 0.9980 | 0.9960 |
| minilm | 0.9940 | 0.9980 |

## Training Time

- **tfidf+logreg**: head 4.8s = **4.8s**
- **tfidf+mlp**: head 25.7s = **25.7s**
- **clip+logreg**: cache 27.2s + head 9.8s = **37.0s**
- **clip+mlp**: head 19.4s = **19.4s**
- **qwen+logreg**: cache 45.1s + head 10.0s = **55.1s**
- **qwen+mlp**: head 19.9s = **19.9s**
- **minilm+logreg**: cache 23.4s + head 9.7s = **33.1s**
- **minilm+mlp**: head 30.4s = **30.4s**

## Inference Latency (single utterance)

Measured on 200 test utterances (5 warmup runs). Unit: milliseconds.

| Model | Mean | p50 | p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf+logreg | 1.6 | 1.6 | 2.3 | 1.1 | 3.0 |
| tfidf+mlp | 1.7 | 1.5 | 2.8 | 1.1 | 3.5 |
| clip+logreg | 24.2 | 22.7 | 34.0 | 18.8 | 41.7 |
| clip+mlp | 23.5 | 21.6 | 30.3 | 19.1 | 40.0 |
| qwen+logreg | 132.8 | 130.7 | 152.6 | 113.5 | 182.9 |
| qwen+mlp | 130.5 | 127.8 | 151.7 | 110.2 | 184.2 |
| minilm+logreg | 15.3 | 14.2 | 21.2 | 12.2 | 25.4 |
| minilm+mlp | 15.1 | 14.3 | 19.7 | 12.2 | 24.4 |

## Hard-set Accuracy (boundary cases)

| Model | Accuracy | Macro F1 | False Motion | False Text |
| --- | ---: | ---: | ---: | ---: |
| tfidf+logreg | 0.8667 | 0.8661 | 0.0333 | 0.1000 |
| tfidf+mlp | 0.8333 | 0.8316 | 0.0667 | 0.1000 |
| clip+logreg | 0.7667 | 0.7436 | 0.2000 | 0.0333 |
| clip+mlp | 0.8333 | 0.8316 | 0.0667 | 0.1000 |
| qwen+logreg | 0.7667 | 0.7436 | 0.2000 | 0.0333 |
| qwen+mlp | 0.7000 | 0.6970 | 0.1333 | 0.1667 |
| minilm+logreg | 0.9000 | 0.8942 | 0.1000 | 0.0000 |
| minilm+mlp | 0.8667 | 0.8643 | 0.0667 | 0.0667 |

## Notes

- **Train time** for embedding models includes one-time encoder caching on the full 10k dataset, amortized across both LogReg and MLP heads of the same encoder.
- **Latency** includes model loading is excluded; numbers reflect steady-state per-query inference.
- TF-IDF models are CPU-friendly; embedding models ran on the configured device above.
