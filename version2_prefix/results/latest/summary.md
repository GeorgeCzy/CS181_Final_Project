# Version2 Prefix Experiment Summary

| run | val acc | test acc | final-prefix acc | avg decision rate | epochs |
|---|---:|---:|---:|---:|---:|
| clip__mlp | 0.9607 | 0.9540 | 0.9767 | 0.7820 | 60 |
| qwen3_0_6b__mlp | 0.9487 | 0.9500 | 0.9700 | 0.7827 | 38 |
| tfidf__mlp | 0.9267 | 0.9440 | 0.9367 | 0.7740 | 19 |
| minilm__mlp | 0.9607 | 0.9440 | 0.9733 | 0.7953 | 44 |
| tfidf__logreg | 0.9147 | 0.9380 | 0.9233 | 0.7733 | 60 |
| qwen3_0_6b__logreg | 0.9187 | 0.9147 | 0.9667 | 0.7920 | 60 |
| minilm__logreg | 0.9127 | 0.9067 | 0.9633 | 0.8047 | 60 |
| clip__logreg | 0.8887 | 0.8960 | 0.9667 | 0.8040 | 60 |
