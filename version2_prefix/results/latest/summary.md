# Version2 Prefix Selective-Decision Summary

At the summary threshold, `wait` is treated as abstention, not as a correct answer.

| run | threshold | first-decision acc | first-decision coverage | correct coverage | mean first prefix | forced final acc |
|---|---:|---:|---:|---:|---:|---:|
| clip__mlp | 0.70 | 0.9898 | 0.9767 | 0.9667 | 0.4754 | 0.9900 |
| qwen3_0_6b__logreg | 0.70 | 0.9932 | 0.9733 | 0.9667 | 0.5009 | 0.9867 |
| qwen3_0_6b__mlp | 0.70 | 0.9864 | 0.9800 | 0.9667 | 0.4757 | 0.9833 |
| minilm__mlp | 0.70 | 0.9731 | 0.9900 | 0.9633 | 0.4712 | 0.9800 |
| minilm__logreg | 0.70 | 0.9861 | 0.9600 | 0.9467 | 0.4959 | 0.9700 |
| clip__logreg | 0.70 | 1.0000 | 0.9433 | 0.9433 | 0.5235 | 0.9800 |
| tfidf__mlp | 0.70 | 0.9924 | 0.8767 | 0.8700 | 0.4183 | 0.9900 |
| tfidf__logreg | 0.70 | 0.9923 | 0.8667 | 0.8600 | 0.4212 | 0.9933 |
