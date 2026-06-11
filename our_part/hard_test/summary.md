# Hard Test Summary

The hard test contains 150 examples: 73 chat and 77 motion_query.
FMR/FTR below are reported as rates over all hard-test examples, matching the report table.

| run | accuracy | macro F1 | FMR | FTR | false motion | false text |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_0_6b__mlp | 0.8400 | 0.8390 | 0.1067 | 0.0533 | 16 | 8 |
| clip__mlp | 0.8333 | 0.8321 | 0.1133 | 0.0533 | 17 | 8 |
| tfidf__mlp | 0.7933 | 0.7918 | 0.1333 | 0.0733 | 20 | 11 |
| tfidf__logreg | 0.7867 | 0.7848 | 0.1400 | 0.0733 | 21 | 11 |
| minilm__mlp | 0.7733 | 0.7707 | 0.1533 | 0.0733 | 23 | 11 |
| clip__logreg | 0.7533 | 0.7463 | 0.1933 | 0.0533 | 29 | 8 |
| qwen3_0_6b__logreg | 0.7533 | 0.7438 | 0.2067 | 0.0400 | 31 | 6 |
| minilm__logreg | 0.6800 | 0.6684 | 0.2400 | 0.0800 | 36 | 12 |
