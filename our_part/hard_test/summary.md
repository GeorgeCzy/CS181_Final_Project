# Hard Test Summary

The hard test contains 120 manually curated examples: 60 chat and 60 motion_query.
FMR/FTR below are reported as rates over all hard-test examples, matching the report table.

| run | accuracy | macro F1 | FMR | FTR | false motion | false text |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_0_6b__mlp | 0.8417 | 0.8398 | 0.1333 | 0.0250 | 16 | 3 |
| clip__mlp | 0.8250 | 0.8235 | 0.1333 | 0.0417 | 16 | 5 |
| tfidf__logreg | 0.7750 | 0.7723 | 0.1667 | 0.0583 | 20 | 7 |
| tfidf__mlp | 0.7667 | 0.7650 | 0.1583 | 0.0750 | 19 | 9 |
| minilm__mlp | 0.7583 | 0.7545 | 0.1833 | 0.0583 | 22 | 7 |
| clip__logreg | 0.7583 | 0.7507 | 0.2083 | 0.0333 | 25 | 4 |
| qwen3_0_6b__logreg | 0.7417 | 0.7279 | 0.2417 | 0.0167 | 29 | 2 |
| minilm__logreg | 0.6583 | 0.6401 | 0.2833 | 0.0583 | 34 | 7 |
