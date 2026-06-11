# Hard Test Evaluation

This folder contains the V1 full-utterance hard-test evaluation.

## Dataset

The dataset is stored at `../data/hard_test/hard_test.csv`.
It contains 150 examples:

- 120 manually curated base examples
- 30 externally provided extension examples in `../data/hard_test/provided_hard_test.csv`
- 73 `chat`
- 77 `motion_query`

The examples are designed to stress lexical shortcuts:

- chat examples with motion words
- negation and "do not move" cases
- capability and hypothetical questions
- word-only explanation requests
- indirect physical requests
- demonstration and context-driven action requests

## Outputs

- `summary.csv`: metrics for all eight encoder/head combinations.
- `summary.md`: readable summary table.
- `hard_accuracy.png`: hard-test accuracy bar chart.
- `evaluation_manifest.json`: evaluation configuration.
- `{encoder}__{head}/metrics.json`: full metrics for one combination.
- `{encoder}__{head}/predictions.csv`: per-example predictions and probabilities.
- `{encoder}__{head}/category_metrics.csv`: accuracy by hard-test category.
- `{encoder}__{head}/source_run_config.json`: original V1 run configuration.

FMR and FTR in the summary are rates over all 150 hard-test examples, matching
the report table.
