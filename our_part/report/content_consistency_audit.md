# Report Consistency Audit

Source checked: `our_part/report/overleaf_report_source.tex`

Status after the latest edit: the report source has been updated to match the
current organized V1/V2 evidence in `our_part/`, except for the hard-test
section, which was intentionally left unchanged per request.

## Updated To Match Current Evidence

1. Dataset size and split counts

   The report now describes the tracked V1 full-input dataset as 3,000
   DeepSeek-generated utterances, split into 2,400 train / 300 validation /
   300 test rows.

   Evidence:
   - `our_part/README.md`
   - `our_part/data/v1_full_input/manifest.json`

2. Prefix ratios and prefix test size

   The report now uses the current V2 prefix ratios: 20%, 40%, 60%, 80%, and
   100%. It also states the current test prefix size: 300 source utterances
   and 1,500 prefix rows.

   Evidence:
   - `our_part/data/v2_prefix_input/manifest.json`

3. Full-input V1 accuracy table

   The report's V1 accuracy matrix now matches `our_part/V1_full_input/summary.csv`.

   | run | test accuracy |
   |---|---:|
   | tfidf__logreg | 0.9933 |
   | tfidf__mlp | 0.9900 |
   | minilm__logreg | 0.9900 |
   | minilm__mlp | 0.9867 |
   | clip__logreg | 0.9833 |
   | clip__mlp | 0.9967 |
   | qwen3_0_6b__logreg | 0.9900 |
   | qwen3_0_6b__mlp | 0.9933 |

4. V2 prefix and selective early-decision metrics

   The report now uses the current V2 prefix-ratio results and threshold
   summary metrics. At threshold 0.70, the report describes selective
   first-decision accuracy, decision coverage, correct coverage, and mean /
   median first prefix rather than treating `wait` as a correct final output.

   Evidence:
   - `our_part/V2_prefix/summary.csv`
   - `our_part/V2_prefix/*/test_metrics_by_prefix_ratio.csv`
   - `our_part/V2_prefix/*/test_threshold_sweep.csv`

5. Runtime and latency wording

   The report no longer claims single-utterance inference latency values.
   It now reports tracked classifier-head runtime after feature extraction,
   which is what the current summary files actually provide.

   Evidence:
   - `our_part/V1_full_input/summary.csv`
   - `our_part/V2_prefix/summary.csv`

6. Model implementation details

   The report now matches the current implementation details:

   - TF-IDF uses 1--2 grams with max 10,000 features for V1 and 15,000 for V2.
   - CLIP uses `sentence-transformers/clip-ViT-B-32`.
   - The LogReg option is a PyTorch linear classifier trained with
     cross-entropy and AdamW.
   - The MLP is Linear(256) -> ReLU -> Dropout -> Linear.
   - Training runs for up to 60 epochs with early stopping patience 15.

   Evidence:
   - `experiments/run_experiments.py`
   - `version2_prefix/scripts/run_prefix_experiments.py`

## Intentionally Unchanged

The hard-test discussion, hard-test table, FMR/FTR values, and keyword
baseline entries were intentionally kept as-is. These are still not backed by
the current organized V1/V2 result files in `our_part/`, so they should be
checked separately before final submission.
