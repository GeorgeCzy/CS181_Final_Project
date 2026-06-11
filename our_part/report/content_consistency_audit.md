# Report Consistency Audit

Source checked: `our_part/report/overleaf_report_source.tex`

This audit compares the Overleaf report source against the organized current
project evidence in `our_part/`. The LaTeX source itself was not edited.

## Major Inconsistencies

1. Dataset size and split counts

   The report says the full-input dataset has 10,000 utterances split into
   8,000 train / 1,000 validation / 1,000 test rows. Current V1 data has
   3,000 utterances split into 2,400 train / 300 validation / 300 test rows.

   Evidence:
   - Report lines 39, 73, 122-124, 367, 378.
   - `our_part/README.md`
   - `our_part/data/v1_full_input/manifest.json`

2. Template-expanded data and hard test set

   The report says 8,000 template-expanded rows and a separate 30-example hard
   test set were used. The current organized project data does not contain
   these template-expanded rows or a hard-set evaluation artifact.

   Evidence:
   - Report lines 74, 126, 135, 229-266.
   - No hard-set file or hard-set metrics exist outside `mode-classifier-main/`.

3. Prefix ratios and prefix test size

   The report says prefix evaluation uses 25%, 50%, 75%, and 100% prefixes,
   yielding 4,000 test prefix rows from 1,000 source utterances. Current V2
   uses five ratios: 0.20, 0.40, 0.60, 0.80, and 1.00. The current test split
   has 300 source utterances and 1,500 prefix rows.

   Evidence:
   - Report lines 69, 176-177, 302-303, 311-316.
   - `our_part/data/v2_prefix_input/manifest.json`

4. Full-input accuracy table

   The report's V1 accuracy matrix does not match the current V1 summary. The
   current test accuracies are:

   | run | test accuracy |
   |---|---:|
   | clip__mlp | 0.9967 |
   | tfidf__logreg | 0.9933 |
   | qwen3_0_6b__mlp | 0.9933 |
   | tfidf__mlp | 0.9900 |
   | minilm__logreg | 0.9900 |
   | qwen3_0_6b__logreg | 0.9900 |
   | minilm__mlp | 0.9867 |
   | clip__logreg | 0.9833 |

   Evidence:
   - Report lines 42, 208, 214-223.
   - `our_part/V1_full_input/summary.csv`

5. Prefix accuracy and early-decision claims

   The report says TF-IDF+LogReg reaches 94.5% accuracy at 25% prefix, 98.0%
   averaged over prefix lengths, and mean correct-decision latency of 33% at
   threshold 0.7. Current V2 does not have a 25% prefix condition. At threshold
   0.7, current TF-IDF+LogReg has:

   - first-decision accuracy: 0.9923
   - first-decision coverage: 0.8667
   - correct coverage: 0.8600
   - mean first-decision visible ratio: 0.4212
   - median first-decision visible ratio: 0.3333
   - forced-final accuracy: 0.9933

   The 33% value is the current median, not the mean.

   Evidence:
   - Report lines 46-49, 314-331, 357.
   - `our_part/V2_prefix/summary.csv`
   - `our_part/V2_prefix/tfidf__logreg/test_threshold_sweep.csv`

6. Latency and training-time table

   The report includes inference latency measurements such as 1.6 ms for
   TF-IDF+LogReg and about 130 ms for Qwen3. The current tracked summaries do
   not include single-utterance inference latency benchmarks. The V1/V2
   `elapsed_seconds` fields are run/training elapsed times and do not support
   the report latency table.

   Evidence:
   - Report lines 43-44, 196, 273-295, 370-374.
   - `our_part/V1_full_input/summary.csv`
   - `our_part/V2_prefix/summary.csv`

7. Model implementation details

   Several method descriptions differ from the current code:

   - Report says TF-IDF uses max 50,000 features. Current defaults are 10,000
     for V1 and 15,000 for V2.
   - Report says CLIP uses `openai/clip-vit-base-patch32`. Current code uses
     `sentence-transformers/clip-ViT-B-32`.
   - Report says LogReg is sklearn logistic regression. Current `logreg` head
     is a PyTorch linear layer trained with cross-entropy and AdamW.
   - Report says MLP is LayerNorm -> Linear -> GELU -> Linear. Current MLP is
     Linear -> ReLU -> Dropout -> Linear.
   - Report says heads are trained for 30 epochs. Current default is 60 epochs
     with early stopping.

   Evidence:
   - Report lines 163-170, 189-190.
   - `experiments/run_experiments.py`
   - `version2_prefix/scripts/run_prefix_experiments.py`

## Consistent High-Level Claims

- The project compares four encoders: TF-IDF, MiniLM, CLIP, and Qwen3-0.6B.
- The project compares two classifier heads: a linear/logreg-style head and an
  MLP head.
- V1 is full-input binary classification over `chat` and `motion_query`.
- V2 is prefix-based early classification with a third `wait` label.
- V2 selective-decision metrics correctly treat `wait` as abstention rather
  than as a correct final answer.
- The summary threshold of 0.70 is consistent with the current V2 reports.
