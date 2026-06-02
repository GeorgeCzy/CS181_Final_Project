# TF-IDF Logistic Regression Baseline

Run date: 2026-05-18

Weights & Biases:

```text
https://wandb.ai/chengzy2023-shanghaitech-university/mode-classifier/runs/i7chueur
```

Input dataset:

```text
data_generation/data/raw/deepseek_generated_2000.csv
```

Split:

| Split | Rows | chat | motion_query |
| --- | ---: | ---: | ---: |
| Train | 1600 | 800 | 800 |
| Validation | 200 | 100 | 100 |
| Test | 200 | 100 | 100 |

Model:

```text
TfidfVectorizer(1-2 grams, max_features=5000)
+ LogisticRegression(class_weight="balanced")
```

Results:

| Split | Accuracy |
| --- | ---: |
| Train | 0.9950 |
| Validation | 0.9750 |
| Test | 0.9900 |

Validation confusion matrix:

| gold \ pred | chat | motion_query |
| --- | ---: | ---: |
| chat | 99 | 1 |
| motion_query | 4 | 96 |

Test confusion matrix:

| gold \ pred | chat | motion_query |
| --- | ---: | ---: |
| chat | 99 | 1 |
| motion_query | 1 | 99 |

Observed errors:

| Split | ID | Gold | Predicted | Utterance |
| --- | --- | --- | --- | --- |
| Validation | deepseek2000-01064 | motion_query | chat | Stay where you are. |
| Validation | deepseek2000-00607 | chat | motion_query | Who wrote 'Pride and Prejudice'? |
| Validation | deepseek2000-00631 | motion_query | chat | Can you show me what you can do? |
| Validation | deepseek2000-00582 | motion_query | chat | Stop what you are doing. |
| Validation | deepseek2000-00312 | motion_query | chat | Sign the word 'thank you' in sign language. |
| Test | deepseek2000-00048 | motion_query | chat | Can you guide me through this maze? |
| Test | deepseek2000-01409 | chat | motion_query | Who invented the telephone? |

Notes:

- The baseline is already strong, likely because many motion queries contain clear lexical cues.
- The observed errors are useful hard cases for later evaluation and prompt refinement.
- The next model should evaluate whether embeddings improve boundary cases rather than only headline accuracy.
