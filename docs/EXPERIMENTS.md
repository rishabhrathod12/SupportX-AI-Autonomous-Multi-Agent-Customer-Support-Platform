# Experiment Log

Every number here comes from a real run (also logged in MLflow). Scores are on the VALIDATION split.
The test splits have not been used yet; they are reserved for the final model comparison.

## How to reproduce and inspect
```powershell
$env:PYTHONPATH = "src"
python -m supportx.ml.train_intent bitext
python -m supportx.ml.train_intent banking77
mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
```

## Intent classification

### E1 - TF-IDF (1-2 grams, min_df=2) + Logistic Regression (C=10, class_weight=balanced), seed 42
Run date: 2026-11-15. Code: `supportx.ml.train_intent`. Model files: `artifacts/models/` (git-ignored).

| Dataset | Data type | Train / Val | Val accuracy | Val macro-F1 | Majority-class baseline (acc / macro-F1) |
|---|---|---|---|---|---|
| Bitext | **Synthetic** | 19,409 / 2,427 | 0.993 | 0.992 | 0.041 / 0.003 |
| Banking77 | Not stated on card | 8,992 / 1,000 | 0.886 | 0.883 | 0.019 / 0.0005 |

Weakest validation intents (F1):
- Bitext: cancel_order 0.953, delete_account 0.967, create_account 0.971, track_order 0.981, change_order 0.988
- Banking77: pending_transfer 0.629, unable_to_verify_identity 0.706, why_verify_identity 0.720, card_not_working 0.737, pin_blocked 0.762

How to read this:
- **Bitext 0.99 is close to a ceiling and says little about real customers.** The data is synthetic and templated. The near-duplicate leakage hypothesis was tested in E2 below and mostly rejected.
- **Banking77 is the more informative number.** Confusions are between semantically close intents (identity verification, transfers), which is realistic.
- Hyperparameters (C, n-grams) were not tuned in E1; this is a first baseline. Tuning must use validation, never test.
- The Banking77 validation set is small (1,000), so differences of about 1 point are within noise.

### E2 - Near-duplicate leakage check (validation vs train, E1 model)
Run date: 2026-11-15. Helper: `supportx.data.similarity.nearest_similarity` (TF-IDF cosine similarity of each
validation text to its closest training text). The test splits were not used.

| Dataset | Val texts with a close twin in train (sim >= 0.90) | Accuracy, sim >= 0.90 | Accuracy, sim < 0.90 | Accuracy, sim < 0.70 |
|---|---|---|---|---|
| Bitext (synthetic) | 14.1% (342 of 2,427) | 1.000 | 0.992 (n=2,085) | 0.981 (n=690) |
| Banking77 | 3.0% (30 of 1,000) | 1.000 | 0.883 (n=970) | 0.861 (n=767) |

Conclusion: near-duplicates are not what drives Bitext's 0.99. Messages with no close twin score 0.992, so
Bitext is simply easy (intents separable by keywords). Median similarity is higher on Bitext (0.78 vs 0.56),
confirming it is more templated. Limit: this measures shared wording, not paraphrases with different words.
