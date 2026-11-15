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

### E3 - Classical model comparison (validation split, seed 42, same TF-IDF features)
Run date: 2026-11-15. `python -m supportx.ml.train_intent <dataset> --model logreg|xgboost [--c N]`.
XGBoost: 200 trees, max_depth 6, learning_rate 0.2, subsample 0.8, colsample_bytree 0.5 (not tuned).
Train times are wall-clock seconds on this slow, CPU-only Windows machine (other programs were also running).

| Model | Bitext acc / macro-F1 | Bitext train s | Banking77 acc / macro-F1 | Banking77 train s |
|---|---|---|---|---|
| LogReg C=1 | 0.9905 / 0.9896 | 4.3 | 0.870 / 0.866 | 7.9 |
| LogReg C=10 (E1) | 0.993 / 0.992 | 6.0 | 0.886 / 0.883 | 11.7 |
| LogReg C=100 | 0.9909 / 0.990 | 4.2 | 0.892 / 0.890 | 12.4 |
| XGBoost, 200 trees | 0.9827 / 0.9815 | 319.6 | 0.823 / 0.816 | 708.4 |

Weakest XGBoost intents (F1): Bitext create_account 0.935, delete_account 0.935, cancel_order 0.943;
Banking77 card_not_working 0.500, topping_up_by_card 0.556, contactless_not_working 0.600.

Conclusions:
- **Logistic Regression beats XGBoost on both datasets and trains about 50x (Bitext) to 60x (Banking77) faster.**
  Gradient-boosted trees fit high-dimensional sparse text features poorly. XGBoost was not tuned, so this describes
  these settings, not XGBoost in general.
- **C matters little on Bitext (spread 0.3 points) and moderately on Banking77.** C=100 is 0.6 points above C=10 on
  Banking77, but with 1,000 validation messages one standard error is about 1 point, so that gap is within noise.
  We keep C=10 as the reference and do not claim C=100 is better.
- Reference classical model going forward: TF-IDF + LogReg (C=10). The test splits remain untouched.

### E4 - Deep learning: Embedding + BiLSTM vs the classical reference (validation split, seed 42)
Run date: 2026-11-15. `python -m supportx.dl.train_intent <dataset>`. One configuration only, not tuned:
embedding 64, BiLSTM 64 units, dropout 0.3, Adam lr 3e-3, batch 64, balanced class weights, max 30 epochs with
early stopping (patience 3, best-validation-loss weights restored). Vocabulary: words seen at least twice in train
(Bitext 1,020 words, max length 14; Banking77 1,420 words, max length 42). Embeddings are learned from scratch.

| Model | Bitext acc / macro-F1 | Bitext train s | Banking77 acc / macro-F1 | Banking77 train s |
|---|---|---|---|---|
| TF-IDF + LogReg C=10 (E1) | 0.993 / 0.992 | 6.0 | 0.886 / 0.883 | 11.7 |
| Embedding + BiLSTM | 0.992 / 0.991 | 133.1 (9 epochs) | 0.851 / 0.847 | 178.8 (11 epochs) |

Weakest BiLSTM intents (F1): Bitext cancel_order 0.953, delete_account 0.972, create_account 0.977;
Banking77 topping_up_by_card 0.522, transfer_not_received_by_recipient 0.562, card_acceptance 0.571.

Conclusions:
- **Bitext: a tie.** 0.992 vs 0.993 is far inside the noise (about 0.2 points on 2,427 messages), at about 20x
  the training time. Bitext is too easy to separate the models.
- **Banking77: the BiLSTM is worse by about 3.5 points** (0.851 vs 0.886; one standard error is about 1.1 points),
  and about 15x slower to train. This is one run with one seed and no paired significance test, so treat the size of the
  gap as approximate.
- At the chosen stopping point the BiLSTM fit the Banking77 training data much better than validation data
  (training accuracy 0.958 vs validation 0.856), i.e. it overfits.
- Hypotheses, NOT tested: (1) learning embeddings from scratch on about 9k messages is too little data, so
  pretrained embeddings or a pretrained language model (the Hugging Face part we deferred) may close the gap;
  (2) tuning dropout, size or learning rate could help. Neither is verified.
- Decision for now: the classical TF-IDF + LogReg model remains the best and cheapest intent model. The BiLSTM stays as a
  documented experiment; it is not worth its cost on this evidence. Test splits are still untouched.

### E5 - Does the intent model's confidence tell us when to hand over to a human?
Run date: 2026-11-15. Model: TF-IDF + LogReg C=10 trained on Bitext (the intents the agents route on).
Measured on the Bitext VALIDATION split (n=2,427; overall accuracy 0.993). Test splits untouched.

| Confidence | Share of messages | Accuracy | n |
|---|---|---|---|
| < 0.2 | 0.0% | 0.000 | 1 |
| 0.2 - 0.4 | 0.3% | 0.857 | 7 |
| 0.4 - 0.6 | 0.9% | 0.773 | 22 |
| 0.6 - 0.8 | 2.5% | 0.918 | 61 |
| >= 0.8 | 96.3% | 0.998 | 2,336 |

- On in-domain data the model is confident almost always (96% at 0.8 or more, 99.8% accurate), so a threshold rarely
  fires: at the old 0.4 it escalated 0.3% of messages and caught only 2 of the 17 wrong answers. At 0.6 it would
  escalate 1.2% (30 messages) of in-domain validation messages.
- Out-of-domain probe: the 1,000 Banking77 validation messages (banking, not e-commerce, so the model has no valid
  answer for them): median confidence 0.275; 66.7% fall below 0.4, 83.0% below 0.6, and 17.0% are still answered at
  0.6 or higher, i.e. confidently wrong. A threshold cannot remove those.
- Hand-written probes (10 messages, not a benchmark): junk ("hello", "asdf qwerty", "blah blah blah") scores 0.06;
  "What's the weather like today?" 0.20; "I lost my card and need a new one" 0.41 (just above the old limit, wrongly
  routed to create_account); "Where is my order? It was supposed to arrive yesterday" went to delivery_period (a policy
  answer, 0.67) instead of track_order.
- Decision: the minimum confidence is now the setting `INTENT_MIN_CONFIDENCE`, default 0.6 (was a fixed 0.4):
  escalating a ticket is cheap and a confident wrong answer is expensive. This is a judgment call from the numbers above,
  not an optimum. Caveat: the model was trained on synthetic, templated text, so real customers will behave differently;
  re-measure on real messages when they exist.
