# Experiment Log

Every number here comes from a real run (also logged in MLflow). Scores are on the VALIDATION split.
The test split was used exactly once, for the final reference model (E6). Everything else is validation.

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

E5b - first end-to-end run with the REAL trained model (real server, 2026-11-15, 4 hand-written messages, not a benchmark):
- "What is your return policy?" -> answered from the refund policy passage.
- "My kettle arrived broken and I want my money back, order ORD-5001" -> get_refund, paused for a human (correct).
- "Where is my order ORD-5001?" -> escalated as low_confidence (a handoff), although the predicted intent was the right
  one (track_order) at confidence 0.26; the bare "Where is my order?" scores the same 0.26.
- "asdf qwerty" -> escalated as low_confidence (correct).
- Why: the training text carries a placeholder token where an order number goes, and the plain phrasing is rare in the
  synthetic data. Replacing real order numbers with that placeholder before predicting is NOT a clean fix on 10 probes:
  tracking rises 0.26 -> 0.75 but the refund message falls 0.92 -> 0.51. Not adopted.
- Consequence: with this model and the 0.6 threshold, many ordinary messages go to a human. That is the safe direction
  but not yet useful. The fix is better data (realistic, varied phrasing, ideally real tickets) and a proper held-out
  evaluation of confidence, not more threshold tweaking. The unit tests use a fake intent model, so they cannot show this.

### E6 - Final held-out test scores for the reference model (TEST split, run once)
Run date: 2026-11-18. `python -m supportx.ml.final_test <dataset>` (refuses to run twice). Model: TF-IDF + LogReg C=10,
chosen on validation in E1-E4. Reports: `artifacts/models/intent_tfidf_lr_<dataset>.test_report.json`.

| Dataset | n test | Accuracy | Macro-F1 | Majority baseline acc |
|---|---|---|---|---|
| Bitext (synthetic) | 2,427 | 0.993 | 0.993 | 0.041 |
| Banking77 | 3,080 | 0.897 | 0.897 | 0.013 |

Matches validation (0.993 / 0.886), so no sign of overfitting to validation. Full discussion: `docs/EVALUATION.md`.

### E7 - Realistic training phrasings and order-number normalization (probe set, validation only)
Run date: 2026-11-18. Motivation: E5b (ordinary messages escalate). Test split NOT used for this model.
Probe set: `data/sample/intent_probe.csv`, 135 hand-written messages written and committed BEFORE any change:
`dev` (80 in-scope across 10 intents + 10 out-of-scope) for decisions, `holdout` (36 in-scope + 5 out-of-scope)
scored once at the end. Metric (`supportx.ml.probe_eval`, threshold 0.6): *useful* = right intent AND confident
(handled automatically), *escalated* = low confidence (a human takes over), *confidently wrong* = wrong at/above threshold.

| Change (all LogReg C=10, Bitext train) | Dev useful / 80 | Escalated | Confidently wrong | Bitext val acc |
|---|---|---|---|---|
| Baseline (reference model E1) | 31 | 44 | 6 | 0.993 |
| + real order numbers (ORD-123) mapped to the dataset's order-number token | 31 | 44 | 6 | 0.993 |
| + hand-written phrasings, 60 per intent (540 rows) | 55 | 22 | 4 | 0.993 |
| + 120 per intent (1,080 rows) **chosen** | 57 | 21 | 3 | 0.992 |
| + 250 per intent (2,250 rows) | 62 | 16 | 4 | 0.993 |

- Order-number mapping alone changed nothing in total (tracking and cancelling improved, the two refund intents got
  worse) - it is kept because it makes train and serve text consistent, not because it helps by itself.
- The phrasings (`supportx.data.augment`: 9 intents x ~16 hand-written cores, expanded with openers, closers, order
  references, typos; train split only; probe texts removed) are what helped. 120 was chosen over 250 because the
  4-5 message difference is within noise on 80 messages and less synthetic data means less fitting to my own wording.
- HOLDOUT, scored once (36 in-scope, 5 out-of-scope):

| Model | Useful | Escalated | Confidently wrong | Out-of-scope escalated |
|---|---|---|---|---|
| Reference (old tokenization) | 17 (47%) | 16 | 5 | 3 / 5 |
| Augmented (120/intent) | 27 (75%) | 7 | 3 | 4 / 5 |

- Remaining confident mistakes (holdout): "Can I get a refund after 30 days?" -> get_refund 0.94 (a policy question treated
  as a refund request); "when do I get my money back for the return I already sent" -> get_refund 0.84 (track_refund);
  "Where can I buy a train ticket?" -> place_order 0.86. Dev: "how do I reset my bank PIN" -> recover_password 0.93.
  Refunds and cancellations still need a human decision, so a wrong route costs a pending request, not money.
- BIG CAVEAT: I wrote the training phrasings AND the probes, so these probe scores are optimistic about real customers.
  The Bitext validation score did not regress (0.9918 vs 0.9930, noise). The Bitext test split was not scored for the
  augmented model. Real customer messages remain the proper test; the model is now the serving default
  (`artifacts/models/intent_tfidf_lr_bitext_aug.joblib`, reference file kept).
- End-to-end with the real model and a seeded order: "Where is my order ORD-5001?" is now answered (was escalated,
  E5b); "Could you give me a tracking update for order ORD-5001 please" still escalates (low confidence).

### E8 - Retrieval: stemming, heading weight, and quoting a close runner-up (hand-written questions)
Run date: 2026-11-18. Motivation: the live demo answered "What is your return policy?" with the "Changed your mind"
passage, and E5b/the 6.1 check showed 'cancel' not matching 'cancelling'. Questions: `data/sample/retrieval_eval.csv`
(32 hand-written, committed BEFORE any change): `dev` 20 for decisions, `holdout` 12 scored once. Each has one right
section (file + heading). Scorer: `python -m supportx.rag.evaluate`. Retriever is still the LEXICAL TF-IDF one.

Dev (n=20; one question = 5 points, so differences of 1-2 questions are noise):

| Variant | top-1 | top-3 |
|---|---|---|
| Baseline (1.0x heading, no stemming) | 0.75 | 0.90 |
| + stemming (`light_stem`) | 0.80 | 0.85 |
| + stop-word removal | 0.55 | 0.90 |
| stemming + stop words | 0.60 | 0.90 |
| stemming + heading repeated 2x **chosen** | 0.80 | 0.95 |
| stemming + heading 3x / 4x | 0.75 | 0.95 / 0.90 |

- Stop-word removal HURT (words like "can", "after", "not" carry meaning in 16 tiny passages), so it stays off.
- Reply change: the Knowledge Agent returns the best passage plus the runner-up when it scores at least half as high
  (`Retriever.retrieve_for_answer`), and the reply quotes both. On dev the right section is shown 85% of the time
  (top-1 alone: 80%), with 1.5 passages quoted on average. Cutoffs 0.3-0.5 gave the same coverage; 0.5 kept.
- Holdout (n=12, scored once), right section SHOWN to the customer: old configuration 0.58 (top-1 0.50) -> new 0.83
  (top-1 0.67). With 12 questions one standard error is about 11 points, so treat this as "better", not a size.
- Still wrong, and visible in the demo: "What is your return policy?" retrieves "Changed your mind" first (0.23); the right
  section ("Return window") is not shown, and the reply's runner-up is an unrelated tracking passage because it cleared the
  half-score cutoff. Cost of the runner-up rule: sometimes one irrelevant paragraph. A stricter cutoff (0.7) avoids more of
  that but showed the right section for one fewer dev question (0.80 vs 0.85), so 0.5 was kept; this is a judgment call.
- Not fixed: "I want to cancel an order that is already on its way" and "I put the wrong address. Can it still be fixed?"
  still retrieve the wrong passage (lexical matching cannot connect "on its way" to "shipped"); "answered" is 100%
  because the 0.1 minimum score never rejected any of these on-topic questions (no off-topic questions were scored here;
  a gibberish query is rejected, see tests). A semantic embedding model is the real fix (needs an API key or PyTorch).

### E9 - Retrieval ranking metrics, hybrid lexical rankers, and when to hand over (hand-written questions)
Run date: 2026-11-19. Questions: `data/sample/retrieval_eval_v2.csv`, written and committed before any change: `dev` 21
(15 answerable, 6 the documents CANNOT answer) and `holdout` 20 (14 answerable, 6 unanswerable), scored once. The 32
questions from E8 count as extra DEV material (their holdout was already spent). A question may have several relevant
sections. Code: `supportx.rag.metrics` (Recall@K, Precision@K, Hit@K, MRR; for unanswerable questions the false-answer
rate), `supportx.rag.hybrid` (BM25 and character n-gram rankers fused by reciprocal rank fusion). Everything is LEXICAL:
no semantic ranker exists yet (needs an OpenAI key or PyTorch); the hybrid class has a slot for one.

Dev (21 + 32 = 53 questions, 47 answerable):

| Ranker | Recall@1 | Recall@3 | Precision@3 | MRR |
|---|---|---|---|---|
| Word TF-IDF (serving baseline) | 0.699 | 0.904 | 0.326 | 0.823 |
| + BM25 | 0.688 | 0.872 | 0.319 | 0.811 |
| + character n-grams | 0.688 | 0.936 | 0.340 | 0.827 |
| + BM25 + character n-grams | 0.730 | 0.936 | 0.340 | 0.849 |

(Precision@3 cannot exceed |relevant|/3, so it is low for any single-section question.) One question is about 2 points,
so the dev differences are 1-2 questions: noise.

HOLDOUT (14 answerable, scored once; the adoption rule was fixed beforehand: adopt the hybrid only if it also wins here):

| Ranker | Recall@1 | Recall@3 | Precision@3 | MRR |
|---|---|---|---|---|
| Word TF-IDF baseline | 0.714 | 0.786 | 0.262 | 0.780 |
| Hybrid (BM25 + char n-grams) | 0.643 | 0.786 | 0.262 | 0.756 |

**The hybrid did not win on the holdout, so it is NOT used in serving** (the code stays as an extension point for a future
semantic ranker). Adding more lexical rankers to a 16-passage corpus does not help.

**The important finding: the system almost never hands over.** With the old fixed cutoff (0.1) every unanswerable question is
answered with some loosely related passage (false-answer rate 1.00 on dev and holdout). The cutoff can be raised, but lexical
scores overlap too much for a clean line: unanswerable top scores reach 0.22 while answerable ones have a median of 0.26 and
go down to 0.11. The trade-off on dev (baseline ranker; the hybrid was within 0.05 of it everywhere):

| Cutoff | Real questions answered | Answered AND right | Unanswerable still answered (of 6) |
|---|---|---|---|
| 0.10 (old default) | 1.00 | 0.72 | 1.00 |
| 0.15 | 0.89 | 0.70 | 0.67 |
| 0.18 | 0.81 | 0.66 | 0.50 |
| 0.20 | 0.70 | 0.55 | 0.17 |
| 0.22 | 0.62 | 0.51 | 0.00 |
| 0.25 | 0.53 | 0.49 | 0.00 |
| 0.30 | 0.32 | 0.32 | 0.00 |

Holdout, baseline: cutoff 0.10 answers 100% of real questions (71% right), false answers 100%; cutoff 0.20 answers 79% (64%
right), false answers 17%. The cutoff is now a setting, `RETRIEVAL_MIN_SCORE`, **default was 0.1 here; the serving default was then set by the project owner on 2026-11-25: first 0.2, then 0.15 after a demo showed 0.2 handed over an answerable delivery-options question**; choosing is a
product decision (staff time against quoting an irrelevant passage). Around 0.20 looks like the balanced region on this data.

The five example questions today (top passages, baseline): "Can I cancel my order?" gets all three cancellation sections;
"My package has not arrived" gets the two right sections; "Can I return a damaged product?" gets the right section;
"What is your return policy?" is still wrong (right section "Return window" is third, 0.13); "How long does a refund take?"
gets "How refunds are approved" (0.19), which says nothing about duration (the documents have no answer) and would be
handed over at a cutoff of 0.20 or higher. Not built: semantic search; a metadata filter exists (`Retriever.retrieve(sources=...)`)
but is not wired to the intent. Evidence is 53 + 20 hand-written questions over 3 demo documents: it shows direction, not production quality.
