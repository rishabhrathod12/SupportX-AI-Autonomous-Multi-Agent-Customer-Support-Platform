# Evaluation Report

Date: 2026-11-18. Every number below comes from a run recorded in `docs/EXPERIMENTS.md`, MLflow, or the
JSON reports in `artifacts/models/` (git-ignored; model files are identified by SHA-256).

## 1. What was built
A multi-agent support ticket workflow: a rule-based Supervisor routes each ticket through intent, sentiment,
policy retrieval (RAG), order lookup and refund agents (LangGraph), replies from grounded templates, and pauses
for a HUMAN decision before any refund is approved. Served by FastAPI over SQLAlchemy (PostgreSQL in the
compose file, SQLite in tests); paused tickets persist in a SQLite checkpoint file; Alembic migrations; request-id
logging. The LLM response agent and sentiment model are NOT built (see section 5).

## 2. Intent model: final held-out result
Model: TF-IDF (1-2 grams) + Logistic Regression, C=10, balanced class weights, seed 42. It was chosen on VALIDATION
scores only (E1-E4) before the test split was read. The test split was then scored exactly once per dataset
(`python -m supportx.ml.final_test`, which refuses to re-run). No tuning followed.

| Dataset | Data | n test | Test accuracy | Test macro-F1 | Validation acc / macro-F1 | Majority baseline acc |
|---|---|---|---|---|---|---|
| Bitext | Synthetic | 2,427 | 0.993 | 0.993 | 0.993 / 0.992 | 0.041 |
| Banking77 | Not stated on card | 3,080 | 0.897 | 0.897 | 0.886 / 0.883 | 0.013 |

Weakest test intents (F1): Bitext get_invoice 0.977, check_invoice 0.978, create_account 0.983;
Banking77 balance_not_updated_after_bank_transfer 0.747, verify_my_identity 0.773, topping_up_by_card 0.784.

Reading it:
- Test and validation agree (differences of 0.1 and 1.1 points; one standard error on Banking77 test is about
  0.55 points), so there is no sign the choice overfit validation.
- Bitext 0.993 is a ceiling effect of synthetic, templated text (E2), not evidence about real customers.
- Banking77 (0.897) is the more honest figure, and it is a banking domain, not e-commerce.
- Known caveat: 7 Banking77 train/test text overlaps were found during data preparation (`docs/DATASETS.md`);
  7 of 3,080 cannot move accuracy by more than about 0.2 points.
- Model comparison (validation only): LogReg beat XGBoost (50-60x faster) and an Embedding+BiLSTM
  (tie on Bitext, 3.5 points worse on Banking77, 15-20x slower). These two were not scored on test, by design.

## 3. Does the confidence threshold work?
Only partly (E5/E5b). In-domain the model is confident 96% of the time, so the threshold rarely fires. Out of
domain, 17% of Banking77 messages are still answered at confidence >= 0.6, i.e. confidently wrong. Conversely,
"Where is my order ORD-5001?" is routed to a human (right intent, confidence 0.26). With the real model many
ordinary messages therefore go to a human: safe, but not yet useful. This was not fixed by the threshold because the cause is
training data (synthetic, templated). Follow-up (E7): hand-written realistic phrasings added to training lifted
the share of in-scope probe messages handled automatically from 47% to 75% on a holdout of 36 hand-written messages
(confidently wrong 5 -> 3), with no loss on Bitext validation. Both the phrasings and the probes were written by the
same person, so treat this as optimistic; the augmented model's Bitext TEST score was not measured.

## 4. System checks
- Automated tests: 261 passed in a full run at Phase 7.2b (about 21 min, mostly TensorFlow); tests added since
  (migrations, observability, final-test script) passed individually, and the whole suite was not re-run after them.
  ruff and `mypy --strict` clean.
- Real-server runs: ticket submit, human approve, restart with a paused ticket, and double-approve (409) were
  exercised against real `uvicorn` with SQLite (Phase 8.1 and 7.2b).
- Alembic: migration upgrades, matches the models (no drift) and downgrades on SQLite.
- Retrieval: 6 hand-written questions (right document in the top 3 for 6/6, first for 5/6). Too few to measure quality.

## 5. Not done / not verified (read before relying on this)
- Never run against PostgreSQL (not installed): migration, row locking for concurrent refund requests.
- Docker image and compose file were never built or run (Docker not installed).
- No sentiment, priority, routing or resolution-time models; sentiment is a keyword stub.
- No LLM or semantic embeddings (no API key; the retriever is lexical TF-IDF). Replies are templates quoting the top passage.
- Intent data is synthetic or banking; nothing was evaluated on real customer tickets.
- Refund policy numbers (30-day window, 14 days "not received") are demo defaults, not a real policy.
- Security: one shared staff key (approver name is self-declared), customers identified only by order number
  plus email, no rate limiting, no HTTPS.
- Cancellation is rule-checked, human-approved and recorded on the order, but there is no payment reversal, stock release
  or customer notification (no such systems exist here).
- No metrics endpoint or tracing.
