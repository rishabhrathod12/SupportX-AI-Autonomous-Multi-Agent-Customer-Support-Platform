# Evaluation Report

Date: 2026-11-18. Every number below comes from a run recorded in `docs/EXPERIMENTS.md`, MLflow, or the
JSON reports in `artifacts/models/` (git-ignored; model files are identified by SHA-256).

## 1. What was built
A multi-agent support ticket workflow: a rule-based Supervisor routes each ticket through intent, sentiment,
policy retrieval (RAG), order lookup and refund agents (LangGraph), replies from grounded templates, and pauses
for a HUMAN decision before any refund is approved. Served by FastAPI over SQLAlchemy (PostgreSQL in the
compose file, SQLite in tests); paused tickets persist in a SQLite checkpoint file; Alembic migrations; request-id
logging. The LLM reply composer is built but untested against the real API; the sentiment model is not trained (see section 8).

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
- Automated tests: 462 passed, 2 skipped (the PostgreSQL race tests), 0 failed in a full run on 2026-11-25
  (about 2-10 min). ruff and `mypy --strict` clean.
- Real-server runs: ticket submit, human approve, restart with a paused ticket, and double-approve (409) were
  exercised against real `uvicorn` with SQLite (Phase 8.1 and 7.2b).
- Alembic: migrations upgrade, match the models (no drift) and downgrade on SQLite only.
- Retrieval: see E8/E9 in `docs/EXPERIMENTS.md` (hand-written questions; lexical retriever; cutoff is a trade-off).

## 5. End-to-end workflow evaluation (Phase 9, DATA ORIGIN: SYNTHETIC)
`python -m supportx.ml.agent_eval <model>` runs 17 hand-written scenarios (`data/sample/agent_scenarios.csv`,
expectations fixed BEFORE the first run) through the real graph, rule engines, retriever and database services
(fresh in-memory SQLite per scenario, served model `intent_tfidf_lr_bitext_aug.joblib`). Human decisions are simulated.

| Category | Passed | What it checks |
|---|---|---|
| info | 2/2 | order status and policy answers complete without a human |
| refund | 5/5 | eligible -> pauses, approve/reject recorded; outside window, wrong email, no order number -> no request created |
| cancel | 3/3 | unshipped -> pauses, approve cancels, reject leaves the order; delivered order -> denied, unchanged |
| escalation | 3/3 | human requested and gibberish escalate; "Where can I buy a train ticket?" (s13) is handed over because retrieval at cutoff 0.15 finds no matching policy, NOT because the intent model recognises it as out of scope (it was 2/3 at the earlier cutoff) |
| adversarial | 4/4 | prompt-injection / fake-authority / SQL-looking messages never approve a refund or cancel an order |

Total 17/17 on 2026-11-25 (16/17 before the cutoff change and query rewriting; the evaluator uses the serving retriever settings). SAFETY INVARIANTS: 0 violations (no refund approved and no order cancelled without a human
approval; completed tickets always have a reply). The evaluator itself is tested: it fails a wrong expectation
and flags a deliberately broken "auto-approve" workflow. One checker bug (flagging paused tickets as "no reply")
was found and fixed in the checker on the first run; no expectation was changed to fit a result.
Limits: 17 scenarios written by the system's author; the intent step uses the real model but human decisions,
customers and orders are simulated. It shows the workflow behaves as designed, not how customers fare.

## 6. Confidence and abstention (intent threshold), dev probes only
`supportx.ml.probe_eval` on the 80 in-scope + 10 out-of-scope DEV probes (the holdout was spent in E7; these dev
probes also guided the augmentation, so the numbers are optimistic):

| Threshold | Handled automatically (right) | Escalated | Confidently wrong | Out-of-scope escalated |
|---|---|---|---|---|
| 0.4 | 72 | 6 | 4 | 8/10 |
| 0.5 | 66 | 12 | 4 | 8/10 |
| **0.6 (current)** | 57 | 21 | 3 | 9/10 |
| 0.7 | 50 | 28 | 3 | 9/10 |
| 0.8 | 45 | 34 | 2 | 9/10 |
| 0.9 | 35 | 44 | 2 | 9/10 |

Raising the threshold removes few confident mistakes (4 -> 2) but sends many correct tickets to humans (72 -> 35
handled). The threshold is therefore a staffing decision, not a safety lever: safety comes from the human approval
step. Refund and cancellation requests always need a person, so a wrong intent costs a pending request, not money.
Retrieval abstention is separate (`RETRIEVAL_MIN_SCORE`, see E9).

## 7. Synthetic versus real-world evidence
| Evidence | Origin | Real-customer? |
|---|---|---|
| Bitext intent scores | Synthetic, templated | No |
| Banking77 intent scores | Public banking queries | Closer, wrong domain |
| Intent probes, retrieval questions, workflow scenarios | Hand-written by the developer | No |
| Sentiment, priority, team, resolution time | Nothing trained or scored | n/a |
| Semantic retrieval, LLM replies | Built, tested with fakes only | n/a |
No evaluation on real customer tickets exists. Do not read any number above as production performance.

## 8. Not done / not verified (read before relying on this)
- Never run against PostgreSQL (not installed): migrations, row locking; 2 race tests are skipped. Docker image never built.
- Sentiment is a keyword rule; priority/team are rules; resolution time is unknown; no model for these is trained.
- No OpenAI key: the LLM reply composer and semantic embeddings are tested only with a fake SDK.
- Intent data is synthetic or banking; the intent model itself still does not recognise out-of-scope messages (s13 is caught by retrieval abstention only).
- Refund policy numbers are demo defaults, not confirmed policy.
- Security: see `docs/SECURITY.md` (HTTPS not configured, customers verified only by order number + email, rate
  limits per process).
- Cancellation has no payment reversal, stock release or customer notification (mock adapters only).
- No metrics endpoint or tracing.
