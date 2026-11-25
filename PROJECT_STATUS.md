# Project Status

Last updated: 2026-11-18 | Current phase: intent data improvement (E7) done, committed; Docker files untested. Remaining items need user input

## Completed
- [x] Phase 0: workspace inspected, architecture, docs, folder skeleton, requirement files
- [x] Git repository initialized; `origin` set to the GitHub repo (empty, nothing pushed)
- [x] Phase 1.1: dependencies installed from `requirements-dev.txt` into the existing `venv`
      (`pip install --no-cache-dir`, exit code 0; venv was not recreated)
- [x] `pip check`: "No broken requirements found."
- [x] All 29 key packages import successfully; versions match the pins in `requirements.txt`
- [x] PyTorch and transformers confirmed NOT installed (deferred by design)

## Verified environment
- Python 3.12.10 (64-bit, MSC v.1943), pip 25.0.1, venv at `./venv`
- numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0
- tensorflow 2.21.0, keras 3.15.1 (CPU only: devices = ['CPU'])
- nltk 3.10.4, openai 3.50.0, langchain 1.5.0, langchain-openai 1.7.4,
  langchain-community 0.4.2, langgraph 1.2.21, faiss-cpu 1.16.0, tenacity 9.2.1
- fastapi 0.162.6, uvicorn 0.54.0, pydantic 2.14.3, pydantic-settings 2.16.3,
  sqlalchemy 2.1.8, alembic 1.20.1, psycopg 3.3.9 (binary libpq loads), httpx 0.28.1,
  python-dotenv 1.2.4, mlflow 3.18.0
- pytest 9.2.0, ruff 0.17.5, mypy 2.4.1 (pytest-cov 7.1.0 also installed)

- [x] Phase 1.2: `core` package (`config.py`, `exceptions.py`, `logging_config.py`) with 11 unit tests
      (11 passed, 96% coverage of `supportx.core`; ruff and mypy --strict clean)

- [x] Phase 1.3: first commit `729a0c5` (36 files) on branch `main`, pushed to GitHub
      (`origin/main`, upstream tracking set). No secrets, venv, data or models committed.

- [x] Phase 2.1: datasets chosen and documented in `docs/DATASETS.md` (nothing downloaded yet):
      Bitext (intent, synthetic), Banking77 (optional benchmark), Amazon Reviews 2023 subsample
      (sentiment, weak star labels). Priority/routing/resolution-time data deferred.

- [x] Phase 2.2: `supportx.data` package - license-gated downloader (`download.py`, pinned revisions,
      SHA-256 manifest) and synthetic sample generator (`sample.py`); 21 tests pass in total.
      Datasets are NOT downloaded yet (waiting on the license-reading items below).

- [x] Phase 2.3a: `supportx.nlp.preprocessing` (`clean_text`, `tokenize`, `clean_series`), 11 tests,
      100% coverage; found and fixed a case-sensitivity bug in URL/email matching via the idempotency test.
      Not committed yet.

- [x] Phase 2.3b: user confirmed the licenses were read; Bitext and Banking77 downloaded to `data/raw/`
      (git-ignored; hashes in MANIFEST.json files).
- [x] Phase 2.3c: first look at the data, findings in `docs/DATASETS.md` (duplicates, placeholders,
      class balance, 7 Banking77 train/test overlaps).

- [x] Phase 2.3 committed and pushed (`2c65e11`).
- [x] Phase 2.4: `clean_text` maps `{{Slot Name}}` to `<slot_name>`; `supportx.data.prepare` builds
      deduplicated, stratified, leak-checked train/val/test files in `data/processed/` (git-ignored).
      Bitext 19,409/2,427/2,427; Banking77 8,992/1,000/3,080 (official test). 44 tests pass;
      ruff and mypy --strict clean. `pyarrow` pinned in requirements.txt (was only an MLflow dependency).

- [x] Phase 2.4 committed and pushed (`d091bf0`).
- [x] Phase 3.1: intent baseline (TF-IDF + Logistic Regression): `supportx.ml.intent`
      (`IntentClassifier`: fit/predict/predict_proba/save/load), `supportx.ml.evaluate`,
      `supportx.ml.train_intent` with MLflow tracking (SQLite backend) and a majority-class sanity baseline.
      Validation results (test untouched): Bitext (synthetic) acc 0.993 / macro-F1 0.992;
      Banking77 acc 0.886 / macro-F1 0.883. Details in `docs/EXPERIMENTS.md`.
      55 tests pass; ruff and mypy --strict clean.
- [x] MLflow default changed to `sqlite:///mlflow.db` (MLflow 3.18 refuses the `./mlruns` file store).

- [x] Phase 3.1 committed and pushed (`94acb18`).
- [x] Phase 3.2a: Bitext near-duplicate leakage check (`supportx.data.similarity`): only 14% of validation texts
      have a close twin in train; accuracy without a twin is still 0.992, so Bitext is just easy, not leaked.
      Results in `docs/EXPERIMENTS.md` (E2).
- [x] Phase 3.2b (code): `IntentClassifier` supports `model="logreg"` or `"xgboost"` (label encoding, so string
      labels work), `train_intent` has `--model`, `--n-estimators`, `--models-root` and logs `train_seconds`.
      Tests added; ruff and mypy --strict clean. Old saved model files from 3.1 are in the previous format and were
      re-created by the comparison run.
- [x] Phase 3.2b (results): LogReg C sweep and XGBoost comparison finished (E3 in `docs/EXPERIMENTS.md`):
      LogReg C=10 is the reference; XGBoost (untuned) is worse (Bitext 0.983, Banking77 0.823) and ~50-60x slower.
- [x] Phase 4.1: `supportx.dl.intent.KerasIntentClassifier` (Embedding + BiLSTM, early stopping, class weights,
      save/load) and `supportx.dl.train_intent`; shared helpers `supportx.data.splits` and `supportx.ml.experiment`;
      `ml.train_intent` refactored to use them. First runs (E4): Bitext 0.992 (tie with LogReg), Banking77 0.851 vs
      LogReg 0.886 (BiLSTM worse, overfits). Classical model stays the reference.
- [x] Full test suite: 70 tests (69 passed, 1 test of mine was wrong and was rewritten, then passed); ruff and
      mypy --strict clean. The suite takes about 7 minutes on this machine because of TensorFlow.

- [x] Phase 4.1 and 3.2 results committed and pushed (`5793e3c`).
- [x] Phase 5.1: `supportx.db` (enums; models for customers, orders, order items, shipments, refund requests;
      engine/session helpers) and `supportx.services.refund` (deterministic refund-eligibility rule engine:
      pure, takes `now`, integer cents, always `requires_human_approval=True`, machine-readable denial codes).
      28 new tests, 100% coverage of the new code, run on in-memory SQLite with SQLAlchemy warnings as errors;
      ruff and mypy --strict clean.

- [x] Phase 5.1 committed and pushed (`01e9d52`).
- [x] Phase 5.2: `supportx.db.repositories` (order/refund queries, UTC normalization), `supportx.services.orders`
      (owner-only order summary; wrong email and missing order give the same error) and
      `supportx.services.refund_workflow` (check eligibility, create a PENDING request only when the rules allow,
      human approve/reject with approver, time and note recorded, no double decisions, amount re-checked at approval).
      22 new tests (50 across db/services), 100% coverage, SQLAlchemy warnings treated as errors;
      ruff and mypy --strict clean. New exceptions: `NotFoundError`, `WorkflowError`.

- [x] Phase 5.2 committed and pushed (`00a901a`).
- [x] Phase 8.1: FastAPI app (`supportx.api`): health and readiness, customer endpoints (order lookup, eligibility
      check, refund request) and staff endpoints (pending list, approve, reject) behind an `X-API-Key` header
      (constant-time comparison; endpoints are disabled with 503 if `STAFF_API_KEY` is unset). Errors map to 404/409/
      503/502/500 with generic messages; docs are hidden in production. 21 API tests (94% coverage; the engine-building
      startup path is covered by a real-server run instead) plus an end-to-end run of real `uvicorn` against a SQLite
      file over HTTP: request, staff auth, approve, double-approve 409, row saved in the file.
      ruff and mypy --strict clean. New setting: `STAFF_API_KEY`.

- [x] Phase 8.1 committed and pushed (`161a5a4`), including `tests/conftest.py`: the API tests left a logging handler
      behind that broke a later test (reproduced without the fixture, fixed with it). Full suite at that point:
      137 tests, 136 passed + that 1, now fixed.
- [x] Phase 6.1 (retrieval half of RAG, no API key needed): `supportx.rag` - Markdown chunking that keeps source file
      and heading path, a local TF-IDF embedder (LEXICAL: matches words, not meaning), FAISS exact inner-product
      index with save/load (bytes via Python I/O, tested in a folder containing an en dash), and a retriever with a
      minimum-score cutoff so "nothing relevant found" is possible. Three DEMO policy documents (fictional, consistent
      with the demo refund rules) in `data/sample/policies/`. 26 tests, 99% coverage of `supportx.rag`;
      ruff and mypy --strict clean.
      Retrieval check on 6 hand-written questions (written before seeing results): right document in the top 3 for
      6/6, ranked first for 5/6. The miss ("Can I cancel my order after it has shipped?") ranks the shipping policy first
      because 'cancel' does not match 'cancelling'/'cancelled' (no stemming); the right passage is second.
      6 questions cannot measure real quality; a semantic embedding model should fix this class of miss.

- [x] Phase 6.1 committed and pushed (`43fc099`).
- [x] Phase 7.1 (LangGraph skeleton with stub agents): `supportx.agents` - `TicketState`, a pure rule-based
      Supervisor (`decide_next`) in a hub-and-spoke graph, the agent interface (`AgentSet`), deterministic stub agents
      (keyword intent/sentiment, the REAL RAG retriever, an in-memory order book), a grounded template reply composer,
      human-in-the-loop via `interrupt()` + `Command(resume=...)` with a validated `HumanDecision`, node retries
      (`RetryPolicy` on `ExternalServiceError`), and an `error_handler` that degrades to a human handoff.
      Escalation triggers: low intent confidence, urgent negative sentiment, complaint / human requested, no matching
      policy ("never answer from nothing"), failed agent, step limit. Missing order number/email is asked for, not guessed.
      Verified against the installed LangGraph 1.2.21 by experiment first. 40 tests, 98% coverage of `supportx.agents`;
      ruff and mypy --strict clean.
      Bugs found by the tests: (1) an agent that still failed after its retries ended the run with no reply, because
      the error handler must return `Command(goto="supervisor")`; fixed.

- [x] Phase 7.1 committed and pushed (`3d456ba`). Full suite then: 203 passed.
- [x] Phase 7.2a (real components behind the graph): `supportx.agents.adapters` - the trained intent model
      (`make_intent_agent`), rule-based refund-reason inference, database-backed order lookup and refund check
      (`make_database_agents`: the refund agent CREATES the pending request in the database, commits before returning;
      database trouble becomes `ExternalServiceError`, which the graph retries), and `build_real_agents`.
      Intent confidence threshold measured (E5 in `docs/EXPERIMENTS.md`) and made a setting,
      `INTENT_MIN_CONFIDENCE`, default 0.6 (was 0.4). 23 adapter tests (plus supervisor/config updates), 100% coverage of
      `adapters.py`; ruff and mypy --strict clean. Sentiment is STILL the keyword stub (no sentiment model yet).

## Pending
- [x] Phase 7.2a committed and pushed (`9fe50dd`).
- [x] Phase 7.2b (full suite: 261 passed; ruff and mypy --strict clean): `tickets` table, `TicketService` (submit, owner-only status,
      two-step idempotent decision: record on the refund request, then resume the ticket), SqliteSaver checkpointer
      (`langgraph-checkpoint-sqlite` added to requirements, pip check clean), ticket endpoints (`/tickets`,
      `/tickets/status`, `/staff/tickets/awaiting`, `/staff/tickets/{id}/decision`), old refund endpoints now resume
      the waiting ticket. 19 service tests + 15 API tests pass; verified on a real server across a real restart.
- [x] 7.2b committed after the full run (261 passed in ~21 min, 2026-11-18).
- [x] MODEL QUALITY (E7, 2026-11-18): real order numbers map to the dataset's order-number token (`clean_text`);
      hand-written realistic phrasings (`supportx.data.augment`, `train_intent --augment 120`) added to training; frozen
      probe set `data/sample/intent_probe.csv` (dev/holdout). Holdout (scored once): 75% of in-scope probes handled
      automatically vs 47% before; confidently wrong 3 vs 5. New serving default `intent_tfidf_lr_bitext_aug.joblib`.
      Caveats: same author wrote phrasings and probes (optimistic); the augmented model's Bitext TEST was not scored; the
      full suite was not re-run after E7 (related tests passed).
- [ ] MODEL QUALITY, still open: needs REAL customer messages. A policy question ("refund after 30 days?") can still be
      routed as a refund request, and one tracking phrasing ("tracking update for order ORD-...") still escalates.
- [x] Retrieval (E8, 2026-11-18): stemming + doubled heading weight + reply quotes a close runner-up. Holdout: right section
      shown 83% (was 58%, n=12). Still lexical; a semantic embedder is the real fix. Full suite 294 passed.
- [ ] (superseded by E8, kept for history) Known limitation: the stub reply quotes only the top-1 passage and the lexical retriever's scores can be nearly
      tied and low (e.g. 0.143 / 0.128 / 0.126 for "What is your return policy? How many days do I have?", where the right
      passage ranked second). The minimum score (0.1) was not tuned from a few examples. The real LLM Response Agent
      should see several passages; a semantic embedding model should help.
- [x] Phase 9.1 (committed `f936582`): `requirements-api.txt` (no TensorFlow/MLflow; verified the API import path loads
      none of them), `docker/Dockerfile.api` (non-root, healthcheck, models mounted not baked in), `docker-compose.yml`
      (PostgreSQL 16, API, optional MLflow profile). NEVER BUILT OR RUN: Docker is not installed here.
- [x] Phase 9.2: Alembic (`alembic.ini`, `migrations/env.py` reads DATABASE_URL, `0001_initial_schema`).
      `tests/unit/test_migrations.py` upgrades to head on SQLite, checks there is no drift from the models, downgrades.
      The API container runs `alembic upgrade head` at start. The migration has NOT been run on PostgreSQL.
- [x] Phase 9.3 (observability): request id on every log line (`X-Request-ID`, reused only if it matches a safe pattern),
      access log of method/path/status/duration with NO query string, headers or bodies, `LOG_FORMAT=json` option,
      intent-model fingerprint (file, SHA-256, size, mtime) logged at startup and served by staff-only
      `GET /staff/model-info`. 8 new tests; ruff and mypy --strict clean. The full suite was NOT re-run after 9.3
      (74 related tests passed). Not done: metrics endpoint, tracing, per-ticket decision logging.
- [x] Phase 9.3b: `supportx.ml.final_test` (one-time test-split scoring, refuses to re-run) and `docs/EVALUATION.md`.
      TEST results, reference LogReg C=10, scored once: Bitext acc 0.993 / macro-F1 0.993 (n=2,427); Banking77 acc 0.897 /
      macro-F1 0.897 (n=3,080). Consistent with validation. The test split is now spent for this model: do not tune on it.
- [x] Local demo (2026-11-18): `scripts/seed_demo.py` (4 demo orders) and a self-contained demo page at `/demo`
      (customer chat + staff approvals; not mounted when `APP_ENV=production`; test added). The page's JavaScript was NOT run
      in a browser by me; I checked the page is served and the endpoints it calls. Run: set DATABASE_URL=sqlite:///demo.db,
      STAFF_API_KEY, APP_ENV=development, then `uvicorn supportx.api.app:create_app --factory`.
- [ ] Paused tickets now persist in SQLite; several server processes would need a PostgreSQL saver.
- [x] Cancellations (Phase 7.3, 2026-11-18): `supportx.services.cancellation` - pure rule (only a placed, unshipped order can be
      cancelled; DEMO DEFAULT matching `cancellation_policy.md`), denied at once with a reason (no human asked), otherwise paused
      for a human; on approval the order is really set to CANCELLED, with `cancelled_at`/`cancelled_by` (migration 0002), the rule
      is re-checked at approval time (order shipped meanwhile -> 409, staff can reject), and the step is idempotent after a crash.
      Not checked: a PENDING refund on a cancelled order; no customer notification; no stock/payment reversal (no such systems).
      Full suite 291 passed (one fixture count updated); ruff and mypy --strict clean; real-component run verified.
- [x] (done in 7.2a/7.2b) The agents are wired to the real intent model, database services and retriever; only sentiment is still a stub.
- [x] Phase 6.2 code (2026-11-25): `supportx.llm` - `LLMClient` (SDK timeout + retries, errors mapped to generic
      `ExternalServiceError`, token usage and cost estimate from YOUR configured prices, no prompts/keys in logs),
      `OpenAIEmbedder` (same `Embedder` interface; NOT yet wired into the retriever), `make_llm_composer` (rephrases only
      order-status and policy answers; refund/cancel/approval/handoff/missing-info text is never sent to the LLM; output must
      keep exactly the template's numbers/identifiers or the template is used). `LLM_REPLIES_ENABLED=false` by default.
      21 tests with a FAKE SDK; NEVER RUN AGAINST THE REAL API (no key). Verify against the installed openai 3.50.0 once a key exists.
- [ ] (original entry) Phase 6.2: OpenAI embeddings and chat wrapper (retries, timeouts) + grounded answers; needs OPENAI_API_KEY and
      OPENAI_CHAT_MODEL / OPENAI_EMBEDDING_MODEL in `.env` (not configured). The installed `openai` package is a
      newer major version than I know well, so the wrapper must be checked against the installed SDK before use.
- [x] Rate limiting (2026-11-18): per-client sliding-window limits (`RATE_LIMIT_PER_MINUTE`, default 120) and a lockout of
      /staff/* after `AUTH_FAILURES_PER_MINUTE` (default 10) wrong keys; health probes exempt; 429 + Retry-After; 7 tests.
      In-memory and per process (resets on restart, not shared between processes), does not read X-Forwarded-For.
- [x] Personal staff keys (2026-11-19): `STAFF_API_KEYS=name:key,...` (keys >= 16 chars, validated at start-up). The recorded
      approver is the key's name and a caller cannot claim another; the shared `STAFF_API_KEY` still works and still needs a
      stated approver. 9 tests. Not done: key rotation/expiry, revoking one key without a restart, role separation.
- [x] Audit log + revocable staff keys (2026-11-19, master-prompt Phase 7 part): `audit_logs` (refund requested/approved/
      rejected, order cancelled, ticket decided, staff key created/revoked; same transaction as the change; no customer
      emails), `GET /staff/audit`; `staff_keys` table with hashed keys, expiry, revocation, CLI
      `python -m supportx.services.staff_keys`, migration 0003. A revoked key fails immediately; environment keys still work
      if the key table is down. 14 new tests; 225 related tests pass; the full suite was NOT re-run. Not done: HTTPS,
      stronger customer authentication, distributed rate limiting, audit of failed logins, tamper-proof audit storage,
      authenticated approver for the shared key, authorization checks on agent tool calls, last-used tracking for keys.
- [x] Outbox + MOCK integrations (2026-11-19, master-prompt Phase 8): `supportx.integrations` (ports, labeled mocks,
      transactional outbox, dispatcher with idempotency keys, exponential backoff, dead-letter + requeue, worker CLI),
      `outbox_events` table (migration 0004), hooks on approved/rejected refunds and approved cancellations,
      `GET /staff/outbox`, `POST /staff/outbox/{id}/retry`. 16 new tests. NO REAL PROVIDER exists: with
      `INTEGRATIONS_MODE=mock` nothing real is paid, released or sent; with `disabled` (default) events wait. Not done: a
      real payment/inventory/notification adapter, PostgreSQL run of `SKIP LOCKED` delivery, a periodic worker in compose,
      alerting on dead events, payment records (a cancellation refunds the order total on the assumption it was paid).
- [x] Retrieval metrics + hybrid experiment (2026-11-19, master-prompt Phase 3 part): `supportx.rag.metrics` (Recall@K,
      Precision@K, MRR, abstention), `supportx.rag.hybrid` (BM25 + char n-gram + RRF; NOT adopted: it did not beat the baseline
      on the holdout), source filter on `Retriever.retrieve`, setting `RETRIEVAL_MIN_SCORE` (default unchanged 0.1).
      KEY FINDING (E9): at 0.1 the system answers 100% of unanswerable questions; lexical scores cannot separate them cleanly,
      so the cutoff is a trade-off (about 0.2 answers 70-79% of real questions and refuses 83% of unanswerable). Needs your
      choice. Still open: semantic embeddings (OpenAI key or PyTorch), wiring the source filter to the intent.
- [x] Phase 3 plumbing (2026-11-25): `Retriever.from_documents(..., embedder=...)` accepts any ready `Embedder` (e.g.
      `OpenAIEmbedder`); `python -m supportx.rag.metrics --embedder openai` will score it on the frozen questions. Tested with
      a fake synonym embedder only. NO semantic result exists: needs OPENAI_API_KEY + OPENAI_EMBEDDING_MODEL (paid calls).
      NOT done: choosing RETRIEVAL_MIN_SCORE (your decision, see E9: ~0.2 balanced), wiring the source filter to the intent
      (no intent-labelled retrieval questions to evaluate it on). Existing questions are synthetic/hand-written by the
      same author: they show direction, not production quality.
- [x] Phase 4 architecture (2026-11-25): `supportx.ml.sentiment` (`SentimentClassifier` negative/neutral/positive with
      confidence, dedupe-before-split 70/15/15 pipeline, CLI `python -m supportx.ml.sentiment data.csv`, test split scored only
      with `--score-test`), `make_sentiment_agent` (low confidence -> keyword rules; negative if model OR rules say so; urgency
      always from rules), setting `SENTIMENT_MODEL_PATH` (unset = keyword rules, the default). 8 tests on tiny SYNTHETIC data
      that only exercise code paths. NO MODEL IS TRAINED and NO SCORE EXISTS: the Amazon Reviews licence is still unverified,
      so nothing was downloaded. Sentiment is NOT production-ready. Needs: licensed/in-domain labelled data, then evaluate
      (incl. mixed/ambiguous messages) on a held-out set before setting the path.
- [x] Phase 5 pipeline (2026-11-25): `supportx.services.triage` (rule-based priority + team, DEMO tables; models plug in via
      `make_triager`, used only when confident, else rules + `needs_review`; a failing model never blocks a ticket; arrival-time
      facts only), `supportx.ml.triage_models` (priority/team LogReg with confidence, resolution-time Ridge on log hours, dedupe
      then chronological split when `created_at` exists, median-baseline MAE, test split only with `--score-test`). Tickets now
      store priority/team/source/needs_review/expected_resolution_hours (migration 0005; staff output only); the staff queue
      lists the most urgent first. Resolution time is None without a model (no invented hours). NO MODEL IS TRAINED: there is no
      real labelled ticket data, and the unit-test data is synthetic code-path data. Needs: real tickets with priority, team and
      resolution-time labels (>=100 distinct, 10+ per class), then evaluate chronologically before wiring models in.
      Not done: loading trained triage models from settings (nothing to load yet), a PostgreSQL run of migration 0005.
- [x] Phase 7 hardening (2026-11-25): `supportx.api.security` (security headers on all responses incl. 4xx/413/429,
      Content-Length size limit `MAX_REQUEST_BYTES`, optional `ALLOWED_HOSTS`, HSTS only with `HSTS_ENABLED`), failed staff
      logins logged without the key, production start-up warnings, `docs/SECURITY.md`, untested `deploy/Caddyfile.example`.
      6 tests. HTTPS is NOT configured or tested here. Still open: customer verification (needs a notification provider),
      shared rate limiting, chunked-body limit, dependency scan.
- [ ] API limits before production, still open: customers identified only by order number + email, no HTTPS, no PostgreSQL
      run, rate limits need a proxy/gateway layer too.
- [ ] Row locking (`SELECT ... FOR UPDATE`) that stops two simultaneous refund requests for one order is written but
      NOT exercised: SQLite ignores it and PostgreSQL is not installed. Needs a PostgreSQL test later.
- [x] Business rules configurable (2026-11-19, master-prompt Phase 1): `REFUND_RETURN_WINDOW_DAYS`,
      `REFUND_NOT_RECEIVED_AFTER_DAYS`, `BUSINESS_RULES_VERSION` in settings; production start-up warns while
      `BUSINESS_RULES_CONFIRMED=false`; `/staff/model-info` shows the live rules. New tests: delivered-without-date,
      not-received-without-ship-date, one-second boundaries, cancellation edges, duplicate cancellation tickets (4),
      PostgreSQL race tests for cancellation and refund requests (WRITTEN BUT NOT RUN: skipped without TEST_POSTGRES_URL).
      The cancellation rule (only an unshipped order) is structural, not numeric, so it is not a setting.
- [ ] **User decision still needed:** the real refund numbers/rules. Current values are DEMO defaults (30 days after
      delivery, 14 days after shipping for "not received"; final-sale items excluded; one pending request per order;
      cancellation only before shipping). Set them in `.env`, then `BUSINESS_RULES_CONFIRMED=true`.
- [ ] Not yet tested against PostgreSQL (Docker and PostgreSQL are not installed); SQLite returns naive datetimes,
      so the repository layer must normalize them to UTC (planned for 5.2).
- [x] The DL tests were re-run after the epoch-logging callback (they are part of every full run since; 301 passed).
- [ ] Sentiment, priority, routing and resolution-time models (need data decisions; see `docs/DATASETS.md`)
- [ ] Open dataset verification items (see `docs/DATASETS.md`), notably CDLA-Sharing-1.0 terms
- [x] Roadmap phases 2-9 all have code (see the phase entries above).

## Known notes
- Harmless import-time messages: TensorFlow prints an absl log notice and an oneDNN notice
  (can be silenced with `TF_ENABLE_ONEDNN_OPTS=0`); MLflow prints an INFO hint.
- TensorFlow is CPU-only on native Windows (GPU would need WSL2).
- pip 25.0.1 is older than the latest (26.3); upgrading was not requested, so it was left alone.
- Docker and PostgreSQL are not installed (not needed until Phase 5).
- Not verified yet: a real PostgreSQL connection, OpenAI API access (no key configured),
  and NLTK data downloads (e.g., stopwords/punkt) - these belong to later phases.
- Models trained: TF-IDF + LogReg intent models (the served one is `intent_tfidf_lr_bitext_aug.joblib`), XGBoost and
  BiLSTM experiments. Datasets: Bitext and Banking77 (downloaded, prepared). No sentiment/priority/routing models.

## Next task
`docs/ARCHITECTURE.md` was rewritten on 2026-11-19 to match the code (the `/chat` and `/approvals` references are gone).
Buildable without outside resources: key rotation/revocation and audit logging (security), mock payment/inventory/
notification adapters with an outbox, hybrid retrieval and retrieval metrics. Blocked: Docker/PostgreSQL runs, OpenAI.
All roadmap phases have code. Next depends on you: refund policy numbers, an OpenAI key (Phase 6.2), the Amazon Reviews
license (sentiment), or better intent data (E5b). Also pending: build the Docker image and run on PostgreSQL where Docker exists.
Still open elsewhere: sentiment model (blocked until the Amazon Reviews 2023 terms are read) and improving the
DL intent model (needs the deferred Hugging Face/PyTorch install).
