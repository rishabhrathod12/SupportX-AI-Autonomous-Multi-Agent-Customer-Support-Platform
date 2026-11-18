# Project Status

Last updated: 2026-11-13 | Current phase: intent data improvement (E7) done, uncommitted; Docker files untested. Remaining items need user input

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
- [ ] Known limitation: the stub reply quotes only the top-1 passage and the lexical retriever's scores can be nearly
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
- [ ] Paused tickets now persist in SQLite; several server processes would need a PostgreSQL saver.
- [ ] Cancellations only get a human decision here; nothing actually cancels an order yet (no cancellation rules/workflow).
- [ ] The stub agents are not wired to the real ML intent model, database services or retriever-in-API yet.
- [ ] Phase 6.2: OpenAI embeddings and chat wrapper (retries, timeouts) + grounded answers; needs OPENAI_API_KEY and
      OPENAI_CHAT_MODEL / OPENAI_EMBEDDING_MODEL in `.env` (not configured). The installed `openai` package is a
      newer major version than I know well, so the wrapper must be checked against the installed SDK before use.
- [ ] API limits before production: one shared staff key (approver name is self-declared), customers identified only
      by order number + email, no rate limiting, no HTTPS, no PostgreSQL run.
- [ ] Row locking (`SELECT ... FOR UPDATE`) that stops two simultaneous refund requests for one order is written but
      NOT exercised: SQLite ignores it and PostgreSQL is not installed. Needs a PostgreSQL test later.
- [ ] **User decision needed:** the refund policy numbers (30-day return window after delivery, 14 days before an
      order can be reported "not received") are DEMO DEFAULTS I chose, not your real policy. Please confirm or
      replace them, along with the rules themselves (e.g. final-sale items, one pending request per order).
- [ ] Not yet tested against PostgreSQL (Docker and PostgreSQL are not installed); SQLite returns naive datetimes,
      so the repository layer must normalize them to UTC (planned for 5.2).
- [ ] The epoch-logging callback in `dl/intent.py` was added after the last full test run; the training runs
      exercised it, but the DL tests were not re-run after it.
- [ ] Sentiment, priority, routing and resolution-time models (need data decisions; see `docs/DATASETS.md`)
- [ ] Open dataset verification items (see `docs/DATASETS.md`), notably CDLA-Sharing-1.0 terms
- [ ] Phase 2.2 onward per `DEVELOPMENT_ROADMAP.md`

## Known notes
- Harmless import-time messages: TensorFlow prints an absl log notice and an oneDNN notice
  (can be silenced with `TF_ENABLE_ONEDNN_OPTS=0`); MLflow prints an INFO hint.
- TensorFlow is CPU-only on native Windows (GPU would need WSL2).
- pip 25.0.1 is older than the latest (26.3); upgrading was not requested, so it was left alone.
- Docker and PostgreSQL are not installed (not needed until Phase 5).
- Not verified yet: a real PostgreSQL connection, OpenAI API access (no key configured),
  and NLTK data downloads (e.g., stopwords/punkt) - these belong to later phases.
- No models trained; no datasets chosen yet.

## Next task
All roadmap phases have code. Next depends on you: refund policy numbers, an OpenAI key (Phase 6.2), the Amazon Reviews
license (sentiment), or better intent data (E5b). Also pending: build the Docker image and run on PostgreSQL where Docker exists.
Still open elsewhere: sentiment model (blocked until the Amazon Reviews 2023 terms are read) and improving the
DL intent model (needs the deferred Hugging Face/PyTorch install).
