# SupportX AI - Autonomous Multi-Agent Customer Support Platform

An e-commerce customer support workflow: a rule-based Supervisor routes each ticket through intent
classification, sentiment, policy retrieval (RAG), order lookup and refund/cancellation checks
(LangGraph), then replies from grounded templates. **Refunds and cancellations always wait for a
human decision.** Served by FastAPI over SQLAlchemy.

> Status: working end to end on a local demo (see `PROJECT_STATUS.md` for the full, honest list of what is
> done, unverified and open). Key limits: LLM integration is built and tested only with a fake SDK (never run against OpenAI; off by default, needs a key), sentiment is a
> keyword stub, retrieval is lexical, the intent model was trained on synthetic data, and the Docker image and
> PostgreSQL have never been run (not installed on the development machine). The refund and cancellation rules
> use DEMO numbers, not a real policy. Evaluation: `docs/EVALUATION.md`.

## Quick start (local demo, Windows PowerShell)

```powershell
python -m venv venv ; .\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt          # large: includes TensorFlow and MLflow
$env:PYTHONPATH = "src"

# 1. data and the intent model (public datasets; read their licenses first: docs/DATASETS.md)
python -m supportx.data.download bitext --accept-license
python -m supportx.data.prepare all
python -m supportx.ml.train_intent bitext --augment 120    # writes artifacts/models/intent_tfidf_lr_bitext_aug.joblib

# 2. demo database with four orders, then the server
$env:DATABASE_URL = "sqlite:///demo.db"; $env:STAFF_API_KEY = "choose-a-key"; $env:APP_ENV = "development"
python scripts/seed_demo.py
uvicorn supportx.api.app:create_app --factory
```

Open `http://127.0.0.1:8000/demo` (customer chat and staff approvals; not served when `APP_ENV=production`) or
`http://127.0.0.1:8000/docs`. Demo customer: `demo@example.com`; orders ORD-1001 (refundable), ORD-2002 (past the
return window), ORD-3003 (cancellable), ORD-4004 (shipped). If the model or policy documents are missing the server
still starts and the ticket endpoints answer 503.

## Configuration (environment or `.env`; see `.env.example`)

| Variable | Meaning |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL (`sqlite:///demo.db` locally, PostgreSQL in compose) |
| `STAFF_API_KEYS` | personal staff keys `name:key,...` (keys 16+ chars); the recorded approver is the key's name |
| `STAFF_API_KEY` | one shared staff key (the caller states the approver); unset and no personal keys = staff endpoints off |
| `APP_ENV` | `development` creates tables and serves `/demo`; `production` hides docs and `/demo` and expects migrations |
| `REFUND_RETURN_WINDOW_DAYS`, `REFUND_NOT_RECEIVED_AFTER_DAYS`, `BUSINESS_RULES_VERSION` | refund rule numbers (DEMO defaults 30 / 14 / `demo-1`); every refund decision records the version |
| `BUSINESS_RULES_CONFIRMED` | set `true` only after the numbers are your real policy; production start-up logs a warning while it is `false`; `GET /staff/model-info` shows the live rules |
| `RETRIEVAL_MIN_SCORE` | best-passage score needed to quote policy text, else a person takes over (default 0.2, chosen 2026-11-25: on the hand-written questions it answers about 70-79% of real questions and hands over about 83% of unanswerable ones; 0.1 answers everything; trade-off in `docs/EXPERIMENTS.md`, E9) |
| `INTEGRATIONS_MODE`, `OUTBOX_MAX_ATTEMPTS` | `disabled` (default), `mock` (demo adapters) or `live` (refuses to start: no real provider exists); attempts before an event goes `dead` (5) |
| `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, price settings | OpenAI wrapper (never run against the real API); cost estimates only when you set the prices |
| `LLM_REPLIES_ENABLED` | `false` (default). `true` lets the LLM rephrase order-status and policy replies behind a grounding check; refunds, cancellations, approvals and handoffs stay template text |
| `SENTIMENT_MODEL_PATH`, `SENTIMENT_MIN_CONFIDENCE` | optional trained sentiment model (none shipped; unset = keyword rules) |
| `MAX_REQUEST_BYTES`, `ALLOWED_HOSTS`, `HSTS_ENABLED` | request-size limit (64 KB), Host allow-list, HSTS header (only with real HTTPS); see `docs/SECURITY.md` |
| `INTENT_MODEL_PATH`, `POLICY_DOCS_DIR`, `CHECKPOINT_DB_PATH` | model file, policy folder, where paused tickets are saved |
| `INTENT_MIN_CONFIDENCE` | below this a ticket goes to a human (default 0.6) |
| `RATE_LIMIT_PER_MINUTE`, `AUTH_FAILURES_PER_MINUTE` | per-client limits (120 / 10; 0 = off) |
| `LOG_LEVEL`, `LOG_FORMAT` | `text` or `json` log lines; every line carries a request id |

## Endpoints

| Endpoint | Who | Purpose |
|---|---|---|
| `GET /health`, `GET /health/ready` | anyone | process up / database reachable (never rate limited) |
| `POST /tickets` | customer | send a message to the workflow; answers at once, or 202 when a person must decide first |
| `POST /tickets/status` | customer (ticket id + email) | the reply or current status |
| `POST /orders/lookup`, `POST /refunds/eligibility`, `POST /refunds/requests` | customer (order number + email) | order status, refund check, create a pending refund request |
| `GET /staff/tickets/awaiting` | staff (`X-API-Key`) | tickets paused for a decision |
| `POST /staff/tickets/{id}/decision` | staff | approve/reject; an approved refund or cancellation is applied, then the ticket resumes |
| `GET /staff/refunds/pending`, `POST /staff/refunds/{id}/approve` or `/reject` | staff | the same decisions by refund id (reject needs a note) |
| `GET /staff/model-info` | staff | which intent model file (SHA-256) and settings are serving |
| `GET /staff/outbox`, `GET /staff/outbox/stats`, `POST /staff/outbox/{id}/retry` | staff | follow-up queue, counts and oldest waiting age for monitoring, requeue a dead event |
| `GET /staff/audit?limit=&action=&target_id=` | staff | append-only trail: refund requested/approved/rejected, order cancelled, ticket decided, staff keys created/revoked (newest first) |

### Payment, stock and customer-message follow-ups (outbox; MOCK adapters only)

An approved refund, a rejected refund and an approved cancellation each save follow-up events in the SAME database
transaction as the decision (`outbox_events`): refund the payment, release the stock, notify the customer. A
separate worker delivers them with idempotency keys, retries with exponential backoff, and marks events that keep
failing as `dead` for a person to look at.

```powershell
$env:INTEGRATIONS_MODE = "mock"      # default "disabled": events are saved but nothing delivers them
python -m supportx.integrations.worker --once                  # or: --loop --interval 10
```

**Only MOCK adapters exist.** In `mock` mode nothing real is paid, released or sent, and every delivered event is
labeled `is_mock`. A real provider must implement the protocols in `src/supportx/integrations/ports.py` and honour
the idempotency key. Staff see the queue at `GET /staff/outbox?status=dead` and requeue with
`POST /staff/outbox/{id}/retry`. Delivery is at least once (a crash can repeat a call, which the key makes harmless).

### Staff keys without a restart

Besides `STAFF_API_KEYS` / `STAFF_API_KEY` (read at start-up), keys can live in the database and change while the
server runs. Only a SHA-256 hash is stored; the key is shown once.

```powershell
python -m supportx.services.staff_keys create jane --days 90    # prints the key once
python -m supportx.services.staff_keys list
python -m supportx.services.staff_keys revoke jane              # or: revoke --id 3
```

Rotation = create a new key, hand it over, revoke the old one; the old key stops working immediately. Requires
migration `0003` (`alembic upgrade head`; or `APP_ENV=development`, which creates the tables).

Cancellation rule (demo default): only an order that has not shipped can be cancelled; otherwise the customer gets
a reason at once and no human is asked. Refund rule (demo defaults): 30 days after delivery; "not received" only
after 14 days from shipping.

## Tests and checks

```powershell
pytest                       # full suite: 423 passed, 2 skipped (PostgreSQL race tests) at its last full run (3-10 minutes; TensorFlow is slow)
ruff check src tests migrations ; ruff format --check src tests migrations
mypy --strict src
python -m supportx.ml.probe_eval artifacts/models/intent_tfidf_lr_bitext_aug.joblib --split dev --show-failures
python -m supportx.rag.evaluate --split dev --show-misses
python -m supportx.rag.metrics --split dev --legacy --show-misses   # Recall@K, Precision@K, MRR, false-answer rate
python -m supportx.ml.agent_eval artifacts/models/intent_tfidf_lr_bitext_aug.joblib   # 17 end-to-end scenarios (synthetic)
```

The holdout probe/retrieval sets and the test split of the datasets were each scored once for the reported numbers;
do not tune against them (`docs/EXPERIMENTS.md`).

## Database migrations and Docker

`alembic upgrade head` (reads `DATABASE_URL`) applies `migrations/`; a test checks the migrations match the models.
`docker-compose.yml` (PostgreSQL, API, optional MLflow) and `docker/Dockerfile.api` exist but were **never built or
run**. Mount your trained model at `artifacts/models`. Exact build, PostgreSQL verification, backup, rollback and
recovery steps: `docs/DEPLOYMENT.md` (all unverified).

## Not production-ready

Customers are identified only by order number + email; HTTPS is not configured (example proxy file untested); rate limits are in memory and per process (also
limit at a proxy); the audit log lives in the same database (anyone with write access to it can alter it); the
shared `STAFF_API_KEY` still lets the caller name the approver; there is NO real payment, inventory or notification
provider (mock adapters only), so no money moves, no stock changes and no customer is contacted. Details and the open list: `PROJECT_STATUS.md`.

## Documentation
- `docs/ARCHITECTURE.md` - architecture and design decisions
- `docs/EXPERIMENTS.md` - every experiment, with its caveats; `docs/EVALUATION.md` - results and limitations
- `docs/DATASETS.md` - datasets, licenses, data checks
- `docs/SECURITY.md` - what is protected, what is not; `docs/DEPLOYMENT.md` - Docker, PostgreSQL, backup, rollback, recovery (unverified)
- `DEVELOPMENT_ROADMAP.md` - phased plan; `PROJECT_STATUS.md` - done / pending

## Project layout
```
src/supportx/   core, db, nlp, ml, dl, rag, llm, agents, services, integrations, api
migrations/     Alembic            scripts/   demo seed
tests/          unit/              data/      raw/ and processed/ (ignored), sample/ (demo policies, probe sets)
artifacts/      trained models (ignored)       docs/ docker/ notebooks/
```
