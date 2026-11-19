# SupportX AI - Autonomous Multi-Agent Customer Support Platform

An e-commerce customer support workflow: a rule-based Supervisor routes each ticket through intent
classification, sentiment, policy retrieval (RAG), order lookup and refund/cancellation checks
(LangGraph), then replies from grounded templates. **Refunds and cancellations always wait for a
human decision.** Served by FastAPI over SQLAlchemy.

> Status: working end to end on a local demo (see `PROJECT_STATUS.md` for the full, honest list of what is
> done, unverified and open). Key limits: no LLM (replies are templates; needs an OpenAI key), sentiment is a
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
| `GET /staff/audit?limit=&action=&target_id=` | staff | append-only trail: refund requested/approved/rejected, order cancelled, ticket decided, staff keys created/revoked (newest first) |

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
pytest                       # full suite: 301 passed at its last full run (about 5-8 minutes; TensorFlow is slow)
ruff check src tests migrations ; ruff format --check src tests migrations
mypy --strict src
python -m supportx.ml.probe_eval artifacts/models/intent_tfidf_lr_bitext_aug.joblib --split dev --show-failures
python -m supportx.rag.evaluate --split dev --show-misses
```

The holdout probe/retrieval sets and the test split of the datasets were each scored once for the reported numbers;
do not tune against them (`docs/EXPERIMENTS.md`).

## Database migrations and Docker

`alembic upgrade head` (reads `DATABASE_URL`) applies `migrations/`; a test checks the migrations match the models.
`docker-compose.yml` (PostgreSQL, API, optional MLflow) and `docker/Dockerfile.api` exist but were **never built or
run**. Mount your trained model at `artifacts/models`.

## Not production-ready

Customers are identified only by order number + email; no HTTPS; rate limits are in memory and per process (also
limit at a proxy); the audit log lives in the same database (anyone with write access to it can alter it); the
shared `STAFF_API_KEY` still lets the caller name the approver; nothing reverses a payment, releases stock or
notifies the customer. Details and the open list: `PROJECT_STATUS.md`.

## Documentation
- `docs/ARCHITECTURE.md` - architecture and design decisions
- `docs/EXPERIMENTS.md` - every experiment, with its caveats; `docs/EVALUATION.md` - results and limitations
- `docs/DATASETS.md` - datasets, licenses, data checks
- `DEVELOPMENT_ROADMAP.md` - phased plan; `PROJECT_STATUS.md` - done / pending

## Project layout
```
src/supportx/   core, db, nlp, ml, dl, rag, agents, services, api
migrations/     Alembic            scripts/   demo seed
tests/          unit/              data/      raw/ and processed/ (ignored), sample/ (demo policies, probe sets)
artifacts/      trained models (ignored)       docs/ docker/ notebooks/
```
