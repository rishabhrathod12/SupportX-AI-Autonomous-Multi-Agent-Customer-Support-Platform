# SupportX AI - Autonomous Multi-Agent Customer Support Platform

An e-commerce customer support platform that combines classical ML, deep learning (TensorFlow/Keras),
NLP, RAG, and a LangGraph multi-agent workflow. It classifies intent, analyzes sentiment, investigates
orders, retrieves company policy, checks refund eligibility, and drafts accurate replies.
**Refunds and cancellations always require human approval.**

> Status: project initialization complete. No application code yet - see `PROJECT_STATUS.md`.

## Tech stack
Python 3.12, Pandas/NumPy, scikit-learn, XGBoost, TensorFlow/Keras, NLTK, OpenAI API,
LangChain, LangGraph, FAISS, FastAPI, PostgreSQL, SQLAlchemy, Pydantic, MLflow, Pytest, Docker.

## Documentation
- `docs/ARCHITECTURE.md` - architecture and design decisions
- `DEVELOPMENT_ROADMAP.md` - phased plan
- `PROJECT_STATUS.md` - what is done / pending

## Setup (Windows PowerShell)

```powershell
# 1. Create and activate the virtual environment (Python 3.12)
python -m venv venv
.\venv\Scripts\Activate.ps1
#    If blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install dependencies (large: TensorFlow + MLflow, several GB of downloads)
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# 3. Configure secrets
Copy-Item .env.example .env      # then edit .env; never commit it

# 4. Check the toolchain
pytest
```

Linux/macOS: use `source venv/bin/activate` and `cp .env.example .env`.

## Run the API

```powershell
# in .env (copy of .env.example): set DATABASE_URL and STAFF_API_KEY
#   quick local option without PostgreSQL:  DATABASE_URL=sqlite:///supportx.db
$env:PYTHONPATH = "src"
uvicorn supportx.api.app:create_app --factory --reload
# interactive docs: http://127.0.0.1:8000/docs  (hidden when APP_ENV=production)
```

| Endpoint | Who | Purpose |
|---|---|---|
| `GET /health`, `GET /health/ready` | anyone | process up / database reachable |
| `POST /orders/lookup` | customer (order number + email) | order status |
| `POST /refunds/eligibility` | customer | would a refund be allowed? (creates nothing) |
| `POST /refunds/requests` | customer | create a pending refund request if the rules allow |
| `GET /staff/refunds/pending` | staff (`X-API-Key`) | refunds waiting for a decision |
| `POST /staff/refunds/{id}/approve`, `/reject` | staff (`X-API-Key`) | human decision (reject needs a note); also resumes a paused ticket waiting on that refund |
| `POST /tickets` | customer | send a message to the agent workflow; answers immediately, or returns 202 when a person must decide first |
| `POST /tickets/status` | customer (ticket id + email) | the reply or current status of a ticket |
| `GET /staff/tickets/awaiting` | staff (`X-API-Key`) | tickets paused for a human decision |
| `POST /staff/tickets/{id}/decision` | staff (`X-API-Key`) | approve/reject a paused ticket (records the refund decision and resumes the ticket) |

The ticket endpoints need the trained intent model (`python -m supportx.ml.train_intent bitext`, which writes
`artifacts/models/intent_tfidf_lr_bitext.joblib`) and the policy documents. If either is missing the server still
starts, logs a warning, and the ticket endpoints answer 503. Paused tickets are saved in a SQLite file
(`CHECKPOINT_DB_PATH`) and survive a restart. Note: the demo-trained model currently hands many ordinary messages to a
human (see `docs/EXPERIMENTS.md`, E5b).

Not production-ready yet: staff use one shared API key (the approver name is self-declared), customers are
identified only by order number + email, and there is no rate limiting or HTTPS. Add real authentication before
exposing it. The refund policy numbers are demo defaults.

## Notes
- TensorFlow on native Windows runs on CPU only. For GPU, use WSL2 (optional, later).
- PostgreSQL and Docker are required from the database phase onward; they are not needed yet.
- `data/sample/` will contain synthetic demo data only. Real datasets go in `data/raw/` (git-ignored).

## Project layout
```
src/supportx/   core, db, nlp, ml, dl, llm, rag, agents, services, api
tests/          unit/ and integration/
data/           raw/ (real, ignored), sample/ (synthetic demo), processed/
artifacts/      trained models and vector stores (ignored)
docs/ scripts/ docker/ notebooks/
```
