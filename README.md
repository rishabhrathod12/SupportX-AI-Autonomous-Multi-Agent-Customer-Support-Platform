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
