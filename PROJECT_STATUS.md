# Project Status

Last updated: 2026-11-13 | Current phase: 1.2 - core package (complete, awaiting review before Phase 1.3)

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

## Pending
- [ ] Phase 1.3 and onward per `DEVELOPMENT_ROADMAP.md` (needs user approval)
- [ ] First git commit and push (waiting for user go-ahead)

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
Phase 1.3: first git commit (after reviewing what will be committed) and push to GitHub.
