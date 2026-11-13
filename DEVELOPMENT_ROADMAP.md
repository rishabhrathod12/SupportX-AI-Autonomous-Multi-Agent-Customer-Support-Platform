# Development Roadmap

Rule: finish, test, and review one phase before starting the next. Each task follows: what/why ->
architecture -> code -> Hinglish explanation -> run/test commands -> common errors -> interview questions.

## Phase 0 - Initialization (done)
Workspace inspection, architecture, dependency resolution, repo skeleton, docs.

## Phase 1 - Foundation
1. Install dependencies; verify imports and versions
2. `core`: settings (pydantic-settings), logging, custom exceptions + unit tests
3. First commit, push to GitHub, CI-less local checks (ruff, mypy, pytest)

## Phase 2 - Data and NLP
1. Pick real datasets (e.g., public support-ticket / intent datasets); document source and license
2. Synthetic sample data generator (clearly labeled), for tests only
3. `nlp`: text cleaning + tokenization, with tests
4. EDA notebook; train/validation/test split strategy

## Phase 3 - Classical ML (baselines, tracked in MLflow)
1. Intent classification (TF-IDF + LogisticRegression / XGBoost)
2. Sentiment analysis
3. Ticket priority prediction
4. Ticket routing
5. Resolution time prediction (regression)
Each: train, validate, evaluate, save, load, predict, tests.

## Phase 4 - Deep Learning (TensorFlow/Keras)
1. Dense NN on TF-IDF for intent
2. LSTM / BiLSTM for intent and sentiment
3. Compare against Phase 3 baselines in MLflow; keep a DL model only if it justifies its cost

## Phase 5 - Database and Services
1. PostgreSQL via Docker; SQLAlchemy models; Alembic migrations
2. Repositories; order and refund-eligibility services with unit tests

## Phase 6 - LLM and RAG
1. `llm` wrapper (retries, timeouts, prompt templates)
2. Policy documents -> chunking -> embeddings -> FAISS
3. Retriever with source citations; retrieval evaluation

## Phase 7 - Agents (LangGraph)
1. `TicketState` and graph skeleton with mocked agents
2. Implement agents one at a time: Query Understanding, Sentiment, Knowledge, Order, Refund, Response
3. Supervisor: conditional routing, retries, escalation
4. Human-in-the-loop approval with checkpointer

## Phase 8 - API
1. FastAPI app, health check, ticket/chat endpoints
2. Approval endpoints, error handling, request validation
3. Integration tests

## Phase 9 - Productionization
1. Dockerfile + docker-compose (API, PostgreSQL, MLflow)
2. Observability: structured logging, model/version tracking
3. Evaluation report, documentation, demo
