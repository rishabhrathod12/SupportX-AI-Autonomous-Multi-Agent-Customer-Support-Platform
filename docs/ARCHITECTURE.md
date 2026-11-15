# SupportX AI - Architecture

## 1. High-level flow

```
Customer query
      |
      v
 FastAPI  (/tickets, /chat, /approvals)
      |
      v
 LangGraph workflow  <---- shared TicketState (Pydantic)
      |
 Supervisor Agent  --routes-->  Query Understanding  (intent ML/DL model)
      |                         Sentiment Agent       (sentiment + urgency model)
      |                         Knowledge Agent       (RAG over policies, FAISS)
      |                         Order Agent           (PostgreSQL via repositories)
      |                         Refund Agent          (rules + order facts)
      |                               |
      |                    sensitive action? --yes--> HUMAN APPROVAL (interrupt)
      v
 Response Agent (LLM, grounded only in retrieved facts)
      |
      v
 Reply + audit log in PostgreSQL;  metrics/experiments in MLflow
```

## 2. Module responsibilities

| Package (`src/supportx/`) | Responsibility | May import from |
|---|---|---|
| `core` | Settings (env vars), logging, custom exceptions, shared types | nothing internal |
| `db` | SQLAlchemy models, sessions, repositories, Alembic migrations | `core` |
| `data` | License-gated dataset download (pinned revisions, SHA-256 manifest) and a labeled synthetic sample generator for tests | `core` |
| `nlp` | Text cleaning, tokenization, NLTK utilities (shared preprocessing) | `core` |
| `ml` | Classical ML: intent, sentiment, priority, routing, resolution time (sklearn/XGBoost). Train, evaluate, save, load, predict | `core`, `nlp` |
| `dl` | TensorFlow/Keras: dense NN, LSTM/BiLSTM for intent and sentiment; same train/evaluate/save/load/predict contract as `ml` | `core`, `nlp`, `data` (split loading), and the shared metric/logging helpers `ml.evaluate` / `ml.experiment` |
| `llm` | Thin OpenAI/LangChain wrapper: retries, timeouts, prompt templates | `core` |
| `rag` | Document loading, chunking, embeddings, FAISS index, retriever (pgvector-ready interface) | `core`, `llm` |
| `agents` | LangGraph state, graph, one module per agent, routing, human-in-the-loop | `core`, `ml`, `dl`, `rag`, `llm`, `services` |
| `services` | Business logic (order lookup, refund eligibility rules, ticket handling) - no HTTP, no LLM | `core`, `db` |
| `api` | FastAPI app, routers, request/response schemas, dependencies | `services`, `agents` |

Dependency rule: arrows only point downward in the table. `ml`, `dl`, `rag`, `agents`, and `api` never import each other sideways, except `agents` consuming the others through small interfaces.

## 3. Key decisions (and why)

1. **`src/` layout** - prevents accidental imports from the repo root and mirrors real packages.
2. **ML and DL are separate packages with the same interface** (`train`, `evaluate`, `save`, `load`, `predict`). The agents depend on the interface, so a model can be swapped (e.g., XGBoost -> BiLSTM) after comparing metrics in MLflow.
3. **Refund eligibility is deterministic code, not LLM judgment.** The LLM only explains the decision. Money-related logic must be testable and auditable.
4. **Human approval is a LangGraph interrupt with a checkpointer**, so a paused workflow survives restarts and resumes after an approver decides via the API.
5. **FAISS first behind a `Retriever` interface**; pgvector can replace it later without touching agents.
6. **Services layer has no framework code**, so business rules are unit-testable without FastAPI, the DB, or the LLM.
7. **Secrets only through environment variables** (`pydantic-settings`); `.env` is git-ignored, `.env.example` is committed.
8. **Data honesty**: `data/sample/` holds clearly labeled synthetic demo data for tests and demos. Reported metrics must come from real datasets stored in `data/raw/` and logged in MLflow with the dataset name and version.
9. **Hugging Face deferred**: it requires a backend (PyTorch), which is large and not needed until the NLP/embedding phase. Adding it later avoids early dependency bloat.

## 4. Agents

| Agent | Input | Output | Tools |
|---|---|---|---|
| Supervisor | TicketState | next node, retry/escalation decision | routing rules, LLM fallback |
| Query Understanding | raw text | intent + confidence | intent model |
| Sentiment | raw text | sentiment, urgency, priority | sentiment/priority models |
| Knowledge | question | policy passages + sources | RAG retriever |
| Order Management | order id | order, shipment status | order service |
| Refund | order + policy | eligible / not eligible + reason; approval request | refund service |
| Response | all state | final reply grounded in facts | LLM |

## 5. Data model (initial sketch)

Implemented (Phase 5.1): `customers`, `orders`, `order_items`, `shipments`, `refund_requests`.
Money is stored as integer cents; timestamps are timezone-aware UTC. The human decision is stored on the
refund request itself (`status` pending_approval / approved / rejected, `decided_by`, `decided_at`,
`decision_note`), so there is no separate `approvals` table.
Planned: `tickets`, `ticket_messages`, `audit_logs`.

## 6. Testing strategy

- Unit tests: services (refund rules), preprocessing, model interfaces, state schemas.
- Integration tests: API + test database, graph with a mocked LLM.
- No test may call the real OpenAI API.
