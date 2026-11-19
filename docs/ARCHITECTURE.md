# SupportX AI - Architecture

This document describes what is IMPLEMENTED. Things that are planned but not built are marked **(planned)** and
collected in section 8. Facts about quality and limits live in `docs/EVALUATION.md` and `PROJECT_STATUS.md`.

## 1. High-level flow (as built)

```
Customer message  ->  POST /tickets  (FastAPI; request id, rate limit)
                            |
                     TicketService.submit  (writes the ticket row, runs the graph)
                            |
              LangGraph workflow  <---- shared TicketState (TypedDict), one thread per ticket
                            |
   Supervisor (rule-based, pure function `decide_next`) routes, after every step, to:
        query_understanding  intent + confidence   (TF-IDF + LogReg model)
        sentiment            label + urgency       (KEYWORD STUB, no model yet)
        knowledge            policy passages       (RAG: lexical TF-IDF + FAISS)
        order                order summary         (owner-checked database lookup)
        refund               eligibility rules; creates a PENDING refund request
                            |
        refund or cancellation that is allowed  -->  human_approval  (LangGraph interrupt, paused)
                            |
        respond              grounded TEMPLATE reply (no LLM)
                            |
          ticket row updated (status, reply, escalation reason)

Paused ticket:  POST /staff/tickets/{id}/decision  ->  (1) apply the decision in the database
                (refund approved/rejected, or order cancelled; idempotent)  ->  (2) resume the graph.
```

Escalation to a person happens (reply = a handoff message) when: intent confidence is below
`INTENT_MIN_CONFIDENCE`, sentiment is urgent and negative, the intent is a complaint or a request for a human, no
policy passage matches ("never answer from nothing"), an agent still fails after its retries, or the step limit
(`MAX_STEPS = 12`) is reached. A missing order number or email is asked for, never guessed.

## 2. Module responsibilities

| Package (`src/supportx/`) | Responsibility | Imports from |
|---|---|---|
| `core` | Settings (env vars, validation), logging (request id, text/JSON), exceptions, model-file fingerprint | nothing internal |
| `db` | SQLAlchemy models, enums, sessions, repositories (UTC normalization, row locks). Alembic migrations live in `migrations/` at the repo root | `core` |
| `data` | License-gated dataset download, split preparation, synthetic sample generator, similarity check, hand-written training phrasings (`augment`) | `core`, `nlp` |
| `nlp` | Text cleaning and tokenization shared by training and serving (order numbers become the dataset's order-number token) | - |
| `ml` | Intent classifier (TF-IDF + LogReg or XGBoost), training with MLflow, evaluation, the one-shot test scorer, the probe-set scorer | `core`, `nlp`, `data` |
| `dl` | Keras Embedding + BiLSTM intent model (a documented experiment; not used in serving) | `core`, `nlp`, `data`, `ml` helpers |
| `llm` | EMPTY placeholder (no LLM integration yet) **(planned)** | - |
| `rag` | Markdown chunking, local TF-IDF embedder with light stemming, FAISS exact index, retriever (`retrieve_for_answer` picks the best passage plus a close runner-up), retrieval scorer | `core`, `nlp` |
| `services` | Business rules with no HTTP and no LLM: refund eligibility and workflow, cancellation rule and action, order lookup | `core`, `db` |
| `agents` | The graph, the Supervisor rules, `TicketState`, the agent interface (`ports.AgentSet`), real adapters (`adapters.py`), deterministic stubs (`stubs.py`), template replies, `TicketService` and the checkpointer wiring | `core`, `ml`, `rag`, `services`, `db` |
| `api` | App factory, routers (customer, tickets, staff, health, demo), schemas, dependencies, request-logging and rate-limit middleware | `agents`, `services`, `db`, `core` |

Notes on the dependency direction: `agents` consumes models and retrieval through small callables, so a model can be
swapped without touching the graph. The Supervisor imports the pure cancellation rule from `services`. `api` never
touches models directly. Agents are functions grouped in an `AgentSet`, not one module per agent.

## 3. Key decisions (and why)

1. **`src/` layout** - prevents accidental imports from the repo root.
2. **Classical ML is the serving model.** The BiLSTM and XGBoost were measured and were not better (`docs/EXPERIMENTS.md`).
3. **Refund and cancellation rules are deterministic code, not model judgment.** Money logic must be testable and
   auditable. Every refund decision records the rules version that produced it. The numbers are settings and are
   DEMO defaults until `BUSINESS_RULES_CONFIRMED=true`.
4. **A person always decides refunds and cancellations.** Human approval is a LangGraph interrupt with a checkpointer
   (`SqliteSaver`, a file), so paused tickets survive a restart (verified with a real server restart). Several server
   processes would need a PostgreSQL saver **(planned)**. When a ticket resumes, the paused node re-runs from its
   start, so code before `interrupt()` must have no side effects.
5. **Two-step decisions are idempotent.** Apply the decision (refund status, or order cancelled with approver and
   time), then resume the ticket. A crash between the steps is safe to retry. The cancellation rule is re-checked
   when approved, because the order may have shipped while the ticket waited.
6. **Failures degrade to a person.** An agent that still fails after its retries does not crash the ticket: an error
   handler records it (details only in the logs) and the Supervisor hands over to a human.
7. **FAISS behind a `Retriever` interface**; a semantic embedder or pgvector can replace the lexical one without
   touching agents **(planned)**.
8. **Owner checks look the same for "wrong email" and "no such order"**, so callers cannot probe which orders exist.
9. **Secrets only through environment variables** (`pydantic-settings`, `SecretStr`); `.env` is git-ignored.
10. **Data honesty.** Synthetic data is labeled; the test splits and the holdout probe sets were each scored once.
    Scores come with their caveats in `docs/EVALUATION.md`.
11. **Hugging Face / PyTorch deferred**: large, and only needed for semantic embeddings or a transformer model.

## 4. Agents (what each really does)

| Step | Input | Output | Implementation |
|---|---|---|---|
| Supervisor | `TicketState` | next step, escalation | rule-based function `decide_next`; no LLM |
| query_understanding | message | intent + confidence | trained TF-IDF + LogReg (default file `intent_tfidf_lr_bitext_aug.joblib`) |
| sentiment | message | label, urgency | keyword stub (`stubs.stub_sentiment`) |
| knowledge | question | 1-2 policy passages with source and heading | lexical retrieval; empty result escalates |
| order | order number + email | order summary (status, tracking) | database, owner-checked |
| refund | order, message | eligibility, amount, reason codes; creates a PENDING request when allowed | `services.refund` rules + `RefundWorkflow` |
| human_approval | what is being decided | the person's decision | LangGraph `interrupt` |
| respond | all state | reply | grounded templates: only retrieved text and order facts are quoted |

A cancellation has no separate agent: the Supervisor applies the pure cancellation rule to the order summary,
answers a denial at once, and otherwise pauses for approval; the order is changed only after the human approves.

## 5. HTTP API (see `README.md` for the table)

Customer: `POST /tickets`, `POST /tickets/status`, `POST /orders/lookup`, `POST /refunds/eligibility`,
`POST /refunds/requests`. Staff (`X-API-Key`, personal keys `STAFF_API_KEYS` or the shared `STAFF_API_KEY`):
`GET /staff/tickets/awaiting`, `POST /staff/tickets/{id}/decision`, `GET /staff/refunds/pending`,
`POST /staff/refunds/{id}/approve|reject`, `GET /staff/model-info`, `GET /staff/audit`. Health: `GET /health`, `GET /health/ready`.
Development only: `GET /demo`, `/docs`. There is no `/chat` and no `/approvals`. Middleware: request id on every
log line, access log without query strings, per-client rate limits with a staff lockout after wrong keys.

## 6. Data model

`customers`, `orders` (status, `cancelled_at`, `cancelled_by`), `order_items`, `shipments`, `refund_requests`
(status pending_approval / approved / rejected, `decided_by`, `decided_at`, `decision_note`), `tickets` (status
open / awaiting_approval / completed, link to its refund request, customer email for owner checks, reply,
escalation reason). The ticket id is also the workflow's thread id. Money is integer cents; timestamps are
timezone-aware UTC. `audit_logs` (append-only: actor, action, target, JSON detail with ids and numbers only, never customer emails; written
in the same transaction as the change it records) and `staff_keys` (SHA-256 hash of each key, optional expiry,
revocation time; keys can be created, rotated and revoked without a restart). Schema changes go through Alembic
(`0001` initial, `0002` cancellation columns, `0003` audit log and staff keys); a test checks the migrations match the models. There is no separate approvals table: decisions live on the refund request
and on the order.

## 7. Testing strategy (as built)

- Unit tests (about 300) on in-memory or file SQLite: rules, services, graph, supervisor, adapters, ticket service,
  API through `TestClient`, migrations, observability, rate limiting, staff keys, retrieval, training.
- The agent tests use a fake intent model so routing does not depend on model quirks; real-model behavior is checked by
  `ml.probe_eval`, `rag.evaluate` and manual end-to-end runs.
- `tests/integration/` holds PostgreSQL race tests for cancellation and refund requests. They are skipped unless
  `TEST_POSTGRES_URL` is set and have NOT been run yet.
- No test calls an external API.

## 8. Planned, not built

LLM reply generation and its `llm` wrapper (needs an API key); semantic embeddings and hybrid retrieval; a real
sentiment model; priority, routing and resolution-time models; `ticket_messages`; audit of failed staff logins and
of customer-side events beyond refund requests; a write-once audit sink; PostgreSQL checkpointer for several
processes; HTTPS and stronger customer authentication; an authenticated approver for the shared key; payment, inventory and notification integrations; Docker image and PostgreSQL runs
(the files exist but were never built or run). See `PROJECT_STATUS.md`.
