# Deployment, PostgreSQL, rollback and recovery

**STATUS: NOTHING IN THIS FILE HAS BEEN RUN.** Docker and PostgreSQL are not installed on the development machine,
so the image, the compose file, the PostgreSQL migration, row locking and the outbox's `SKIP LOCKED` delivery are
all unverified. These are the exact commands to verify them; record the real results in `PROJECT_STATUS.md` before
trusting any of it.

## 1. Prerequisites
- Docker Desktop (Windows: needs WSL2) or Docker Engine + the compose plugin. Check: `docker --version`, `docker compose version`.
- A trained intent model at `artifacts/models/intent_tfidf_lr_bitext_aug.joblib` (not baked into the image; mounted read-only).
  Train it with the commands in `README.md`.
- A `.env` file (copy `.env.example`). For compose at least `POSTGRES_PASSWORD` and a staff key (`STAFF_API_KEYS` preferred).

## 2. Build and start (single host)
```powershell
docker build -f docker/Dockerfile.api -t supportx-api .
docker compose up --build -d
docker compose ps                    # db should become "healthy", api "running"
docker compose logs api --tail 50    # should show the migrations finishing, then uvicorn start-up
```
Check:
```powershell
curl http://localhost:8000/health          # process is up
curl http://localhost:8000/health/ready    # database reachable
```
`APP_ENV` defaults to `production` in compose: docs and `/demo` are off, tables come from migrations, and start-up logs
warnings for unconfirmed business rules, unset `ALLOWED_HOSTS` and `HSTS_ENABLED=false`.

Demo orders (demo only, never in real production): `scripts/` is not copied into the image, so run the seed from the
host against the published database port (add a `ports: ["5432:5432"]` entry to the `db` service temporarily):
```powershell
$env:DATABASE_URL = "postgresql+psycopg://supportx:<POSTGRES_PASSWORD>@localhost:5432/supportx"
$env:PYTHONPATH = "src"; python scripts/seed_demo.py
```

## 3. PostgreSQL verification checklist
1. Migrations: `docker compose exec api alembic current` shows `0005 (head)`. On a scratch database, `alembic downgrade base`
   then `alembic upgrade head` works.
2. Race tests (skipped everywhere else): create a DISPOSABLE database and run
   ```powershell
   $env:TEST_POSTGRES_URL = "postgresql+psycopg://supportx:pw@localhost:5432/supportx_test"
   pytest tests/integration -v
   ```
   Expect the 2 skipped tests to run: concurrent refund requests and concurrent cancellation for one order.
3. Outbox on PostgreSQL: set `INTEGRATIONS_MODE=mock`, cancel an order, run two workers at once
   (`python -m supportx.integrations.worker --once` in two shells); each event must be delivered exactly once
   (`GET /staff/outbox?status=sent`).
4. Restart recovery: submit a refund ticket (it pauses), `docker compose restart api`, then
   `GET /staff/tickets/awaiting` still lists it and approving it completes it. Paused tickets live in the SQLite
   checkpoint file in the `apistate` volume; with SEVERAL API containers this needs a PostgreSQL checkpointer (not built).
5. Persistence: `docker compose down` (no `-v`) then `up`: orders and tickets remain.

## 4. HTTPS
The app does not terminate TLS. Put a reverse proxy in front (`deploy/Caddyfile.example`, untested), then set
`ALLOWED_HOSTS=<your domain>` and `HSTS_ENABLED=true`. See `docs/SECURITY.md`.

## 5. Configuration that must be real before production
- `BUSINESS_RULES_CONFIRMED=true` only after `REFUND_RETURN_WINDOW_DAYS` / `REFUND_NOT_RECEIVED_AFTER_DAYS` are your real policy.
- Personal staff keys or database keys (`python -m supportx.services.staff_keys create <name> --days 90`); avoid the shared key.
- `INTEGRATIONS_MODE`: keep `disabled` until a real provider exists (`live` refuses to start). `mock` is for demos only.
- `RETRIEVAL_MIN_SCORE` and `INTENT_MIN_CONFIDENCE` are staffing trade-offs (see `docs/EVALUATION.md`).

## 6. Backup
- Database: `docker compose exec db pg_dump -U supportx supportx > backup.sql` (also before every migration).
- Paused tickets: the `apistate` volume (`/app/state/supportx_checkpoints.db`). Stop the API before copying it.
- Models: `artifacts/models/` (the file hash is logged at start-up and shown at `GET /staff/model-info`).

## 7. Rollback
- Application: redeploy the previous image tag. Keep the previous image.
- Schema: migrations `0002`-`0005` only ADD columns and tables, so the previous app version keeps working on a newer schema.
  If you must go back: take a `pg_dump` first, then `docker compose exec api alembic downgrade 0004` (drops the triage
  columns and their data). Downgrades have been tested on SQLite only.
- Model: point `INTENT_MODEL_PATH` back to the previous file and restart; confirm the hash at `/staff/model-info`.
- Business rules: set the previous `BUSINESS_RULES_VERSION` and numbers and restart; every decision records its version.

## 8. Recovery
- Outbox events stuck or dead: `GET /staff/outbox/stats` (`needs_attention`), then `GET /staff/outbox?status=dead`; fix the
  cause and `POST /staff/outbox/{id}/retry`. A stopped worker shows as a growing `oldest_pending_age_seconds`.
- Lost checkpoint file: paused tickets can no longer be resumed through the ticket endpoint. The refund requests are still
  in the database and can be decided with `POST /staff/refunds/{id}/approve|reject`.
- Compromised staff key: `python -m supportx.services.staff_keys revoke <name>` (immediate, no restart); environment keys
  need a changed `.env` and a restart. Review `GET /staff/audit`.
- A model or rule change that misbehaves: roll back as in section 7 and review recent tickets through the audit log.
