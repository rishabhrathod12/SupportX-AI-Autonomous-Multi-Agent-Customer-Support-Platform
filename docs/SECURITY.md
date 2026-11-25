# Security notes

What exists, what was tested, and what is still missing. Nothing here has been penetration-tested.

## In place (code + tests)
- Staff auth: `X-API-Key`, constant-time comparison; personal keys (`STAFF_API_KEYS`) and revocable, hashed,
  expiring database keys (`python -m supportx.services.staff_keys`). A personal key fixes the recorded approver.
- Append-only audit log of money and decision events (no customer emails), `GET /staff/audit`.
- Rate limiting per client address and a lockout of `/staff/*` after repeated wrong keys. In memory, per process.
- Failed staff logins are logged (reason, client address, request id; never the key).
- Request limits: `MAX_REQUEST_BYTES` (Content-Length check, default 64 KB), field length limits in the schemas,
  optional `ALLOWED_HOSTS` (Host header allow-list).
- Response headers on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cache-Control: no-store`; `Strict-Transport-Security` only when `HSTS_ENABLED=true`.
- Generic error messages (no stack traces, no SDK or database detail to clients); API docs off in production.
- Secrets are `SecretStr`, never logged; `.env` is git-ignored. Production start-up warns about missing staff keys,
  unset `ALLOWED_HOSTS`, `HSTS_ENABLED=false` and unconfirmed business rules.
- The LLM can only rephrase informational replies behind a grounding check; refunds, cancellations and approvals
  always go through the rule engine and a human decision (see `supportx.llm.replies`).

## HTTPS
The application does NOT terminate TLS. Serve it behind a reverse proxy. `deploy/Caddyfile.example` is an
UNTESTED example; HTTPS has not been verified locally. After the proxy works, set `HSTS_ENABLED=true`.

## Known gaps
- Customers are identified only by order number + email. Proper verification (one-time code by email/SMS) needs a
  real notification provider.
- Rate limits are per process and do not read `X-Forwarded-For`; use a gateway or Redis for a shared limit.
- The size limit does not stop chunked bodies without Content-Length; cap bodies at the proxy as well.
- Audit log is not tamper-proof; failed logins are logged, not stored in the audit table.
- The shared `STAFF_API_KEY` cannot prove who approved; prefer personal or database keys.
- No dependency vulnerability scan was run (`pip-audit` not installed).
