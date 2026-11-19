"""Append-only audit trail of sensitive actions (who did what to which record, and when).

Rows are written in the SAME database transaction as the change they describe, so a change and its
audit row are saved together or not at all. There is no update or delete function on purpose.

PRIVACY: callers pass ids, order numbers and staff names, never customer emails or message text.
A staff member's decision note is stored (truncated) because the reason is part of the record.
LIMITS: this is application-level logging in the same database; someone with direct database write
access can still alter it. Use database permissions / a write-once sink for stronger guarantees.
"""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from supportx.db.models import AuditLog

MAX_DETAIL_TEXT = 500


def record_audit(
    session: Session,
    actor: str,
    action: str,
    target_type: str,
    target_id: str | int,
    detail: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> AuditLog:
    """Add one audit row to the caller's transaction (the caller commits)."""
    cleaned = {
        k: (v[:MAX_DETAIL_TEXT] if isinstance(v, str) else v) for k, v in (detail or {}).items()
    }
    row = AuditLog(
        occurred_at=now or datetime.now(UTC),
        actor=actor[:120],
        action=action,
        target_type=target_type,
        target_id=str(target_id),
        detail=json.dumps(cleaned, sort_keys=True, default=str) if cleaned else None,
    )
    session.add(row)
    session.flush()
    return row


def list_audit(
    session: Session,
    limit: int = 50,
    action: str | None = None,
    target_id: str | None = None,
) -> list[AuditLog]:
    """Newest first."""
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(max(1, min(limit, 500)))
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if target_id:
        stmt = stmt.where(AuditLog.target_id == target_id)
    return list(session.scalars(stmt))
