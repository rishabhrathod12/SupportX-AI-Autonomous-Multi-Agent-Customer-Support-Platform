"""Transactional outbox: follow-up work (refund payment, stock release, customer message) is saved
in the SAME database transaction as the business change that causes it, then delivered later by a
worker. So a change is never saved without its follow-up, and a provider outage never blocks or
undoes a human's decision.

Delivery is AT LEAST ONCE: a crash after the provider acted but before the row was marked sent
means the event is delivered again. That is safe only because every event carries an idempotency
key that the provider must honour (see supportx.integrations.ports).

Lifecycle of an event:  PENDING --success--> SENT
                        PENDING --transient failure--> PENDING (later, exponential backoff)
                        PENDING --permanent failure, or too many attempts--> DEAD (a person decides:
                        fix the cause, then requeue from `POST /staff/outbox/{id}/retry`)

Each outcome writes an audit row. Error text kept on the event is the exception type and a short
message only (no payloads).
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from supportx.core import NotFoundError, WorkflowError, get_logger
from supportx.db.enums import OutboxStatus
from supportx.db.models import Order, OutboxEvent
from supportx.db.repositories import to_utc
from supportx.db.session import session_scope
from supportx.integrations.ports import (
    InventoryService,
    Notifier,
    PaymentProvider,
    PermanentProviderError,
    ProviderResult,
    StockLine,
    TransientProviderError,
)
from supportx.services.audit import record_audit

logger = get_logger(__name__)

PAYMENT_REFUND = "payment.refund"
STOCK_RELEASE = "inventory.release"
NOTIFY = "notification.send"


# ------------------------------------------------ writing events (inside the caller's transaction)
def enqueue(
    session: Session,
    event_type: str,
    idempotency_key: str,
    payload: dict[str, Any],
    now: datetime | None = None,
) -> OutboxEvent:
    """Add an event to the caller's transaction (the caller commits). Enqueueing the same
    idempotency key twice returns the first event and adds nothing."""
    existing = session.scalars(
        select(OutboxEvent).where(OutboxEvent.idempotency_key == idempotency_key)
    ).one_or_none()
    if existing is not None:
        return existing
    moment = now or datetime.now(UTC)
    row = OutboxEvent(
        event_type=event_type,
        idempotency_key=idempotency_key,
        payload=json.dumps(payload, sort_keys=True),
        status=OutboxStatus.PENDING,
        attempts=0,
        created_at=moment,
        next_attempt_at=moment,
    )
    session.add(row)
    session.flush()
    return row


def enqueue_refund_payment(
    session: Session, refund_id: int, order_number: str, amount_cents: int
) -> OutboxEvent:
    return enqueue(
        session,
        PAYMENT_REFUND,
        f"refund-{refund_id}-payment",
        {"order_number": order_number, "amount_cents": amount_cents},
    )


def enqueue_refund_notice(
    session: Session, refund_id: int, customer_id: int, order_number: str, approved: bool
) -> OutboxEvent:
    template = "refund_approved" if approved else "refund_rejected"
    return enqueue(
        session,
        NOTIFY,
        f"refund-{refund_id}-notice-{template}",
        {
            "customer_id": customer_id,
            "template": template,
            "params": {"order_number": order_number},
        },
    )


def enqueue_cancellation_effects(session: Session, order: Order) -> None:
    """A cancelled (paid, unshipped) order: refund the money, free the stock, tell the customer."""
    number = order.order_number
    total = sum(i.unit_price_cents * i.quantity for i in order.items)
    enqueue(
        session,
        PAYMENT_REFUND,
        f"cancel-{number}-payment",
        {"order_number": number, "amount_cents": total},
    )
    enqueue(
        session,
        STOCK_RELEASE,
        f"cancel-{number}-stock",
        {
            "order_number": number,
            "lines": [{"sku": i.sku, "quantity": i.quantity} for i in order.items],
        },
    )
    enqueue(
        session,
        NOTIFY,
        f"cancel-{number}-notice",
        {
            "customer_id": order.customer_id,
            "template": "order_cancelled",
            "params": {"order_number": number},
        },
    )


# ---------------------------------------------------------------- delivering events
@dataclass
class Providers:
    """Whichever adapters are configured; an event whose adapter is missing stays PENDING."""

    payments: PaymentProvider | None = None
    inventory: InventoryService | None = None
    notifier: Notifier | None = None


@dataclass
class DispatchSummary:
    sent: int = 0
    retried: int = 0
    dead: int = 0
    skipped: int = 0  # no adapter configured for the event type, or locked by another worker
    errors: list[str] = field(default_factory=list)


class OutboxDispatcher:
    def __init__(
        self,
        engine: Engine,
        providers: Providers,
        max_attempts: int = 5,
        base_backoff_seconds: float = 30.0,
        max_backoff_seconds: float = 3600.0,
    ) -> None:
        self._engine = engine
        self._providers = providers
        self._max_attempts = max_attempts
        self._base = base_backoff_seconds
        self._cap = max_backoff_seconds

    def run_once(self, now: datetime | None = None, batch: int = 20) -> DispatchSummary:
        """Deliver every due event (up to `batch`). Each event has its own transaction, so one
        failure never rolls back another."""
        moment = now or datetime.now(UTC)
        with session_scope(self._engine) as session:
            due = list(
                session.scalars(
                    select(OutboxEvent.id)
                    .where(OutboxEvent.status == OutboxStatus.PENDING)
                    .where(OutboxEvent.next_attempt_at <= moment)
                    .order_by(OutboxEvent.id)
                    .limit(batch)
                )
            )
        summary = DispatchSummary()
        for event_id in due:
            self._deliver(event_id, moment, summary)
        return summary

    def _deliver(self, event_id: int, now: datetime, summary: DispatchSummary) -> None:
        with session_scope(self._engine) as session:
            # skip_locked: if another worker holds this row, leave it to them (PostgreSQL only)
            row = session.scalars(
                select(OutboxEvent)
                .where(OutboxEvent.id == event_id)
                .with_for_update(skip_locked=True, of=OutboxEvent)
            ).one_or_none()
            if row is None or row.status is not OutboxStatus.PENDING:
                summary.skipped += 1
                return
            try:
                result = self._call_provider(row)
            except _NoAdapter:
                summary.skipped += 1
                return
            except PermanentProviderError as exc:
                self._fail(session, row, now, str(exc), permanent=True, summary=summary)
            except TransientProviderError as exc:
                self._fail(session, row, now, str(exc), permanent=False, summary=summary)
            except Exception as exc:  # a bug or an unexpected provider error: treat as retryable
                logger.exception("Unexpected error delivering outbox event %s", event_id)
                self._fail(session, row, now, type(exc).__name__, permanent=False, summary=summary)
            else:
                row.attempts += 1
                row.status = OutboxStatus.SENT
                row.provider_reference = result.reference
                row.is_mock = result.is_mock
                row.processed_at = now
                row.last_error = None
                record_audit(
                    session,
                    "system",
                    "outbox.sent",
                    "outbox_event",
                    row.id,
                    {"type": row.event_type, "reference": result.reference, "mock": result.is_mock},
                    now,
                )
                summary.sent += 1

    def _fail(
        self,
        session: Session,
        row: OutboxEvent,
        now: datetime,
        message: str,
        permanent: bool,
        summary: DispatchSummary,
    ) -> None:
        row.attempts += 1
        row.last_error = message[:200]
        if permanent or row.attempts >= self._max_attempts:
            row.status = OutboxStatus.DEAD
            record_audit(
                session,
                "system",
                "outbox.dead",
                "outbox_event",
                row.id,
                {"type": row.event_type, "attempts": row.attempts, "permanent": permanent},
                now,
            )
            logger.error("Outbox event %s is DEAD after %d attempt(s)", row.id, row.attempts)
            summary.dead += 1
        else:
            delay = min(self._cap, self._base * 2 ** (row.attempts - 1))
            row.next_attempt_at = now + timedelta(seconds=delay)
            summary.retried += 1

    def _call_provider(self, row: OutboxEvent) -> ProviderResult:
        payload = json.loads(row.payload)
        key = row.idempotency_key
        if row.event_type == PAYMENT_REFUND and self._providers.payments:
            return self._providers.payments.refund(
                key, payload["order_number"], payload["amount_cents"]
            )
        if row.event_type == STOCK_RELEASE and self._providers.inventory:
            lines = [StockLine(line["sku"], line["quantity"]) for line in payload["lines"]]
            return self._providers.inventory.release(key, payload["order_number"], lines)
        if row.event_type == NOTIFY and self._providers.notifier:
            return self._providers.notifier.send(
                key, payload["customer_id"], payload["template"], payload["params"]
            )
        raise _NoAdapter


class _NoAdapter(Exception):
    """No adapter is configured for this event type: leave the event waiting."""


# ---------------------------------------------------------------- operator actions
def requeue(
    session: Session, event_id: int, actor: str, now: datetime | None = None
) -> OutboxEvent:
    """Give a DEAD event another chance (after the cause was fixed). The caller commits."""
    row = session.get(OutboxEvent, event_id)
    if row is None:
        raise NotFoundError(f"Outbox event {event_id} does not exist.")
    if row.status is not OutboxStatus.DEAD:
        raise WorkflowError(f"Outbox event {event_id} is {row.status.value}, not dead.")
    moment = now or datetime.now(UTC)
    row.status = OutboxStatus.PENDING
    row.attempts = 0
    row.next_attempt_at = moment
    row.last_error = None
    record_audit(
        session, actor, "outbox.requeued", "outbox_event", row.id, {"type": row.event_type}, moment
    )
    session.flush()
    return row


def list_events(
    session: Session, status: OutboxStatus | None = None, limit: int = 50
) -> list[OutboxEvent]:
    stmt = select(OutboxEvent).order_by(OutboxEvent.id.desc()).limit(max(1, min(limit, 500)))
    if status is not None:
        stmt = stmt.where(OutboxEvent.status == status)
    return list(session.scalars(stmt))


def event_age_seconds(row: OutboxEvent, now: datetime | None = None) -> float:
    created = to_utc(row.created_at) or row.created_at
    return ((now or datetime.now(UTC)) - created).total_seconds()
