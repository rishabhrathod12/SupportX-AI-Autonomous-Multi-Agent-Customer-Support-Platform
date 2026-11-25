"""SQLAlchemy 2.0 ORM models.

All money is stored as integer cents (no floats, no rounding surprises).
Timestamps are timezone-aware UTC. PostgreSQL keeps the timezone; SQLite (used in tests)
returns naive datetimes, so the repository layer must normalize them to UTC.
"""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from supportx.db.enums import OrderStatus, OutboxStatus, RefundReason, RefundStatus, TicketStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _enum[E: StrEnum](enum_cls: type[E]) -> Enum:
    """Store enums as their lowercase string values in a plain VARCHAR (portable, readable)."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=20,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
    )


class Base(DeclarativeBase):
    """Base class for all models."""


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(40), unique=True, index=True)  # e.g. ORD-12345
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus), default=OrderStatus.PLACED)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_by: Mapped[str | None] = mapped_column(String(120))  # the approver

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )
    shipment: Mapped["Shipment | None"] = relationship(
        back_populates="order", uselist=False, cascade="all, delete-orphan"
    )
    refund_requests: Mapped[list["RefundRequest"]] = relationship(back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_items_quantity_positive"),
        CheckConstraint("unit_price_cents >= 0", name="ck_order_items_price_nonnegative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    sku: Mapped[str] = mapped_column(String(60))
    name: Mapped[str] = mapped_column(String(200))
    unit_price_cents: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    final_sale: Mapped[bool] = mapped_column(default=False)  # not refundable once sold

    order: Mapped[Order] = relationship(back_populates="items")


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), unique=True)
    carrier: Mapped[str] = mapped_column(String(60))
    tracking_number: Mapped[str] = mapped_column(String(80))
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    order: Mapped[Order] = relationship(back_populates="shipment")


class Ticket(Base):
    """One customer conversation handled by the agent workflow.

    `id` is also the workflow's thread id, so a paused ticket can be found and resumed.
    """

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # random UUID
    customer_email: Mapped[str | None] = mapped_column(String(255), index=True)
    order_number: Mapped[str | None] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[TicketStatus] = mapped_column(_enum(TicketStatus))
    approval_kind: Mapped[str | None] = mapped_column(String(20))  # "refund" / "cancellation"
    refund_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("refund_requests.id"), index=True
    )
    response: Mapped[str | None] = mapped_column(Text)
    escalated: Mapped[bool] = mapped_column(default=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(60))
    # Triage (services.triage): known when the ticket is recorded; for staff queues only.
    priority: Mapped[str | None] = mapped_column(String(10))
    team: Mapped[str | None] = mapped_column(String(30))
    triage_source: Mapped[str | None] = mapped_column(String(20))
    triage_needs_review: Mapped[bool] = mapped_column(default=False)
    expected_resolution_hours: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    refund_request: Mapped["RefundRequest | None"] = relationship()


class RefundRequest(Base):
    """A refund proposed by the system. A human must approve or reject it."""

    __tablename__ = "refund_requests"
    __table_args__ = (CheckConstraint("amount_cents > 0", name="ck_refund_amount_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    reason: Mapped[RefundReason] = mapped_column(_enum(RefundReason))
    status: Mapped[RefundStatus] = mapped_column(
        _enum(RefundStatus), default=RefundStatus.PENDING_APPROVAL
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[str | None] = mapped_column(String(120))  # who approved/rejected
    decision_note: Mapped[str | None] = mapped_column(Text)

    order: Mapped[Order] = relationship(back_populates="refund_requests")


class AuditLog(Base):
    """Append-only record of a sensitive action. See supportx.services.audit."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actor: Mapped[str] = mapped_column(String(120), index=True)  # staff name, "customer", "cli"
    action: Mapped[str] = mapped_column(String(60), index=True)  # e.g. "order.cancelled"
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[str | None] = mapped_column(Text)  # JSON object, ids and numbers only


class StaffKey(Base):
    """A staff API key, stored as a SHA-256 hash. See supportx.services.staff_keys."""

    __tablename__ = "staff_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(60), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OutboxEvent(Base):
    """Follow-up work saved with the change that caused it. See supportx.integrations.outbox."""

    __tablename__ = "outbox_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    payload: Mapped[str] = mapped_column(Text)  # JSON: ids, order numbers, amounts; no emails
    status: Mapped[OutboxStatus] = mapped_column(_enum(OutboxStatus), index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(200))
    provider_reference: Mapped[str | None] = mapped_column(String(80))
    is_mock: Mapped[bool] = mapped_column(default=False)  # True: a MOCK adapter handled it
