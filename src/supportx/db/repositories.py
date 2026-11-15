"""Database access in one place: queries live here, business rules live in services."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from supportx.db.enums import RefundReason, RefundStatus
from supportx.db.models import Order, RefundRequest


def to_utc(value: datetime | None) -> datetime | None:
    """Make a datetime timezone-aware UTC. SQLite returns naive values; PostgreSQL does not."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def normalize_order_number(order_number: str) -> str:
    return order_number.strip().upper()


class OrderRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_number(self, order_number: str, *, for_update: bool = False) -> Order | None:
        """Load an order with its customer, items, shipment and refund requests.

        `for_update=True` locks the order row until the transaction ends (PostgreSQL), so two
        simultaneous refund requests for one order cannot both pass the "one pending" rule.
        SQLite ignores the lock.
        """
        stmt = (
            select(Order)
            .where(Order.order_number == normalize_order_number(order_number))
            .options(
                selectinload(Order.customer),
                selectinload(Order.items),
                selectinload(Order.shipment),
                selectinload(Order.refund_requests),
            )
        )
        if for_update:
            # populate_existing: overwrite anything already loaded in this session with fresh data
            stmt = stmt.with_for_update(of=Order).execution_options(populate_existing=True)
        return self._session.scalars(stmt).one_or_none()


class RefundRepository:
    def __init__(self, session: Session):
        self._session = session

    def add(self, order: Order, amount_cents: int, reason: RefundReason) -> RefundRequest:
        """Create a refund request waiting for a human decision."""
        request = RefundRequest(
            order=order,
            amount_cents=amount_cents,
            reason=reason,
            status=RefundStatus.PENDING_APPROVAL,
        )
        self._session.add(request)
        self._session.flush()  # assigns request.id
        return request

    def get(self, refund_id: int, *, for_update: bool = False) -> RefundRequest | None:
        stmt = (
            select(RefundRequest)
            .where(RefundRequest.id == refund_id)
            .options(
                selectinload(RefundRequest.order).selectinload(Order.items),
                selectinload(RefundRequest.order).selectinload(Order.shipment),
                selectinload(RefundRequest.order).selectinload(Order.refund_requests),
            )
        )
        if for_update:
            stmt = stmt.with_for_update(of=RefundRequest).execution_options(populate_existing=True)
        return self._session.scalars(stmt).one_or_none()

    def list_pending(self) -> list[RefundRequest]:
        stmt = (
            select(RefundRequest)
            .where(RefundRequest.status == RefundStatus.PENDING_APPROVAL)
            .order_by(RefundRequest.created_at, RefundRequest.id)
        )
        return list(self._session.scalars(stmt))
