"""Order lookup for the Order Management Agent. Read-only, and only for the order's owner."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from supportx.core import NotFoundError
from supportx.db.enums import OrderStatus, RefundStatus
from supportx.db.models import Order
from supportx.db.repositories import OrderRepository, to_utc


class ItemSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    quantity: int
    unit_price_cents: int
    final_sale: bool


class ShipmentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    carrier: str
    tracking_number: str
    shipped_at: datetime | None
    delivered_at: datetime | None


class OrderSummary(BaseModel):
    """What an agent may tell a customer about their order."""

    model_config = ConfigDict(frozen=True)

    order_number: str
    status: OrderStatus
    placed_at: datetime
    items: tuple[ItemSummary, ...]
    total_cents: int
    shipment: ShipmentSummary | None
    has_pending_refund: bool


def ensure_owner(order: Order | None, customer_email: str) -> Order:
    """Return the order only if it exists and belongs to this customer.

    A wrong email gives the SAME error as a missing order, so callers cannot probe which
    order numbers exist.
    """
    if order is None or order.customer.email.casefold() != customer_email.strip().casefold():
        raise NotFoundError("No order with that number was found for this customer.")
    return order


def find_order_for_customer(session: Session, order_number: str, customer_email: str) -> Order:
    return ensure_owner(OrderRepository(session).get_by_number(order_number), customer_email)


class OrderService:
    def __init__(self, session: Session):
        self._session = session

    def get_order_summary(self, order_number: str, customer_email: str) -> OrderSummary:
        order = find_order_for_customer(self._session, order_number, customer_email)
        shipment = order.shipment
        return OrderSummary(
            order_number=order.order_number,
            status=order.status,
            placed_at=to_utc(order.placed_at) or order.placed_at,
            items=tuple(
                ItemSummary(
                    name=i.name,
                    quantity=i.quantity,
                    unit_price_cents=i.unit_price_cents,
                    final_sale=i.final_sale,
                )
                for i in order.items
            ),
            total_cents=sum(i.unit_price_cents * i.quantity for i in order.items),
            shipment=None
            if shipment is None
            else ShipmentSummary(
                carrier=shipment.carrier,
                tracking_number=shipment.tracking_number,
                shipped_at=to_utc(shipment.shipped_at),
                delivered_at=to_utc(shipment.delivered_at),
            ),
            has_pending_refund=any(
                r.status is RefundStatus.PENDING_APPROVAL for r in order.refund_requests
            ),
        )
