"""Database layer: ORM models, enums, engine/session helpers."""

from supportx.db.enums import OrderStatus, RefundReason, RefundStatus, TicketStatus
from supportx.db.models import (
    Base,
    Customer,
    Order,
    OrderItem,
    RefundRequest,
    Shipment,
    Ticket,
)
from supportx.db.session import init_db, make_engine, session_scope

__all__ = [
    "Base",
    "Customer",
    "Order",
    "OrderItem",
    "OrderStatus",
    "RefundReason",
    "RefundRequest",
    "RefundStatus",
    "Shipment",
    "Ticket",
    "TicketStatus",
    "init_db",
    "make_engine",
    "session_scope",
]
