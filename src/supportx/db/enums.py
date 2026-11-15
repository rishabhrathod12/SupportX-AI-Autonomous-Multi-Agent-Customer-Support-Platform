"""Domain enums shared by the database models and the business-rule services."""

from enum import StrEnum


class OrderStatus(StrEnum):
    PLACED = "placed"  # paid, not shipped yet
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class RefundReason(StrEnum):
    DAMAGED = "damaged"
    NOT_AS_DESCRIBED = "not_as_described"
    CHANGED_MIND = "changed_mind"
    NOT_RECEIVED = "not_received"


class RefundStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"  # waiting for a human decision
    APPROVED = "approved"
    REJECTED = "rejected"
