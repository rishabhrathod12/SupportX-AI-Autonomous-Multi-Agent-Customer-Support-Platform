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


class TicketStatus(StrEnum):
    OPEN = "open"  # being processed (a ticket stuck here means the process died mid-run)
    AWAITING_APPROVAL = "awaiting_approval"  # paused: a person must decide
    COMPLETED = "completed"  # the customer has a reply (an automated answer or a handoff)


class RefundStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"  # waiting for a human decision
    APPROVED = "approved"
    REJECTED = "rejected"
