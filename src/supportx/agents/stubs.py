"""Deterministic STAND-IN agents for tests and demos. Not real intelligence.

- understand / analyze_sentiment: simple keyword rules (the real ones use the trained models).
- search_policy: the real Retriever when given one, otherwise "nothing found".
- lookup_order / check_refund: a small in-memory order book (the real ones use the
  database services).
"""

import re
from collections.abc import Mapping
from typing import Any

from supportx.agents.ports import AgentSet
from supportx.agents.replies import compose_reply
from supportx.core import NotFoundError
from supportx.rag import Retriever

_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("get_refund", ("refund", "money back", "damaged")),
    ("cancel_order", ("cancel",)),
    ("track_order", ("where is my order", "track", "status of my order", "has not arrived")),
    ("check_refund_policy", ("return policy", "refund policy", "days to return", "return window")),
    ("delivery_options", ("express", "delivery option", "how long does")),
    ("complaint", ("complaint", "terrible", "unacceptable")),
    ("contact_human_agent", ("human", "real person", "agent")),
)
_NEGATIVE = ("terrible", "unacceptable", "angry", "furious", "worst", "awful", "damaged", "!")
_URGENT = ("urgent", "immediately", "asap", "right now", "today")


def stub_understand(message: str) -> tuple[str, float]:
    text = message.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(k in text for k in keywords):
            return intent, 0.9
    return "unknown", 0.2


def stub_sentiment(message: str) -> tuple[str, str]:
    text = message.lower()
    sentiment = "negative" if any(w in text for w in _NEGATIVE) else "neutral"
    urgency = "high" if any(w in text for w in _URGENT) else "low"
    return sentiment, urgency


def extract_order_number(message: str) -> str | None:
    """Find something like ORD-12345 in free text."""
    match = re.search(r"\bORD-\d+\b", message, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def make_stub_agents(
    retriever: Retriever | None = None,
    orders: Mapping[str, dict[str, Any]] | None = None,
) -> AgentSet:
    """`orders` maps an order number to {"email", "status", "shipment", "refund_eligible", ...}."""
    book = dict(orders or {})

    def search_policy(question: str) -> list[dict[str, Any]]:
        if retriever is None:
            return []
        return [
            {
                "source": hit.chunk.source,
                "heading": hit.chunk.heading,
                "text": hit.chunk.text,
                "score": round(hit.score, 4),
            }
            for hit in retriever.retrieve(question, k=3, min_score=0.1)
        ]

    def lookup_order(order_number: str, email: str) -> dict[str, Any]:
        order = book.get(order_number.upper())
        if order is None or order["email"].casefold() != email.casefold():
            raise NotFoundError("No order with that number was found for this customer.")
        return {
            "order_number": order_number.upper(),
            "status": order["status"],
            "shipment": order.get("shipment"),
        }

    def check_refund(order_number: str, email: str, message: str) -> dict[str, Any]:
        order = book.get(order_number.upper())
        if order is None or order["email"].casefold() != email.casefold():
            raise NotFoundError("No order with that number was found for this customer.")
        eligible = bool(order.get("refund_eligible", False))
        return {
            "eligible": eligible,
            "proposed_cents": order.get("refundable_cents", 0) if eligible else 0,
            "reason_codes": [] if eligible else ["outside_return_window"],
            "explanations": [] if eligible else ["The 30-day return window has passed."],
            "requires_human_approval": True,
        }

    return AgentSet(
        understand=stub_understand,
        analyze_sentiment=stub_sentiment,
        search_policy=search_policy,
        lookup_order=lookup_order,
        check_refund=check_refund,
        compose_reply=compose_reply,
    )
