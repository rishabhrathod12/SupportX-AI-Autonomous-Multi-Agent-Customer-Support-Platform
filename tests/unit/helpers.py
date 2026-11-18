"""Shared building blocks for the ticket tests (not a test module)."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import RetryPolicy
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from supportx.agents import TicketService, build_graph, build_real_agents
from supportx.core import ExternalServiceError
from supportx.db import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    Shipment,
    init_db,
    session_scope,
)
from supportx.rag import Retriever

POLICIES = Path(__file__).resolve().parents[2] / "data" / "sample" / "policies"
EMAIL = "sam@example.com"
FAST_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.01, retry_on=ExternalServiceError)


class FakeIntentModel:
    """Deterministic stand-in so routing does not depend on a trained model's quirks."""

    RULES = (
        ("refund", ("get_refund", 0.95)),
        ("cancel", ("cancel_order", 0.95)),
        ("where is my order", ("track_order", 0.9)),
        ("return policy", ("check_refund_policy", 0.9)),
    )

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        for keyword, answer in self.RULES:
            if keyword in text.lower():
                return answer
        return ("unknown", 0.1)


def make_engine() -> Engine:
    """In-memory SQLite shared between threads, seeded with two delivered orders for EMAIL."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    now = datetime.now(UTC)
    with session_scope(engine) as session:
        customer = Customer(email=EMAIL, name="Sam")
        for number, days in (("ORD-1001", 5), ("ORD-2002", 40)):
            session.add(
                Order(
                    order_number=number,
                    customer=customer,
                    status=OrderStatus.DELIVERED,
                    items=[OrderItem(sku="M", name="Mug", unit_price_cents=2500, quantity=2)],
                    shipment=Shipment(
                        carrier="DemoPost",
                        tracking_number=f"TRK-{number}",
                        shipped_at=now - timedelta(days=days + 3),
                        delivered_at=now - timedelta(days=days),
                    ),
                )
            )
    return engine


def sqlite_saver(path: Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection), connection


def make_service(
    engine: Engine, retriever: Retriever, saver: BaseCheckpointSaver[Any]
) -> TicketService:
    agents = build_real_agents(FakeIntentModel(), retriever, engine)
    return TicketService(engine, build_graph(agents, saver, retry=FAST_RETRY))
