"""Real components behind the AgentSet interface: trained intent model, database services.

Still stand-ins (documented, not hidden): the sentiment agent is the keyword stub, because no
sentiment model has been trained yet (the data decision is pending in docs/DATASETS.md).
"""

import re
from collections.abc import Callable
from typing import Any, Protocol

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from supportx.agents.ports import AgentSet
from supportx.agents.replies import compose_reply
from supportx.agents.stubs import stub_sentiment
from supportx.core import ExternalServiceError, get_logger
from supportx.db import RefundReason, session_scope
from supportx.rag import Retriever
from supportx.services import OrderService, RefundPolicy, RefundWorkflow

logger = get_logger(__name__)


class IntentModel(Protocol):
    """What the intent agent needs from a model (IntentClassifier satisfies this)."""

    def predict_with_confidence(self, text: str) -> tuple[str, float]: ...


def make_intent_agent(model: IntentModel) -> Callable[[str], tuple[str, float]]:
    def understand(message: str) -> tuple[str, float]:
        return model.predict_with_confidence(message)

    return understand


_REASON_PATTERNS: tuple[tuple[RefundReason, re.Pattern[str]], ...] = (
    (
        RefundReason.NOT_RECEIVED,
        re.compile(
            r"not (yet )?(arrived|received|delivered)|never (arrived|came|received)|"
            r"(hasn't|has not|haven't|have not|didn't|did not) (arrive|come)|missing parcel",
            re.I,
        ),
    ),
    (
        RefundReason.DAMAGED,
        re.compile(r"damag|broken|broke|cracked|defect|faulty|smashed|torn", re.I),
    ),
    (
        RefundReason.NOT_AS_DESCRIBED,
        re.compile(
            r"not as described|wrong (item|size|colou?r)|different from|doesn't match", re.I
        ),
    ),
)


def infer_refund_reason(message: str) -> RefundReason:
    """Guess WHY the customer wants a refund from their words (rules, not a model).

    If nothing matches we assume CHANGED_MIND, the most neutral reason: it follows the same
    return window as damage claims and needs no extra evidence. A human still reviews every refund.
    """
    for reason, pattern in _REASON_PATTERNS:
        if pattern.search(message):
            return reason
    return RefundReason.CHANGED_MIND


def make_database_agents(
    engine: Engine, policy: RefundPolicy | None = None
) -> tuple[Callable[[str, str], dict[str, Any]], Callable[[str, str, str], dict[str, Any]]]:
    """(lookup_order, check_refund) backed by the real services.

    check_refund has a side effect: when the rules allow a refund it CREATES a pending request
    that a person must decide. The transaction commits before the function returns, so a retry
    can only happen when nothing was saved. Database trouble becomes ExternalServiceError, which
    the graph retries; "no such order" (NotFoundError) is passed through untouched.
    """

    def lookup_order(order_number: str, email: str) -> dict[str, Any]:
        try:
            with session_scope(engine) as session:
                summary = OrderService(session).get_order_summary(order_number, email)
                return summary.model_dump(mode="json")
        except SQLAlchemyError as exc:
            logger.error("Order lookup failed: %s", type(exc).__name__)
            raise ExternalServiceError("The order database is unavailable.") from exc

    def check_refund(order_number: str, email: str, message: str) -> dict[str, Any]:
        reason = infer_refund_reason(message)
        try:
            with session_scope(engine) as session:
                decision, request = RefundWorkflow(session, policy).request_refund(
                    order_number, email, reason
                )
                return {
                    **decision.model_dump(mode="json"),
                    "reason": reason.value,
                    "request_id": request.id if request else None,
                }
        except SQLAlchemyError as exc:
            logger.error("Refund check failed: %s", type(exc).__name__)
            raise ExternalServiceError("The refund database is unavailable.") from exc

    return lookup_order, check_refund


def build_real_agents(
    intent_model: IntentModel,
    retriever: Retriever,
    engine: Engine,
    policy: RefundPolicy | None = None,
    min_score: float = 0.1,
) -> AgentSet:
    """The production agent set: trained intent model, real retriever, real database services."""
    lookup_order, check_refund = make_database_agents(engine, policy)

    def search_policy(question: str) -> list[dict[str, Any]]:
        return [
            {
                "source": hit.chunk.source,
                "heading": hit.chunk.heading,
                "text": hit.chunk.text,
                "score": round(hit.score, 4),
            }
            for hit in retriever.retrieve(question, k=3, min_score=min_score)
        ]

    return AgentSet(
        understand=make_intent_agent(intent_model),
        analyze_sentiment=stub_sentiment,  # TODO: replace once a sentiment model exists
        search_policy=search_policy,
        lookup_order=lookup_order,
        check_refund=check_refund,
        compose_reply=compose_reply,
    )
