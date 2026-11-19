"""Business rules come from settings, boundaries hold, and demo numbers are visible to operators."""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from supportx.api import create_app
from supportx.core import Settings
from supportx.db import init_db
from supportx.db.enums import OrderStatus, RefundReason
from supportx.services import (
    DenialCode,
    ItemFacts,
    OrderFacts,
    evaluate_cancellation,
    evaluate_refund,
)
from supportx.services.refund import refund_policy_from_settings

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
ITEMS = (ItemFacts(unit_price_cents=2000, quantity=1),)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


# ---------------------------------------------------------------- numbers come from settings
def test_defaults_are_the_documented_demo_numbers() -> None:
    policy = refund_policy_from_settings(make_settings())
    assert (policy.version, policy.return_window_days, policy.not_received_after_days) == (
        "demo-1",
        30,
        14,
    )
    assert make_settings().business_rules_confirmed is False


def test_settings_change_the_policy_and_the_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFUND_RETURN_WINDOW_DAYS", "10")
    monkeypatch.setenv("REFUND_NOT_RECEIVED_AFTER_DAYS", "21")
    monkeypatch.setenv("BUSINESS_RULES_VERSION", "acme-2026-11")
    policy = refund_policy_from_settings(Settings(_env_file=None))  # type: ignore[call-arg]
    assert (policy.return_window_days, policy.not_received_after_days) == (10, 21)

    delivered_12_days_ago = OrderFacts(
        status=OrderStatus.DELIVERED,
        shipped_at=NOW - timedelta(days=15),
        delivered_at=NOW - timedelta(days=12),
        items=ITEMS,
    )
    decision = evaluate_refund(delivered_12_days_ago, (), RefundReason.DAMAGED, NOW, policy=policy)
    assert decision.reason_codes == (DenialCode.OUTSIDE_RETURN_WINDOW,)
    assert decision.policy_version == "acme-2026-11"  # every decision says which rules applied


@pytest.mark.parametrize("name", ["refund_return_window_days", "refund_not_received_after_days"])
@pytest.mark.parametrize("bad", [0, -5, 100000])
def test_nonsense_numbers_are_rejected_at_startup(name: str, bad: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: bad})


# ---------------------------------------------------------------- edge cases of the rules
def test_a_delivered_order_without_a_delivery_date_is_not_refundable_yet() -> None:
    order = OrderFacts(
        status=OrderStatus.DELIVERED,
        shipped_at=NOW - timedelta(days=5),
        delivered_at=None,
        items=ITEMS,
    )
    decision = evaluate_refund(order, (), RefundReason.DAMAGED, NOW)
    assert not decision.eligible and decision.reason_codes == (DenialCode.ORDER_NOT_DELIVERED,)


def test_not_received_with_no_shipping_date_is_never_approved_automatically() -> None:
    order = OrderFacts(status=OrderStatus.SHIPPED, shipped_at=None, items=ITEMS)
    decision = evaluate_refund(order, (), RefundReason.NOT_RECEIVED, NOW)
    assert decision.reason_codes == (DenialCode.NOT_RECEIVED_TOO_EARLY,)


def test_one_second_either_side_of_both_boundaries() -> None:
    window = timedelta(days=30)
    inside = OrderFacts(
        status=OrderStatus.DELIVERED,
        shipped_at=NOW - window,
        delivered_at=NOW - window,
        items=ITEMS,
    )
    outside = OrderFacts(
        status=OrderStatus.DELIVERED,
        shipped_at=NOW - window,
        delivered_at=NOW - window - timedelta(seconds=1),
        items=ITEMS,
    )
    assert evaluate_refund(inside, (), RefundReason.DAMAGED, NOW).eligible
    assert not evaluate_refund(outside, (), RefundReason.DAMAGED, NOW).eligible

    fourteen = timedelta(days=14)
    just_in = OrderFacts(status=OrderStatus.SHIPPED, shipped_at=NOW - fourteen, items=ITEMS)
    too_early = OrderFacts(
        status=OrderStatus.SHIPPED, shipped_at=NOW - fourteen + timedelta(seconds=1), items=ITEMS
    )
    assert evaluate_refund(just_in, (), RefundReason.NOT_RECEIVED, NOW).eligible
    assert not evaluate_refund(too_early, (), RefundReason.NOT_RECEIVED, NOW).eligible


def test_cancellation_edges() -> None:
    assert evaluate_cancellation(
        "placed", shipped=False
    ).eligible  # a label may exist, no ship date
    assert not evaluate_cancellation(
        "shipped", shipped=False
    ).eligible  # status wins over a missing date
    assert not evaluate_cancellation("cancelled", shipped=False).eligible
    assert not evaluate_cancellation("delivered", shipped=True).eligible


# ---------------------------------------------------------------- demo numbers are visible
def _app_client(**overrides: object) -> TestClient:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    values: dict[str, object] = {"staff_api_key": SecretStr("k"), **overrides}
    return TestClient(create_app(make_settings(**values), engine))


def test_staff_can_see_which_rules_are_live_and_whether_they_are_confirmed() -> None:
    client = _app_client(refund_return_window_days=45)
    rules = client.get("/staff/model-info", headers={"X-API-Key": "k"}).json()["business_rules"]
    assert rules == {
        "version": "demo-1",
        "refund_return_window_days": 45,
        "refund_not_received_after_days": 14,
        "confirmed_as_real_policy": False,
    }


def test_production_warns_while_the_rules_are_not_confirmed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _app_client(app_env="production")
    assert any("NOT been confirmed" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _app_client(app_env="production", business_rules_confirmed=True)
        _app_client(app_env="development")  # no warning outside production
    assert not any("NOT been confirmed" in r.getMessage() for r in caplog.records)
