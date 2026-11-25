"""Triage tests. Training data below is SYNTHETIC and only exercises the code; the scores it
gives say nothing about real tickets."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supportx.core import ModelError
from supportx.ml.triage_models import (
    ResolutionTimeModel,
    TriageClassifier,
    split_tickets,
    train_triage,
)
from supportx.services.triage import (
    PRIORITIES,
    TEAMS,
    make_triager,
    priority_rank,
    rule_priority,
    rule_team,
    ticket_facts,
    triage_with_rules,
)


def facts(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"message": "m", "intent": "track_order"}
    base.update(kw)
    return base


def test_priority_rules() -> None:
    assert rule_priority(facts(sentiment="negative", urgency="high")) == "urgent"
    assert rule_priority(facts(escalation_reason="urgent_negative")) == "urgent"
    assert rule_priority(facts(sentiment="negative", urgency="low")) == "high"
    assert rule_priority(facts(intent="complaint")) == "high"
    assert rule_priority(facts(urgency="high")) == "high"
    assert rule_priority(facts(intent="get_refund", approval_kind="refund")) == "normal"
    assert rule_priority(facts()) == "low"
    assert rule_priority({"message": "?"}) == "normal"  # unknown intent is not "low"


def test_team_rules() -> None:
    assert rule_team(facts(intent="get_refund", approval_kind="refund")) == "refunds"
    assert rule_team(facts(intent="cancel_order", approval_kind="cancellation")) == "orders"
    assert rule_team(facts(intent="complaint")) == "customer_care"
    assert rule_team(facts(intent="something_else")) == "general"


def test_rules_only_triage_has_no_resolution_time_and_no_review_flag() -> None:
    result = triage_with_rules(facts(intent="track_order"))
    assert (result.priority, result.team, result.source) == ("low", "orders", "rules")
    assert result.expected_resolution_hours is None and not result.needs_review


def test_ticket_facts_use_only_arrival_information() -> None:
    state = {"message": "hi", "intent": "x", "response": "LATER", "approval": {"approved": True}}
    assert set(ticket_facts(state)) == {
        "message", "intent", "sentiment", "urgency", "escalation_reason", "approval_kind",
    }


def confident(label: str, conf: float = 0.95) -> Any:
    return lambda _f: (label, conf)


def test_confident_model_overrides_rules() -> None:
    triage = make_triager(confident("urgent"), confident("refunds"))
    result = triage(facts())
    assert (result.priority, result.team, result.source) == ("urgent", "refunds", "model")
    assert not result.needs_review


def test_uncertain_model_falls_back_to_rules_and_flags_review() -> None:
    result = make_triager(confident("urgent", 0.4))(facts())
    assert result.priority == "low" and result.needs_review and result.source == "rules"


def test_invalid_label_and_failing_model_fall_back() -> None:
    result = make_triager(confident("catastrophic"))(facts())
    assert result.priority == "low" and result.needs_review

    def broken(_f: Mapping[str, Any]) -> tuple[str, float]:
        raise RuntimeError("boom")

    result = make_triager(broken, broken, lambda _f: 1 / 0)(facts())
    assert result.priority == "low" and result.needs_review
    assert result.expected_resolution_hours is None


def test_resolution_estimate_is_passed_through_when_positive() -> None:
    assert make_triager(resolution_model=lambda _f: 5.04)(facts()).expected_resolution_hours == 5.0
    assert make_triager(resolution_model=lambda _f: -1.0)(facts()).expected_resolution_hours is None


def test_priority_rank_order() -> None:
    assert priority_rank("urgent") > priority_rank("high") > priority_rank("low")
    assert priority_rank(None) == priority_rank("normal")


# ---------------------------------------------------------------- trainable models
def toy_tickets(n: int = 240, with_time: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(n):
        p = PRIORITIES[i % 4]
        t = TEAMS[i % 4]
        rows.append(
            {
                "message": f"{p} {t} word{i % 7} ticket {i}",
                "intent": "x" if i % 2 else "y",
                "sentiment": "negative" if p in ("high", "urgent") else "neutral",
                "urgency": "high" if p == "urgent" else "low",
                "approval_kind": "",
                "priority": p,
                "team": t,
                "resolution_hours": float(rng.integers(1, 48)),
                **({"created_at": start + pd.Timedelta(hours=i)} if with_time else {}),
            }
        )
    return pd.DataFrame(rows)


def test_split_is_chronological_when_timestamps_exist() -> None:
    train, val, test = split_tickets(toy_tickets(with_time=True))
    assert train["created_at"].max() <= val["created_at"].min()
    assert val["created_at"].max() <= test["created_at"].min()


def test_split_checks_input() -> None:
    with pytest.raises(ModelError):
        split_tickets(pd.DataFrame({"message": ["a"]}))
    with pytest.raises(ModelError):
        split_tickets(toy_tickets(20))


def test_train_reports_validation_only_unless_asked_and_has_a_baseline() -> None:
    priority, team, time_model, report = train_triage(toy_tickets())
    assert "test" not in report and "resolution_mae_median_baseline" in report["validation"]
    label, conf = priority({"message": "urgent x", "intent": "x"})
    assert label in PRIORITIES and 0.0 < conf <= 1.0
    assert team({"message": "refunds"})[0] in TEAMS
    assert time_model({"message": "x"}) > 0
    assert "test" in train_triage(toy_tickets(), score_test=True)[3]


def test_models_reject_bad_labels_and_untrained_use() -> None:
    with pytest.raises(ModelError):
        TriageClassifier(PRIORITIES).fit([{"message": "a"}], ["catastrophic"])
    with pytest.raises(ModelError):
        TriageClassifier(PRIORITIES)({"message": "a"})
    with pytest.raises(ModelError):
        ResolutionTimeModel().fit([{"message": "a"}], [0.0])
    with pytest.raises(ModelError):
        ResolutionTimeModel()({"message": "a"})
