"""The scenario evaluator itself: it must pass a correct system and catch a broken one.
Uses a deterministic fake intent model; the scenario file is SYNTHETIC (a code-path check)."""

from pathlib import Path

import pandas as pd
import pytest

from supportx.agents import HumanDecision, TicketService
from supportx.agents.tickets import TicketView, _view
from supportx.db import RefundRequest, RefundStatus, Ticket, session_scope
from supportx.ml import agent_eval
from supportx.rag import Retriever
from tests.unit.helpers import FakeIntentModel

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "data" / "sample" / "agent_scenarios.csv"
POLICIES = ROOT / "data" / "sample" / "policies"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


def frame(*ids: str) -> pd.DataFrame:
    rows = pd.read_csv(SCENARIOS, encoding="utf-8")
    return rows[rows["id"].isin(ids)].reset_index(drop=True)


def test_correct_workflow_passes_refund_cancel_and_denial_scenarios(retriever: Retriever) -> None:
    report = agent_eval.evaluate_scenarios(
        FakeIntentModel(), frame("s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"), retriever
    )
    assert report["data_origin"] == "synthetic"
    assert report["safety_violations"] == 0
    assert report["passed"] == report["n"] == 8, report["failures"]


def test_a_wrong_expectation_is_reported_as_a_failure(retriever: Retriever) -> None:
    rows = frame("s03")
    rows.loc[0, "expect_final_refund"] = "rejected"  # the scenario approves, so this must fail
    report = agent_eval.evaluate_scenarios(FakeIntentModel(), rows, retriever)
    assert report["passed"] == 0 and "final refund status" in report["failures"][0]["problems"][0]


def test_an_auto_approving_system_is_caught_as_a_safety_violation(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_decide(self: TicketService, ticket_id: str, decision: HumanDecision) -> TicketView:
        """Simulates a bug: ignores the human and approves the refund anyway."""
        with session_scope(self._engine) as session:
            for request in session.query(RefundRequest):
                request.status = RefundStatus.APPROVED
            ticket = session.get(Ticket, ticket_id)
            assert ticket is not None
            ticket.response = "done"
            return _view(ticket)

    monkeypatch.setattr(TicketService, "decide_ticket", broken_decide)
    report = agent_eval.evaluate_scenarios(FakeIntentModel(), frame("s04"), retriever)  # "reject"
    assert report["safety_violations"] >= 1
