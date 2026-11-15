"""Template reply composer (stand-in for the LLM Response Agent).

GROUNDING RULE: a reply may only state what is in the ticket state. It never invents order
details, amounts, policy text or decisions. A future LLM version must keep this contract.
"""

from supportx.agents.state import TicketState
from supportx.agents.supervisor import ROUTES
from supportx.services.refund import format_money

HANDOFF = (
    "Thank you for contacting us. I'm passing your request to a member of our support team, "
    "who will follow up with you personally."
)
APPROVAL_LABELS = {"refund": "refund", "cancellation": "cancellation"}


def compose_reply(state: TicketState) -> str:
    if state.get("escalated"):
        return HANDOFF

    if state.get("missing_info"):
        needed = " and ".join(state["missing_info"])
        return f"To look into this I need your {needed}. Could you please send it?"

    if state.get("order_not_found"):
        return (
            "I couldn't find an order matching those details. "
            "Please check the order number and the email address used for the order."
        )

    refund = state.get("refund")
    approval = state.get("approval")
    if refund is not None and not refund.get("eligible"):
        reasons = " ".join(refund.get("explanations", []))
        return f"I'm sorry, this order isn't eligible for a refund right now. {reasons}".strip()

    if approval is not None:
        kind = _approval_kind(state)
        label = APPROVAL_LABELS.get(kind, kind)
        if approval.get("approved"):
            amount = (
                f" of {format_money(refund['proposed_cents'])}"
                if kind == "refund" and refund
                else ""
            )
            return f"Good news: our team approved your {label} request{amount}."
        return (
            f"Our team reviewed your {label} request and could not approve it. "
            f"Reason: {approval.get('note', 'not provided')}"
        )

    order = state.get("order")
    if order is not None:
        shipment = order.get("shipment")
        tracking = f" Tracking number: {shipment['tracking_number']}." if shipment else ""
        return f"Order {order['order_number']} is currently {order['status']}.{tracking}"

    passages = state.get("passages")
    if passages:
        top = passages[0]
        return f"According to our {top['source']} ({top['heading']}): {top['text']}"

    return HANDOFF  # nothing we can say with confidence


def _approval_kind(state: TicketState) -> str:
    route = ROUTES.get(state.get("intent", ""))
    return (route.approval_kind if route else None) or "refund"
