"""Template reply composer (stand-in for the LLM Response Agent).

GROUNDING RULE: a reply may only state what is in the ticket state. It never invents order
details, amounts, policy text or decisions. A future LLM version must keep this contract.
"""

from supportx.agents.state import TicketState
from supportx.agents.supervisor import ROUTES
from supportx.services.cancellation import evaluate_cancellation_for_state
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

    order = state.get("order")
    if approval is None and order is not None and _approval_kind(state) == "cancellation":
        cancellation = evaluate_cancellation_for_state(order)
        if not cancellation.eligible:
            return (
                f"I'm sorry, order {order['order_number']} can't be cancelled. "
                f"{cancellation.explanation}"
            )

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

    if order is not None:
        shipment = order.get("shipment")
        tracking = f" Tracking number: {shipment['tracking_number']}." if shipment else ""
        return f"Order {order['order_number']} is currently {order['status']}.{tracking}"

    passages = state.get("passages")
    if passages:
        first, *others = passages
        reply = f"According to our {first['source']} ({first['heading']}): {first['text']}"
        for extra in others:  # the runner-up, when it scored nearly as well
            reply += f"\n\nAlso relevant ({extra['source']}, {extra['heading']}): {extra['text']}"
        if state.get("knowledge_fallback"):
            reply += (
                "\n\nI wasn't completely sure what you were asking. If this doesn't answer it, "
                "reply with a few more details and a member of our team will help."
            )
        return reply

    return HANDOFF  # nothing we can say with confidence


def _approval_kind(state: TicketState) -> str:
    route = ROUTES.get(state.get("intent", ""))
    return (route.approval_kind if route else None) or "refund"
