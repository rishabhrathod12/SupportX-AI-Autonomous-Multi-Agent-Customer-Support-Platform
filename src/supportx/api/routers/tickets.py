"""Customer ticket endpoints: send a message, check on it later.

POST bodies keep emails out of URLs (and therefore out of access logs).
"""

from fastapi import APIRouter, Response

from supportx.api.deps import TicketsDep
from supportx.api.schemas import CustomerTicketOut, TicketCreateIn, TicketStatusIn
from supportx.db import TicketStatus

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=CustomerTicketOut)
def create_ticket(
    body: TicketCreateIn, response: Response, tickets: TicketsDep
) -> CustomerTicketOut:
    """Send a message. The reply comes back in the same call unless a person must decide first."""
    view = tickets.submit(body.message, body.customer_email, body.order_number)
    if view.status is TicketStatus.AWAITING_APPROVAL:
        response.status_code = 202  # accepted, not finished
    return CustomerTicketOut.from_view(view)


@router.post("/status", response_model=CustomerTicketOut)
def ticket_status(body: TicketStatusIn, tickets: TicketsDep) -> CustomerTicketOut:
    """Status or reply of a ticket, for its owner only."""
    return CustomerTicketOut.from_view(
        tickets.get_for_customer(body.ticket_id, body.customer_email)
    )
