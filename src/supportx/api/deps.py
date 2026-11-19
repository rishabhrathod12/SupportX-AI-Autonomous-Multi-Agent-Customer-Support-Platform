"""FastAPI dependencies: database session, services, staff authentication."""

import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session, sessionmaker

from supportx.agents.tickets import TicketService
from supportx.core import Settings
from supportx.services import RefundPolicy, RefundWorkflow


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_session(request: Request) -> Iterator[Session]:
    """One session per request. Endpoints commit explicitly; any error rolls back."""
    factory: sessionmaker[Session] = request.app.state.session_factory
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_workflow(request: Request, session: SessionDep) -> RefundWorkflow:
    policy: RefundPolicy = request.app.state.policy
    return RefundWorkflow(session, policy)


WorkflowDep = Annotated[RefundWorkflow, Depends(get_workflow)]


def get_ticket_service(request: Request) -> TicketService:
    service: TicketService | None = getattr(request.app.state, "tickets", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The ticket workflow is not available (see the server log for why).",
        )
    return service


TicketsDep = Annotated[TicketService, Depends(get_ticket_service)]


@dataclass(frozen=True)
class StaffIdentity:
    """Who is calling. `name` is None for the shared key (the caller then states the approver)."""

    name: str | None


def require_staff(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> StaffIdentity:
    """Allow the request only if it carries a configured staff key; say whose it is."""
    accepted = settings.staff_identities()
    if not accepted:
        raise HTTPException(
            status_code=503,
            detail="Staff endpoints are disabled: STAFF_API_KEY / STAFF_API_KEYS not configured.",
        )
    provided = (x_api_key or "").encode()
    match: StaffIdentity | None = None
    for key, name in accepted:  # compare against every key: the time taken does not reveal which
        if secrets.compare_digest(provided, key.encode()) and x_api_key is not None:
            match = StaffIdentity(name)
    if match is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return match


StaffDep = Depends(require_staff)
StaffIdentityDep = Annotated[StaffIdentity, Depends(require_staff)]


def resolve_approver(staff: StaffIdentity, claimed: str | None) -> str:
    """The name recorded on a decision. A personal key decides it (a caller cannot claim to be
    someone else); with the shared key the caller must state it, as before."""
    if staff.name is not None:
        return staff.name
    if not claimed:
        raise HTTPException(
            status_code=422, detail="approver is required when using the shared staff key."
        )
    return claimed
