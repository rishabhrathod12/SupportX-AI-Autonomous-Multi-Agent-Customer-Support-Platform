"""FastAPI dependencies: database session, services, staff authentication."""

import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session, sessionmaker

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


def require_staff(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Allow the request only if it carries the configured staff API key."""
    if settings.staff_api_key is None:
        raise HTTPException(
            status_code=503,
            detail="Staff endpoints are disabled: STAFF_API_KEY is not configured.",
        )
    expected = settings.staff_api_key.get_secret_value().encode()
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


StaffDep = Depends(require_staff)
