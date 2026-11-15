from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from supportx.api.deps import SessionDep
from supportx.api.schemas import HealthOut
from supportx.core import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """Is the process up? Does not touch the database."""
    return HealthOut(status="ok")


@router.get("/health/ready", response_model=HealthOut)
def ready(session: SessionDep) -> HealthOut:
    """Can the service reach its database?"""
    try:
        session.execute(text("select 1"))
    except Exception as exc:  # any database failure means "not ready"
        logger.error("Readiness check failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Database is not reachable.") from exc
    return HealthOut(status="ready")
