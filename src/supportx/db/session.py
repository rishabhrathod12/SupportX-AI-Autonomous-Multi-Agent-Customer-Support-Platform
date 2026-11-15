"""Engine and session creation. The database URL comes from settings (DATABASE_URL)."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from supportx.core import get_logger, get_settings
from supportx.db.models import Base

logger = get_logger(__name__)


def make_engine(url: str | None = None) -> Engine:
    """Create an engine. Without `url`, use DATABASE_URL (raises ConfigurationError if unset)."""
    return create_engine(url or get_settings().require_database_url(), pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create all tables that do not exist yet. Real deployments use Alembic migrations (later)."""
    Base.metadata.create_all(engine)
    logger.info("Database tables ready")


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Give a session; commit on success, roll back on any error, always close."""
    session = sessionmaker(engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
