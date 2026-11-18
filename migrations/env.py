"""Alembic environment. The URL comes from DATABASE_URL (via supportx settings), or from
`alembic -x url=...` / `config.set_main_option("sqlalchemy.url", ...)` (used by tests)."""

from alembic import context
from sqlalchemy import create_engine

from supportx.core import get_settings
from supportx.db.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return (
        config.get_main_option("sqlalchemy.url")
        or context.get_x_argument(as_dictionary=True).get("url")
        or get_settings().require_database_url()
    )


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
