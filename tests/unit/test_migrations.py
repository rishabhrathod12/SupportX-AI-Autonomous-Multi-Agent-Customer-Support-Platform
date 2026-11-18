"""The Alembic migrations must build exactly the schema the models describe."""

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from supportx.db.models import Base

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def migration_setup(tmp_path: Path) -> tuple[Config, str]:
    url = f"sqlite:///{(tmp_path / 'migr.db').as_posix()}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg, url


def test_upgrade_matches_models_and_downgrade_is_clean(migration_setup: tuple[Config, str]) -> None:
    cfg, url = migration_setup
    command.upgrade(cfg, "head")

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            drift = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert drift == [], f"models and migrations differ: {drift}"
        assert {"customers", "orders", "tickets", "refund_requests"} <= set(
            inspect(engine).get_table_names()
        )

        command.downgrade(cfg, "base")
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    finally:
        engine.dispose()
