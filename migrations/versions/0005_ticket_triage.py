"""ticket triage columns

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tickets") as batch:
        batch.add_column(sa.Column("priority", sa.String(length=10), nullable=True))
        batch.add_column(sa.Column("team", sa.String(length=30), nullable=True))
        batch.add_column(sa.Column("triage_source", sa.String(length=20), nullable=True))
        batch.add_column(
            sa.Column("triage_needs_review", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.add_column(sa.Column("expected_resolution_hours", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tickets") as batch:
        batch.drop_column("expected_resolution_hours")
        batch.drop_column("triage_needs_review")
        batch.drop_column("triage_source")
        batch.drop_column("team")
        batch.drop_column("priority")
