"""lease renewals: lease_renewals table

Revision ID: 0003_lease_renewals
Revises: 0002_lease_release
Create Date: 2026-09-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_lease_renewals"
down_revision: str | None = "0002_lease_release"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One row per successful renewal. The row is the idempotency record for
    # the renew call: idempotency_key is unique, so a lost response followed
    # by a same-key retry replays the recorded before/after expiry instead of
    # extending the lease a second time. No columns are added to `leases`
    # itself — expires_at keeps its single-writer meaning and historical
    # rows need no backfill.
    op.create_table(
        "lease_renewals",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "lease_id",
            sa.BigInteger,
            sa.ForeignKey("leases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_params", sa.Text, nullable=False),
        sa.Column("additional_seconds", sa.Integer(), nullable=False),
        # Expiry before/after this renewal, both copied from the leases row
        # inside the same UPDATE statement, so the record always matches the
        # lease's own history exactly.
        sa.Column(
            "previous_expires_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "new_expires_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.CheckConstraint(
            "additional_seconds >= 5 AND additional_seconds <= 120",
            name="lease_renewals_seconds_range",
        ),
        sa.CheckConstraint(
            "new_expires_at > previous_expires_at",
            name="lease_renewals_extends",
        ),
    )
    op.create_index(
        "uq_lease_renewals_key", "lease_renewals", ["idempotency_key"], unique=True
    )
    op.create_index(
        "ix_lease_renewals_lease", "lease_renewals", ["lease_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_lease_renewals_lease", table_name="lease_renewals")
    op.drop_index("uq_lease_renewals_key", table_name="lease_renewals")
    op.drop_table("lease_renewals")
