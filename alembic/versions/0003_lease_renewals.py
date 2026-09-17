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
    # One row per successful renewal. The row doubles as the idempotency
    # record for POST /leases/{token}/renew: the key is unique, and the
    # stored expires_before/expires_after are exactly what a same-key
    # same-params retry replays, so a lost response can never extend twice.
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
            sa.BigInteger(),
            sa.ForeignKey("leases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        # Stable textual fingerprint of the request (token + seconds), same
        # idea as idempotency_keys.request_params: a reused key with
        # different parameters is a stable conflict.
        sa.Column("request_params", sa.Text(), nullable=False),
        sa.Column("additional_seconds", sa.Integer(), nullable=False),
        # Both snapshots come from the leases row itself inside the renewal
        # transaction (database clock only); expires_after is what the
        # lease's expires_at becomes.
        sa.Column("expires_before", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_after", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.CheckConstraint(
            "additional_seconds > 0",
            name="lease_renewals_seconds_positive",
        ),
        sa.CheckConstraint(
            "expires_after > expires_before",
            name="lease_renewals_extends_expiry",
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
