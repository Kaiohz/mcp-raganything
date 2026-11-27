"""Add tool_names_json column to mcp_servers.

Revision ID: 006
Revises: 005
Create Date: 2026-11-27

Adds the ``tool_names_json`` column so the registry exposes the NAMES of the
tools discovered during the last validation/create/update probe (the UI card
can show them — until now only the count was stored). Non-secret metadata:
existing rows are backfilled with ``'[]'`` and the names are echoed unmasked,
like ``tool_count``.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE mcp_servers
            ADD COLUMN IF NOT EXISTS tool_names_json TEXT NOT NULL DEFAULT '[]';
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE mcp_servers DROP COLUMN IF EXISTS tool_names_json;")
