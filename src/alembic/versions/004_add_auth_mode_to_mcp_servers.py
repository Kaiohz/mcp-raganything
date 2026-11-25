"""Add auth_mode column to mcp_servers.

Revision ID: 004
Revises: 003
Create Date: 2026-11-26

Adds the ``auth_mode`` column (feat-mcp-oauth2 REQ-004) so the credential-
transmission mode of a registered MCP server (``"headers"`` — the stored
headers are sent as-is, or ``"user_token"`` — the connected user's credential
is injected by the consuming service) survives restarts. Non-secret metadata:
existing rows are backfilled with ``'headers'`` (the pre-feature behavior),
and the mode is echoed unmasked in the API's masked projections, unlike the
encrypted ``headers``/``env``/``auth_token`` columns.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE mcp_servers
            ADD COLUMN IF NOT EXISTS auth_mode VARCHAR(20) NOT NULL DEFAULT 'headers';
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE mcp_servers DROP COLUMN IF EXISTS auth_mode;")
