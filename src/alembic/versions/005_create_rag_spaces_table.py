"""Create the RLS-protected rag_spaces table (scope-rag-memories REQ-009).

Revision ID: 005
Revises: 004
Create Date: 2026-11-26

One RAG space per ``(user_id, agent_slug)``: the table records each agent
scope a user actually uses (``personal`` is intentionally NOT stored — the
personal tree needs no record), the composed classical-RAG ``working_dirs``
created inside the space (tracked so the cascade delete can drop the matching
``classical_rag_*`` tables, REQ-008/REQ-010) and the lifecycle timestamps
(exclusive ``last_used_at`` refreshed on every scoped use, REQ-011).

Row-Level Security follows the mcp_servers pattern (migrations 002/003):
FORCE so the table owner cannot bypass it, with the same per-user policy
``user_id = current_setting('app.user_id', true)`` — the application sets the
GUC on each acquired asyncpg connection (``set_rls_context`` in
:class:`~infrastructure.database.rag_space_repository.AsyncpgRagSpaceRepository`)
after a successful dual-auth.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_spaces (
            id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id       VARCHAR(255) NOT NULL,
            agent_slug    VARCHAR(100) NOT NULL,
            working_dirs  JSONB        NOT NULL DEFAULT '[]'::jsonb,
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
            last_used_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
            CONSTRAINT uq_rag_spaces_user_agent UNIQUE (user_id, agent_slug)
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_spaces_user_id
            ON rag_spaces (user_id);
        """
    )
    op.execute("ALTER TABLE rag_spaces ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE rag_spaces FORCE ROW LEVEL SECURITY;")
    op.execute(
        """
        CREATE POLICY rag_spaces_user_isolation
            ON rag_spaces
            USING (user_id = current_setting('app.user_id', true))
            WITH CHECK (user_id = current_setting('app.user_id', true));
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS rag_spaces_user_isolation ON rag_spaces;")
    op.execute("ALTER TABLE rag_spaces NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE rag_spaces DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP INDEX IF EXISTS idx_rag_spaces_user_id;")
    op.execute("DROP TABLE IF EXISTS rag_spaces;")
