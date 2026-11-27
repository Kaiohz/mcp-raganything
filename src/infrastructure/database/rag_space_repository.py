"""Raw asyncpg repository for the RLS-protected ``rag_spaces`` table.

The table is created by the ``005_create_rag_spaces_table`` migration with
Row-Level Security forced and the same per-user policy as ``mcp_servers``
(``user_id = current_setting('app.user_id', true)``).

**RLS GUC lifetime (code-review C4):** asyncpg runs every statement in an
implict single-statement transaction, so a ``set_config(..., is_local=true)``
issued OUTSIDE an explicit transaction evaporates with it — the RLS policy
would silently degenerate to ``user_id = NULL`` for every following
statement. Each repository call therefore opens ONE explicit transaction and
sets the ``app.user_id`` GUC INSIDE it, so the setting survives until exactly
the statement that needs it (and rolls back atomically with it). Every SQL
statement also carries its explicit ``user_id`` WHERE clause (defence in
depth on databases where RLS is not yet engaged or the role bypasses RLS).
The ``user_id`` is provided explicitly by the callers (the use cases) — never
read from ambient state.

Schema ownership: ``rag_spaces`` is created and evolved by THIS service's
Alembic migrations (run at startup via ``_run_alembic_upgrade`` in
``main.py``).
"""

import json
import logging
from typing import Any

from domain.entities.rag_space import RagSpace
from domain.logging.messages import LogMessage
from domain.ports.rag_space_repository import RagSpaceRepository
from infrastructure.database.rls_context import set_rls_context

logger = logging.getLogger(__name__)

_COLUMNS = "id, user_id, agent_slug, working_dirs, created_at, last_used_at"


def _decode_working_dirs(raw: Any) -> list[str]:
    """Decode the ``working_dirs`` jsonb column (asyncpg returns a string)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [str(value) for value in raw] if isinstance(raw, list) else []


def _row_to_space(row: Any) -> RagSpace:
    """Project one asyncpg row into the domain entity (direct transformation)."""
    return RagSpace(
        id=row["id"],
        user_id=row["user_id"],
        agent_slug=row["agent_slug"],
        working_dirs=_decode_working_dirs(row["working_dirs"]),
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
    )


class AsyncpgRagSpaceRepository(RagSpaceRepository):
    """Asyncpg-backed ``rag_spaces`` repository (per-user RLS scoped)."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def ensure_user_space(
        self, user_id: str, agent_slug: str, working_dir: str | None = None
    ) -> RagSpace:
        """Insert-or-touch the record atomically and return the stored row."""
        tracked = json.dumps([working_dir]) if working_dir else None
        sql = f"""
        INSERT INTO rag_spaces (user_id, agent_slug, working_dirs, last_used_at)
        VALUES ($1, $2, COALESCE($3::jsonb, '[]'::jsonb), now())
        ON CONFLICT (user_id, agent_slug) DO UPDATE SET
            last_used_at = now(),
            working_dirs = CASE
                WHEN $3::jsonb IS NULL THEN rag_spaces.working_dirs
                WHEN rag_spaces.working_dirs @> $3::jsonb
                    THEN rag_spaces.working_dirs
                ELSE rag_spaces.working_dirs || $3::jsonb
            END
        RETURNING {_COLUMNS}, (xmax = 0) AS inserted;
        """
        async with self._pool.acquire() as conn, conn.transaction():
            # The GUC is transaction-LOCAL (is_local=true): it survives
            # until exactly this statement because both live in the SAME
            # explicit transaction (code-review C4) — the RLS policy
            # ``user_id = current_setting('app.user_id', true)`` is then
            # actually evaluated against the caller's id.
            await set_rls_context(conn, user_id)
            row = await conn.fetchrow(sql, user_id, agent_slug, tracked)
        if row is None:  # pragma: no cover — RETURNING always yields a row
            raise RuntimeError("rag_spaces upsert returned no row")
        if row["inserted"]:
            # Space CREATED is a provisioning event — logged once per insert,
            # not on every pure refresh (code-review improvement 6).
            logger.info(
                LogMessage.SPACE_CREATED, user_id, agent_slug, working_dir or ""
            )
        return _row_to_space(row)

    async def get(self, user_id: str, agent_slug: str) -> RagSpace | None:
        """Fetch one record scoped to its owner."""
        sql = f"""
        SELECT {_COLUMNS}
        FROM rag_spaces
        WHERE user_id = $1 AND agent_slug = $2;
        """
        async with self._pool.acquire() as conn, conn.transaction():
            await set_rls_context(conn, user_id)
            row = await conn.fetchrow(sql, user_id, agent_slug)
        return _row_to_space(row) if row is not None else None

    async def list_for_user(self, user_id: str) -> list[RagSpace]:
        """List all the caller's records (RLS + explicit WHERE)."""
        sql = f"""
        SELECT {_COLUMNS}
        FROM rag_spaces
        WHERE user_id = $1
        ORDER BY created_at;
        """
        async with self._pool.acquire() as conn, conn.transaction():
            await set_rls_context(conn, user_id)
            rows = await conn.fetch(sql, user_id)
        return [_row_to_space(row) for row in rows]

    async def delete(self, user_id: str, agent_slug: str) -> bool:
        """Delete one record; ``False`` when it did not exist."""
        sql = """
        DELETE FROM rag_spaces
        WHERE user_id = $1 AND agent_slug = $2;
        """
        async with self._pool.acquire() as conn, conn.transaction():
            await set_rls_context(conn, user_id)
            status = await conn.execute(sql, user_id, agent_slug)
        return status.split()[-1] != "0"
