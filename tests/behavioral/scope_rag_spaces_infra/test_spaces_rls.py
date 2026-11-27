"""Behavioral proof of the ``rag_spaces`` RLS backstop (REQ-009, review C4).

The production application connects as a superuser role (``POSTGRES_USER`` in
the compose files), which BYPASSES RLS — so the isolation the app relies on
comes from the explicit ``WHERE user_id`` clauses. The RLS POLICY itself is
the second defence layer and is only observable against a NON-superuser role:
this suite creates a throwaway ``NOSUPERUSER`` role, grants it DML on
``rag_spaces`` ONLY, and drives the REAL :class:`AsyncpgRagSpaceRepository`
through it, asserting:

* the repo pattern (``set_config('app.user_id', ..., is_local=true)`` inside
  the SAME explicit transaction as the statement — code-review C4) actually
  engages the FORCE RLS policy: a row of the caller's own id is inserted and
  only the caller's rows are ever visible;
* a write naming ANOTHER user is rejected by the ``WITH CHECK`` clause
  (``InsufficientPrivilegeError``) — the DB-level backstop does not trust the
  application;
* the pre-fix pattern (``set_config`` OUTSIDE an explicit transaction, the
  asyncpg implicit-transaction trap) is rejected for a NON-superuser role —
  pinning exactly the regression the C4 fix removed.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

_RLS_ROLE = "raganything_rls_probe"
_RLS_PASSWORD = "rls-probe-password"


def _app_role_dsn(cluster) -> str:
    """Build the non-superuser role's DSN against the throwaway cluster."""
    return f"postgresql://{_RLS_ROLE}:{_RLS_PASSWORD}@{cluster.pg_dsn.split('@', 1)[1]}"


@pytest.fixture
async def rls_role(cluster, cluster_app):
    """Create the NOSUPERUSER probe role with DML grants on ``rag_spaces``.

    Setup/teardown both run on the privileged harness connection (the role
    cannot drop itself); ``cluster_app`` is only a dependency to guarantee
    the migrations (and therefore ``rag_spaces``) exist before the grants.
    """
    creator = await cluster.pg()
    try:
        await creator.execute(f"DROP ROLE IF EXISTS {_RLS_ROLE}")  # noqa: S608
        await creator.execute(
            f"CREATE ROLE {_RLS_ROLE} NOSUPERUSER LOGIN "  # noqa: S608
            f"PASSWORD '{_RLS_PASSWORD}'"
        )
        await creator.execute(
            f"GRANT USAGE ON SCHEMA public TO {_RLS_ROLE}"  # noqa: S608
        )
        await creator.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON rag_spaces TO "  # noqa: S608
            f"{_RLS_ROLE}"
        )
    finally:
        await creator.close()

    yield _RLS_ROLE

    # Postgres 16: a role with granted privileges cannot be dropped until the
    # grants are revoked; revoke everything then drop the throwaway probe
    # role (teardown failures must never mask the test run).
    cleanup = await cluster.pg()
    try:
        await cleanup.execute(
            f"REVOKE ALL ON ALL TABLES IN SCHEMA public "  # noqa: S608
            f"FROM {_RLS_ROLE}"
        )
        await cleanup.execute(
            f"REVOKE ALL ON SCHEMA public FROM {_RLS_ROLE}"  # noqa: S608
        )
        await cleanup.execute(f"DROP ROLE IF EXISTS {_RLS_ROLE}")  # noqa: S608
    finally:
        await cleanup.close()


async def test_repository_pattern_engages_rls_with_non_superuser_role(
    cluster, rls_role, space_rows
):
    """The repo's GUC-in-transaction pattern makes the FORCE RLS policy live."""
    import asyncpg

    from infrastructure.database.rag_space_repository import (
        AsyncpgRagSpaceRepository,
    )

    role_pool = await asyncpg.create_pool(dsn=_app_role_dsn(cluster), min_size=1)
    repository = AsyncpgRagSpaceRepository(pool=role_pool)
    try:
        # (a) The caller's own record writes INSIDE the FORCE RLS policy —
        # proof that the transaction-local GUC survived to the statement.
        space = await repository.ensure_user_space(
            "user-alice", "agent-rls", "user-alice/agent-rls/p1/"
        )
        assert space.user_id == "user-alice"
        assert space.agent_slug == "agent-rls"

        # (b) A second user's first-use writes its OWN row (never collides).
        bob_space = await repository.ensure_user_space("user-bob", "agent-rls")
        assert bob_space.user_id == "user-bob"

        # (c) RLS isolates the reads even BESIDE the explicit WHERE clause:
        # alice sees her own rows (this shared suite table may carry other
        # user-alice spaces) but NEVER a user-bob row; and her own
        # agent-rls record is retrievable, while a foreign slug is not.
        alice_rows = await repository.list_for_user("user-alice")
        alice_pairs = {(row.user_id, row.agent_slug) for row in alice_rows}
        assert ("user-alice", "agent-rls") in alice_pairs
        assert all(user_id == "user-alice" for user_id, _slug in alice_pairs)
        assert await repository.get("user-alice", "agent-rls") is not None
        assert await repository.get("user-bob", "agent-rls-foreign") is None

        bob_rows = await repository.list_for_user("user-bob")
        bob_pairs = {(row.user_id, row.agent_slug) for row in bob_rows}
        assert ("user-bob", "agent-rls") in bob_pairs
        assert all(user_id == "user-bob" for user_id, _slug in bob_pairs)

        row = await space_rows.fetch("user-alice", "agent-rls")
        assert row is not None
    finally:
        await role_pool.close()


async def test_cross_user_write_is_rejected_by_the_rls_with_check_clause(
    cluster, rls_role
):
    """A non-superuser write naming ANOTHER user's id is rejected by RLS."""
    import asyncpg

    conn = await asyncpg.connect(dsn=_app_role_dsn(cluster))
    try:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "user-bob")
            # The caller "user-bob" attempts to write user-alice's row: the
            # WITH CHECK clause (user_id = current_setting('app.user_id'))
            # must REJECT it — the database does not trust the application.
            with pytest.raises(asyncpg.InsufficientPrivilegeError) as excinfo:
                await conn.execute(
                    "INSERT INTO rag_spaces (user_id, agent_slug) "
                    "VALUES ('user-alice', 'agent-foreign')"
                )
        assert "row-level security policy" in str(excinfo.value)
    finally:
        await conn.close()


async def test_unscoped_guc_is_rejected_for_non_superuser_roles(cluster, rls_role):
    """The pre-C4 pattern (set_config OUTSIDE a transaction) must be rejected.

    This pins the regression class the C4 fix removed: asyncpg runs a bare
    statement in an implicit single-statement transaction, so the
    transaction-LOCAL ``app.user_id`` evaporates before the next statement —
    the RLS policy then evaluates against an empty GUC and rejects the write
    for non-superuser roles (a superuser silently bypasses, which is why the
    harness needs this dedicated role to observe the backstop at all).
    """
    import asyncpg

    conn = await asyncpg.connect(dsn=_app_role_dsn(cluster))
    try:
        # The BUGGY pattern: autocommit statements, GUC set OUTSIDE the
        # transaction that performs the write.
        await conn.execute("SELECT set_config('app.user_id', $1, true)", "user-alice")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(
                "INSERT INTO rag_spaces (user_id, agent_slug) "
                "VALUES ('user-alice', 'agent-should-fail')"
            )
    finally:
        await conn.close()


async def test_cross_user_rows_are_invisible_to_a_non_superuser_role(
    cluster, rls_role, space_rows
):
    """RLS hides OTHER users' rows even from a direct SELECT without a WHERE."""
    import asyncpg

    # Arrange — a row owned by user-alice (via the privileged harness role).
    conn_owner = await cluster.pg()
    try:
        await conn_owner.execute("SET row_security = off")
        await conn_owner.execute(
            "INSERT INTO rag_spaces (user_id, agent_slug) "
            "VALUES ('user-alice', 'agent-hidden')"
        )
    finally:
        await conn_owner.close()

    conn_probe = await asyncpg.connect(dsn=_app_role_dsn(cluster))
    try:
        async with conn_probe.transaction():
            await conn_probe.execute(
                "SELECT set_config('app.user_id', $1, true)", "user-bob"
            )
        # Reconnect: user-bob's context; the user-alice row must NOT appear
        # even in an UNFILTERED SELECT (no WHERE user_id clause).
        rows = await conn_probe.fetch("SELECT user_id, agent_slug FROM rag_spaces")
        assert ("user-alice", "agent-hidden") not in [
            (row["user_id"], row["agent_slug"]) for row in rows
        ]
    finally:
        await conn_probe.close()
