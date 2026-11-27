"""Session harness for the infra-backed suites (REAL Postgres + MinIO).

These suites exercise behaviors that live in real infrastructure (REQ-009 RLS
table, REQ-010 cascade, REQ-024 migration over MinIO objects + pgvector
chunks): the REAL application is served over HTTP against throwaway Docker
containers (the repo's own docker-compose service shapes: PostgreSQL with the
pgvector extension, MinIO object storage, ``raganything`` bucket).

Docker is a hard requirement here (no fallback); when the Docker engine is not
reachable the harness skips with an explicit reason instead of faking a pass.
Container orchestration intentionally reuses the ``docker`` CLI with EPHEMERAL
published ports (``--publish 127.0.0.1::<port>``) — no new test dependency.

Environment contract for the migration script harness (see
``test_migration_scopes.py``): ``DATABASE_URL``, ``MINIO_HOST``,
``MINIO_ACCESS``, ``MINIO_SECRET``, plus ``MIGRATION_BASE_URL`` /
``MIGRATION_BEARER_TOKEN`` for the served application when the script drives
the re-indexing through the HTTP surfaces.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from unittest.mock import patch

import asyncpg
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
MINIO_IMAGE = "minio/minio:latest"
PG_USER = "raganything"
PG_PASSWORD = "raganything"
PG_DB = "raganything"
MINIO_ROOT_USER = "minioadmin"
MINIO_ROOT_SECRET = "minioadmin"
MINIO_BUCKET = "raganything"
EMBEDDING_DIM = 8

SKIP_REASON = (
    "Docker engine unavailable in this environment: the infra-backed suites "
    "(real Postgres + MinIO testcontainers via docker) cannot run. Re-run on a "
    "host with Docker (docker info must succeed)."
)


class Cluster:
    """Handles for one throwaway Postgres(pgvector) + MinIO pair."""

    def __init__(self, pg_port: int, minio_port: int) -> None:
        self.pg_port = pg_port
        self.minio_port = minio_port
        self.pg_host = "127.0.0.1"
        self.pg_dsn = (
            f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{pg_port}/{PG_DB}"
        )
        self.pg_dsn_asyncpg = (
            f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{pg_port}/{PG_DB}"
        )
        self.minio_host = f"127.0.0.1:{minio_port}"
        self.minio_endpoint = f"http://127.0.0.1:{minio_port}"

    async def pg(self):
        return await asyncpg.connect(dsn=self.pg_dsn)

    async def pg_exec(self, sql: str, *args) -> None:
        conn = await self.pg()
        try:
            await conn.execute(sql, *args)
        finally:
            await conn.close()

    async def pg_fetch(self, sql: str, *args):
        conn = await self.pg()
        try:
            rows = await conn.fetch(sql, *args)
            return rows
        finally:
            await conn.close()

    def classical_table(self, working_dir: str) -> str:
        digest = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"classical_rag_{digest}"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return probe.returncode == 0


def _run(cmd: list[str], timeout: float = 300) -> str:
    completed = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=True
    )
    return completed.stdout.strip()


def _published_host_port(container: str, container_port: str) -> int:
    mapping = _run(["docker", "port", container, container_port])
    host_port = mapping.splitlines()[0].rsplit(":", 1)[1]
    return int(host_port)


def _await_postgres(container: str, timeout_seconds: float = 90) -> None:
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["docker", "exec", container, "pg_isready", "-U", PG_USER],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"Postgres container {container} never became ready")


def _await_minio(host: str, port: int, timeout_seconds: float = 90) -> None:
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"MinIO endpoint {host}:{port} never became reachable")


def _minio_client(cluster: Cluster):
    from minio import Minio

    return Minio(
        f"127.0.0.1:{cluster.minio_port}",
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_SECRET,
        secure=False,
    )


def minio_keys(cluster: Cluster, prefix: str = "") -> list[str]:
    """List object keys in the test bucket (observation utility)."""
    client = _minio_client(cluster)
    return sorted(
        str(obj.object_name)
        for obj in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
    )


def minio_put(
    cluster: Cluster, key: str, data: bytes, content_type: str = "text/plain"
):
    """Seed one object (arrangement utility)."""
    import io

    client = _minio_client(cluster)
    client.put_object(
        MINIO_BUCKET, key, io.BytesIO(data), length=len(data), content_type=content_type
    )


def minio_remove(cluster: Cluster, key: str) -> None:
    client = _minio_client(cluster)
    client.remove_object(MINIO_BUCKET, key)


async def seed_classical_table(
    cluster: Cluster, working_dir: str, chunk_metadata: list[dict]
) -> str:
    """Create a pgvector table for ``working_dir`` and insert chunk rows.

    Every seeded chunk carries a REAL (deterministic) embedding vector: the
    migration's embedding-preservation contract (code-review C3) must be
    observable — NULL embeddings would hide the silent data loss the C3 fix
    guards against.
    """
    table = cluster.classical_table(working_dir)
    conn = await cluster.pg()
    try:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                content text,
                metadata jsonb,
                embedding vector({EMBEDDING_DIM})
            )
            """
        )
        for metadata in chunk_metadata:
            await conn.execute(
                f"INSERT INTO {table} (content, metadata, embedding) "
                f"VALUES ($1, $2::jsonb, $3::text::vector)",
                "chunk text for " + str(metadata.get("file_path")),
                json.dumps(metadata),
                seed_embedding_text(0.125),
            )
    finally:
        await conn.close()
    return table


def seed_embedding_text(value: float = 0.125) -> str:
    """Build a deterministic pgvector literal of EMBEDDING_DIM dimensions."""
    return "[" + ",".join(str(value) for _ in range(EMBEDDING_DIM)) + "]"


async def table_exists(cluster: Cluster, table: str) -> bool:
    rows = await cluster.pg_fetch(
        "SELECT to_regclass($1) AS reg", public_table_name(table)
    )
    return rows[0]["reg"] is not None


def public_table_name(table: str) -> str:
    return f"public.{table}"


@pytest.fixture(scope="session")
def cluster():
    """One throwaway Postgres(pgvector) + MinIO pair (or an explicit skip)."""
    if not _docker_available():
        pytest.skip(SKIP_REASON, allow_module_level=False)

    token = uuid.uuid4().hex[:8]
    pg_container = f"raganything-test-pg-{token}"
    minio_container = f"raganything-test-minio-{token}"
    try:
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                pg_container,
                "-e",
                f"POSTGRES_USER={PG_USER}",
                "-e",
                f"POSTGRES_PASSWORD={PG_PASSWORD}",
                "-e",
                f"POSTGRES_DB={PG_DB}",
                "--publish",
                "127.0.0.1::5432",
                POSTGRES_IMAGE,
            ]
        )
        _await_postgres(pg_container)
        pg_port = _published_host_port(pg_container, "5432/tcp")

        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                minio_container,
                "-e",
                f"MINIO_ROOT_USER={MINIO_ROOT_USER}",
                "-e",
                f"MINIO_ROOT_PASSWORD={MINIO_ROOT_SECRET}",
                "--publish",
                "127.0.0.1::9000",
                MINIO_IMAGE,
                "server",
                "/data",
            ]
        )
        minio_port = _published_host_port(minio_container, "9000/tcp")
        cluster_obj = Cluster(pg_port=pg_port, minio_port=minio_port)
        _await_minio("127.0.0.1", minio_port)

        client = _minio_client(cluster_obj)
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
        yield cluster_obj
    finally:
        for container in (minio_container, pg_container):
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, timeout=60
            )


class _SessionPatches:
    """Ordered object-level attribute patcher with guaranteed restore."""

    def __init__(self) -> None:
        self._originals: list[tuple[object, str, object]] = []
        self._applied: list[tuple[object, str]] = []

    def apply(self, target: object, name: str, value) -> None:
        self._originals.append((target, name, getattr(target, name)))
        setattr(target, name, value)
        self._applied.append((target, name))

    def restore(self) -> None:
        for target, name, original in reversed(self._originals):
            with contextlib.suppress(Exception):  # teardown must not cascade
                setattr(target, name, original)
        self._originals.clear()
        self._applied.clear()


def _run_real_alembic_upgrade() -> None:
    """Run the REAL Alembic migrations to head against the cluster database.

    A clone of ``main._run_alembic_upgrade`` (which sibling sessions may have
    replaced with a no-op mock): the cluster harness owns its throwaway
    database schema (``rag_spaces`` etc. — REQ-009) and must never inherit
    the root ``running_app``'s migration mock.
    """
    from pathlib import Path as _Path

    from alembic.config import Config

    import main as _main
    from alembic import command

    alembic_dir = _Path(_main.__file__).parent / "alembic"
    cfg = Config(str(_Path(_main.__file__).parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(alembic_dir))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def cluster_app(cluster, auth_directory):
    """The REAL application served over HTTP against the throwaway containers.

    Same composition as production: real Alembic migrations (creating the
    ``rag_spaces`` table etc. on the cluster), real dual-auth wiring (through
    the suite's credential directory doubles), real MinIO + pgvector adapters
    pointed at the containers, and a real uvicorn HTTP listener so external
    tooling (the migration script) can reach the app.

    **Module-scoped on purpose (code-review C1):** as a session fixture its
    finalizer — which RESTORES ``main.rest_security._auth_service`` and
    ``main.mcp_middleware._auth_service`` — ran at the very end of the pytest
    session, AFTER ``tests/unit``, leaving the production security singletons
    wired to the harness ``AuthService`` and turning every unauthenticated
    unit-test request into a 401 (119 failures). At module scope the whole
    mutated module state (security wiring, config patches, the asyncpg pool,
    the lifespan) is handed back pristine at the end of each module of THIS
    suite, before any sibling suite runs. The docker ``cluster`` fixture
    stays session-scoped (the containers are shared); its Alembic upgrade is
    idempotent, so the second module's upgrade is a no-op. The fixture's
    async setup/teardown runs on the session event loop (``loop_scope=
    "session"``) because the asyncpg pool it builds must belong to the SAME
    loop the tests serve the app from.
    """
    import dependencies
    import main
    from domain.services.auth.auth_service import AuthService
    from infrastructure.database.user_llm_reader import AsyncpgUserLlmReader
    from infrastructure.security.crypto import FernetSecretCipher
    from infrastructure.storage.minio_adapter import MinioAdapter

    fernet_key = Fernet.generate_key().decode()

    await cluster.pg_exec("CREATE EXTENSION IF NOT EXISTS vector")

    env_overrides = {
        "SECRET_ENCRYPTION_KEY": fernet_key,
        "LOGTO_URL": "https://logto.soludev.tech",
        "MCP_AUTHORIZATION_SERVER": "https://logto.soludev.tech/oidc",
        "DATABASE_URL": cluster.pg_dsn,
        "MINIO_HOST": cluster.minio_host,
        "MINIO_ACCESS": MINIO_ROOT_USER,
        "MINIO_SECRET": MINIO_ROOT_SECRET,
    }

    patches = _SessionPatches()
    with patch.dict(os.environ, env_overrides):
        patches.apply(main.app_config, "SECRET_ENCRYPTION_KEY", fernet_key)
        # NOTE: MCP_ALLOWED_HOSTS is deliberately NOT patched here. The
        # cluster exercises REST surfaces only (no MCP mount request), the
        # AppConfig default allow-list covers every loopback authority used,
        # and patching it would leak the cluster's narrow list into any
        # sibling session suite (the session fixture outlives them) that
        # expects the production proxy host to be allow-listed.
        patches.apply(main.app_config, "LOGTO_URL", "https://logto.soludev.tech")
        patches.apply(
            main.app_config,
            "MCP_AUTHORIZATION_SERVER",
            "https://logto.soludev.tech/oidc",
        )
        patches.apply(dependencies.db_config, "DATABASE_URL", cluster.pg_dsn_asyncpg)
        patches.apply(dependencies.minio_config, "MINIO_HOST", cluster.minio_host)
        patches.apply(dependencies.minio_config, "MINIO_SECURE", False)
        real_minio = MinioAdapter(
            host=cluster.minio_host,
            access=MINIO_ROOT_USER,
            secret=MINIO_ROOT_SECRET,
            secure=False,
        )
        patches.apply(dependencies, "minio_adapter", real_minio)

        auth_service = AuthService(
            jwt_port=auth_directory.jwt_port,
            api_key_repo=auth_directory.api_key_repository,
        )
        previous_middleware_auth = main.mcp_middleware._auth_service
        previous_rest_auth = main.rest_security._auth_service
        main.mcp_middleware.set_auth_service(auth_service)
        main.rest_security.set_auth_service(auth_service)

        # The credential-directory AuthService wired ABOVE is the harness's
        # dual-auth for the whole session (the shared asyncpg pool has no
        # composable-agents ``api_keys``/JWKS backend, so the production
        # rewiring inside the lifespan could never authenticate these
        # requests); patch the lifespan step to a no-op so the fixture wiring
        # survives the real ``combined_lifespan`` run below.
        patches.apply(main, "_initialize_dual_auth", lambda: None)
        # The OTHER sessions sharing this process (e.g. the root
        # ``running_app``) may mock the lifespan's migration step away; the
        # cluster owns its database schema lifecycle and must run the REAL
        # Alembic upgrade (creating ``rag_spaces`` etc. on the containers).
        patches.apply(main, "_run_alembic_upgrade", _run_real_alembic_upgrade)
        # Process globals the lifespan re-assigns downstream: recorded so the
        # teardown can hand the next suite a clean module state (no closed
        # pool / stale registry store leaking into a sibling session).
        patches.apply(main, "_mcp_pool", None)
        patches.apply(dependencies, "mcp_registry_store", None)
        patches.apply(dependencies, "rag_space_repository", None)

        async def _own_lifespan(app):
            async with main.combined_lifespan(app):
                yield

        # Drive the real lifespan explicitly (same pattern as the global
        # ``running_app`` fixture).
        stop = asyncio.Event()
        started = asyncio.Event()

        async def _lifespan_owner():
            try:
                async with main.combined_lifespan(main.app):
                    started.set()
                    await stop.wait()
            except Exception:
                started.set()
                raise

        owner_task = asyncio.create_task(_lifespan_owner())
        await started.wait()
        if owner_task.done():
            await owner_task  # surface the startup failure

        patches.apply(
            dependencies,
            "user_llm_reader",
            AsyncpgUserLlmReader(
                pool=main._mcp_pool, cipher=FernetSecretCipher(fernet_key)
            ),
        )

        # Real HTTP listener so external tooling (migration script) reaches the app.
        import uvicorn

        config = uvicorn.Config(
            main.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"
        )
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        for _ in range(200):
            if server.started:
                break
            if serve_task.done():
                await serve_task
            await asyncio.sleep(0.05)
        assert server.started, "uvicorn did not start for the cluster app"
        http_port = server.servers[0].sockets[0].getsockname()[1]
        cluster.http_base_url = f"http://127.0.0.1:{http_port}"
        cluster.http_bearer = "jwt-alice"

        try:
            yield main.app
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(serve_task, timeout=30)
            except Exception:  # noqa: BLE001 — teardown must not mask failures
                serve_task.cancel()
            stop.set()
            # The lifespan owner's shutdown may raise on the NESTED session
            # teardown (fastmcp session managers were also entered by the
            # sibling session fixture) — the harness restoration below MUST
            # still run so the next suite inherits a clean module state.
            with contextlib.suppress(Exception):
                await owner_task
            main.mcp_middleware.set_auth_service(previous_middleware_auth)
            main.rest_security.set_auth_service(previous_rest_auth)
            patches.restore()


@pytest_asyncio.fixture(loop_scope="session")
async def infra_api(cluster_app):
    """httpx AsyncClient over the REAL cluster-served application."""
    import httpx
    from httpx import ASGITransport

    async with httpx.AsyncClient(
        transport=ASGITransport(app=cluster_app), base_url="http://test"
    ) as client:
        yield client


def script_env(cluster: Cluster) -> dict[str, str]:
    """Environment handed to the migration-script subprocess."""
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": cluster.pg_dsn,
            "MINIO_HOST": cluster.minio_host,
            "MINIO_ACCESS": MINIO_ROOT_USER,
            "MINIO_SECRET": MINIO_ROOT_SECRET,
            "MINIO_BUCKET": MINIO_BUCKET,
            "MIGRATION_BASE_URL": getattr(
                cluster, "http_base_url", "http://127.0.0.1:0"
            ),
            "MIGRATION_BEARER_TOKEN": getattr(cluster, "http_bearer", "jwt-alice"),
            # The script writes its REQ-024 (d) triage report FILE; point the
            # harness at a throwaway path so test runs never pollute the repo.
            "MIGRATION_REPORT_PATH": os.path.join(
                tempfile.gettempdir(), "scope-migration-report.txt"
            ),
        }
    )
    return env


def bearer(credential: str = "jwt-alice") -> dict[str, str]:
    """Credential headers as the real clients present them."""
    if credential.startswith("jwt-"):
        return {"Authorization": f"Bearer {credential}"}
    return {"X-API-Key": credential}


def space_entries(response) -> list[dict]:
    """Normalise a GET /api/v1/spaces response into record dicts."""
    assert response.status_code == 200, response.text
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("spaces") or payload.get("items") or []
    return [entry for entry in payload if isinstance(entry, dict)]


class ClusterMinio:
    """Arrangement/observation utility over the REAL test-bucket objects."""

    def __init__(self, cluster: Cluster) -> None:
        self._cluster = cluster

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> None:
        minio_put(self._cluster, key, data, content_type)

    def object_keys(self, prefix: str = "") -> list[str]:
        return minio_keys(self._cluster, prefix=prefix)

    def remove(self, key: str) -> None:
        minio_remove(self._cluster, key)


class LegacyVectorTables:
    """Arrangement/observation utility for ``classical_rag_*`` tables."""

    def __init__(self, cluster: Cluster) -> None:
        self._cluster = cluster

    def table_for(self, working_dir: str) -> str:
        return self._cluster.classical_table(working_dir)

    async def create(self, working_dir: str, chunk_metadata: list[dict]) -> str:
        return await seed_classical_table(self._cluster, working_dir, chunk_metadata)

    async def exists(self, table: str) -> bool:
        return await table_exists(self._cluster, table)

    async def drop(self, table: str) -> None:
        await self._cluster.pg_exec(f"DROP TABLE IF EXISTS {table}")


class SpaceRows:
    """Arrangement/observation utility for ``rag_spaces`` rows."""

    def __init__(self, cluster: Cluster) -> None:
        self._cluster = cluster

    async def set_working_dirs(
        self, user_id: str, agent_slug: str, working_dirs: list[str]
    ) -> None:
        conn = await self._cluster.pg()
        try:
            await conn.execute("SET row_security = off")
            await conn.execute(
                "UPDATE rag_spaces SET working_dirs = $1::jsonb "
                "WHERE user_id = $2 AND agent_slug = $3",
                json.dumps(working_dirs),
                user_id,
                agent_slug,
            )
        finally:
            await conn.close()

    async def fetch(self, user_id: str, agent_slug: str):
        conn = await self._cluster.pg()
        try:
            await conn.execute("SET row_security = off")
            return await conn.fetchrow(
                "SELECT * FROM rag_spaces WHERE user_id = $1 AND agent_slug = $2",
                user_id,
                agent_slug,
            )
        finally:
            await conn.close()


@pytest.fixture
def cluster_minio(cluster) -> ClusterMinio:
    return ClusterMinio(cluster)


@pytest.fixture
def legacy_tables(cluster) -> LegacyVectorTables:
    return LegacyVectorTables(cluster)


@pytest.fixture
def space_rows(cluster) -> SpaceRows:
    return SpaceRows(cluster)
