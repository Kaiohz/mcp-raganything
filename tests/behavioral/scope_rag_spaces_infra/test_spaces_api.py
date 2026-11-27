"""Behavioral contract for the RAG spaces API (REQ-009, REQ-010, REQ-011).

Runs against REAL infrastructure (throwaway Postgres(pgvector) + MinIO via the
session ``cluster`` harness): RLS-protected ``rag_spaces`` persistence, lazy
provisioning on first scoped use, ``last_used_at`` refresh, cascade deletion of
MinIO objects + tracked ``classical_rag_*`` tables + the space record.

TDD RED: the ``/api/v1/spaces`` routes do not exist yet — every test below
fails (HTTP 404/405, or "relation rag_spaces does not exist" for the SQL
arrangement helpers) until step 2 implements the feature.
"""

import asyncio

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

SPACES = "/api/v1/spaces"


async def poll_until(predicate, timeout_seconds: float = 5.0) -> bool:
    """Poll until ``predicate`` turns True (space writes are backgrounded).

    The lazy space provisioning (REQ-011 + the performance NFR) is
    fire-and-forget: the record write happens off the request's critical
    path, so its visible effect is eventual — the tests poll for it instead
    of assuming synchronous visibility. ``predicate`` may be sync or async.
    """
    import inspect
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return True
        await asyncio.sleep(0.05)
    return await predicate()


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


# ---------------------------------------------------------------------------
# REQ-010: idempotent create + list (caller's spaces only)
# ---------------------------------------------------------------------------


async def test_create_space_is_idempotent_and_listed(infra_api):
    first = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-x"}
    )
    assert first.status_code in (200, 201, 409), first.text

    replay = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-x"}
    )
    assert replay.status_code in (200, 201, 409)

    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    agent_x = [
        entry
        for entry in space_entries(listing)
        if entry.get("agent_slug") == "agent-x"
    ]
    assert len(agent_x) == 1
    record = agent_x[0]
    # REQ-009 columns: user_id, agent_slug, working_dirs, created_at, last_used_at
    assert record.get("user_id") == "user-alice"
    assert record.get("working_dirs") is not None
    assert record.get("created_at") and record.get("last_used_at")


async def test_list_spaces_returns_only_the_callers_spaces(infra_api):
    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "only-alice"}
    )

    listing = await infra_api.get(SPACES, headers=bearer("jwt-bob"))
    assert listing.status_code == 200
    slugs = [entry.get("agent_slug") for entry in space_entries(listing)]
    assert "only-alice" not in slugs


async def test_rejects_invalid_agent_slug_in_the_body_with_422(infra_api):
    response = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "Bad_Name!"}
    )

    assert response.status_code == 422


async def test_create_space_rejects_the_reserved_personal_slug(infra_api, cluster_app):
    """Security review FINDING-3: ``personal`` is NEVER a stored space — the
    create route rejects it (422), so the destructive
    ``DELETE /spaces/personal`` cascade can never find a record to delete."""
    response = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "personal"}
    )

    assert response.status_code == 422, response.text
    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    assert "personal" not in [
        entry.get("agent_slug") for entry in space_entries(listing)
    ], "no personal row may ever be materialized in rag_spaces"
    # The personal tree stays protected by the strict-404 delete gate.
    deleted = await infra_api.request(
        "DELETE", f"{SPACES}/personal", headers=bearer("jwt-alice")
    )
    assert deleted.status_code == 404, deleted.text


async def test_personal_scope_use_never_materializes_a_space_record(
    infra_api, cluster_app
):
    """Security review FINDING-3 (ensure half): the lazy provisioning path
    must also never materialize a ``personal`` record — a personal-scope
    scoped use (no agent header) stores objects without a space row."""
    upload = await infra_api.post(
        "/api/v1/files/upload",
        headers=bearer("jwt-alice"),
        data={"prefix": "personal-notes"},
        files={"file": ("p.txt", b"personal file", "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    assert "personal" not in [
        entry.get("agent_slug") for entry in space_entries(listing)
    ]


async def test_create_space_is_capped_per_user(infra_api, cluster_app, monkeypatch):
    """Security review FINDING-4: a per-user cap on LIVE spaces bounds the
    idempotent-POST abuse surface — the caller's next NEW space answers 422
    with a clear message while the space replay stays idempotent at the cap."""
    import application.use_cases.create_rag_space as create_rag_space_module

    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    current = len(space_entries(listing))

    # One slot of capacity left → exactly one more NEW create is accepted…
    monkeypatch.setattr(create_rag_space_module, "MAX_SPACES_PER_USER", current + 1)
    accepted = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-cap-ok"}
    )
    assert accepted.status_code == 201, accepted.text

    # …the replay of the SAME space still works at the cap (idempotent)…
    replay = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-cap-ok"}
    )
    assert replay.status_code == 201, replay.text

    # …and any NEW space is rejected with 422 + a clear message.
    blocked = await infra_api.post(
        SPACES,
        headers=bearer("jwt-alice"),
        json={"agent_slug": "agent-cap-blocked"},
    )
    assert blocked.status_code == 422, blocked.text
    detail = str(blocked.json().get("detail", ""))
    assert "maximum" in detail and "live spaces" in detail


# ---------------------------------------------------------------------------
# REQ-010: cascade delete (MinIO objects + tracked tables + record)
# ---------------------------------------------------------------------------


def _bearer(credential: str = "jwt-alice") -> dict[str, str]:
    """Credential headers as the real clients present them."""
    return bearer(credential)


class _VectorStoreDouble:
    """Namespace-recording VectorStorePort duck-type (outbound pgvector seam).

    The per-user vector/embeddings factory cannot run against the throwaway
    cluster (the shared ``user_llm_settings`` table is composable-agents-owned
    and absent), and it is NOT the system under test here: the tracking tests
    patch ONLY this outbound seam, while the REST route → use case → scope
    composition → space recorder → REAL ``rag_spaces`` chain stays real.
    """

    def __init__(self) -> None:
        self.tables: list[str] = []
        self.documents: list[tuple[str, list]] = []

    async def ensure_table(self, working_dir: str) -> None:
        self.tables.append(working_dir)

    async def add_documents(self, working_dir: str, documents: list) -> list[str]:
        self.documents.append((working_dir, documents))
        return ["chunk-1"] * len(documents)

    async def similarity_search(self, *args, **kwargs) -> list:
        return []

    async def delete_documents(self, *args, **kwargs) -> int:
        return 0

    async def delete_by_prefix(self, *args, **kwargs) -> int:
        return 0

    async def close(self) -> None:
        return None


@pytest.fixture
def vector_store_double(monkeypatch) -> _VectorStoreDouble:
    """Patch the per-user vector factory to a recording double (C6 test)."""
    double = _VectorStoreDouble()

    async def get_vector_store_for_user():
        return double

    import dependencies

    monkeypatch.setattr(
        dependencies, "get_vector_store_for_user", get_vector_store_for_user
    )
    return double


async def _listed_working_dirs(infra_api, agent_slug: str) -> list[str]:
    """The working_dirs recorded on the caller's space (GET /spaces)."""
    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    assert listing.status_code == 200, listing.text
    for entry in space_entries(listing):
        if entry.get("agent_slug") == agent_slug:
            return list(entry.get("working_dirs") or [])
    return []


async def test_indexing_a_new_project_records_its_working_dir(
    infra_api, cluster_app, vector_store_double
):
    """REQ-008 (review C6): the space MUST track every working_dir the app
    creates — including one indexed WHILE another space write is in flight
    (the exact window the inverted predicate used to drop it)."""

    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-tracked"}
    )
    # A first scoped use (pure refresh) puts a background write in flight…
    upload = await infra_api.post(
        "/api/v1/files/upload",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-tracked"},
        data={"prefix": ""},
        files={"file": ("tick.txt", b"indexable body", "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    # …then a NEW project is indexed while that write may still be in flight.
    indexed = await infra_api.post(
        "/api/v1/classical/file/index",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-tracked"},
        json={"file_name": "tick.txt", "working_dir": "p-new"},
    )
    assert indexed.status_code == 202, indexed.text
    composed = "user-alice/agent-tracked/p-new/"

    async def _composed_and_tracked() -> bool:
        if composed not in vector_store_double.tables:
            return False
        return await _tracked(infra_api, "agent-tracked", composed)

    # The indexing runs in a background task (202): poll for the composed
    # namespace reaching the vector seam AND the record.
    tracked = await poll_until(_composed_and_tracked)
    assert vector_store_double.tables and vector_store_double.tables[-1] == composed, (
        "the use case composed + resolved the NEW namespace before tracking"
    )
    assert tracked, (
        "the NEW working_dir must be recorded in rag_spaces.working_dirs "
        "even when a refresh write was in flight (REQ-008, review C6)"
    )


async def _tracked(infra_api, agent_slug: str, working_dir: str) -> bool:
    return working_dir in await _listed_working_dirs(infra_api, agent_slug)


async def test_indexing_a_seen_project_does_not_write_per_request(
    infra_api, cluster_app, vector_store_double, space_rows, monkeypatch
):
    """The coalescing half of C6: an ALREADY-RECORDED working_dir must not
    schedule one UPSERT per request — the app-recorded path stays real."""
    import dependencies

    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-coalesced"}
    )
    # Index the new project twice (each request passes the SAME working_dir —
    # the composed namespace is already recorded after the first sighting).
    for _index in range(2):
        indexed = await infra_api.post(
            "/api/v1/classical/file/index",
            headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-coalesced"},
            json={
                "file_name": "tick-coalesced.txt",
                "working_dir": "p-seen",
            },
        )
        assert indexed.status_code == 202, indexed.text
    assert await poll_until(
        lambda: _tracked(
            infra_api, "agent-coalesced", "user-alice/agent-coalesced/p-seen/"
        ),
    )

    # Observation seam: count the REPOSITORY upserts the NEXT requests make.
    upserts: list[tuple[str, str, str | None]] = []
    real_repo = dependencies.rag_space_repository
    assert real_repo is not None
    original_ensure = type(real_repo).ensure_user_space

    async def counting_ensure(self, user_id, agent_slug, working_dir=None):
        upserts.append((user_id, agent_slug, working_dir))
        return await original_ensure(self, user_id, agent_slug, working_dir)

    monkeypatch.setattr(type(real_repo), "ensure_user_space", counting_ensure)
    try:
        for _repeat in range(3):
            indexed = await infra_api.post(
                "/api/v1/classical/file/index",
                headers={
                    **bearer("jwt-alice"),
                    "X-Composable-Agent": "agent-coalesced",
                },
                json={"file_name": "tick-coalesced.txt", "working_dir": "p-seen"},
            )
            assert indexed.status_code == 202, indexed.text
        await dependencies.drain_space_touch_tasks()
    finally:
        monkeypatch.setattr(type(real_repo), "ensure_user_space", original_ensure)

    working_dir_key = "user-alice/agent-coalesced/p-seen/"
    space_touches = [write for write in upserts if write[2] == working_dir_key]
    assert len(space_touches) <= 1, (
        f"an already-recorded working_dir must be COALESCED (at most one "
        f"touch per cooldown window), got {len(space_touches)}: {upserts}"
    )


async def test_delete_cascades_objects_tracked_tables_and_record(
    infra_api,
    cluster,
    cluster_app,
    cluster_minio,
    legacy_tables,
    space_rows,
    vector_store_double,
):
    """REQ-009/REQ-008 cascade: MinIO objects, the TRACKED tables and the
    record go; other users' and other spaces' artifacts stay.

    The tracking data is the APP-RECORDED path (review iteration-4 C6 gap):
    the ``p-app`` working_dir is recorded by the real tracking chain (index
    route → use case → space recorder → repository) — only the second table
    is hand-injected to keep the untracked-neighbor case in this one test.
    """
    cluster_minio.put("user-alice/agent-x/docs/deleted-a.txt", b"doomed a")
    cluster_minio.put("user-alice/agent-x/docs/deleted-b.txt", b"doomed b")
    # The file the APP will index into the new p-app project (tracked live).
    cluster_minio.put("user-alice/agent-x/docs/app-tracked.txt", b"app tracked")
    cluster_minio.put("user-alice/agent-y/kept.txt", b"survivor")
    cluster_minio.put("user-bob/agent-x/kept-bob.txt", b"survivor bob")

    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-x"}
    )
    # The hand-injected tracked namespace is seeded FIRST so the app-recorded
    # UPSERT below MERGES its own namespace into the same array (never
    # overwrites it) — both tables must be dropped by the cascade.
    tracked_working_dir = "user-alice/agent-x/p1/"
    await space_rows.set_working_dirs("user-alice", "agent-x", [tracked_working_dir])

    # The APP records the p-app working_dir through the real tracking chain
    # (the vector/embeddings outbound seam is the recording double — see the
    # fixture): this is the path that hid the inverted predicate.
    app_index = await infra_api.post(
        "/api/v1/classical/file/index",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-x"},
        json={"file_name": "docs/app-tracked.txt", "working_dir": "p-app"},
    )
    assert app_index.status_code == 202, app_index.text
    app_tracked_working_dir = "user-alice/agent-x/p-app/"
    recorded = await poll_until(
        lambda: _tracked(infra_api, "agent-x", app_tracked_working_dir),
    )
    assert recorded, "the app-recorded working_dir must land in rag_spaces"

    # The table holding the app-indexed chunks (the double does not touch the
    # real pgvector, so it is seeded by hand at the RECORDED name): the
    # cascade must drop it BECAUSE the app recorded it.
    app_tracked_table = await legacy_tables.create(
        app_tracked_working_dir,
        [
            {
                "file_path": "user-alice/agent-x/p-app/docs/app-tracked.txt",
                "user_id": "user-alice",
            }
        ],
    )
    tracked_table = await legacy_tables.create(
        tracked_working_dir,
        [{"file_path": "user-alice/agent-x/p1/docs/a.txt", "user_id": "user-alice"}],
    )
    untracked_table = await legacy_tables.create(
        "user-alice/agent-y/p1/",
        [{"file_path": "user-alice/agent-y/p1/c.txt", "user_id": "user-alice"}],
    )

    response = await infra_api.request(
        "DELETE", f"{SPACES}/agent-x", headers=bearer("jwt-alice")
    )

    assert response.status_code == 200, response.text
    keys = cluster_minio.object_keys()
    assert "user-alice/agent-x/docs/deleted-a.txt" not in keys
    assert "user-alice/agent-x/docs/deleted-b.txt" not in keys
    assert "user-alice/agent-x/docs/app-tracked.txt" not in keys
    assert "user-alice/agent-y/kept.txt" in keys
    assert "user-bob/agent-x/kept-bob.txt" in keys
    # The table the APP recorded via its tracking chain must be dropped —
    # exactly the REQ-008 guarantee the hand-injected row alone could not pin.
    assert not await legacy_tables.exists(app_tracked_table)
    assert not await legacy_tables.exists(tracked_table)
    assert await legacy_tables.exists(untracked_table)
    assert await space_rows.fetch("user-alice", "agent-x") is None


async def test_delete_nonexistent_space_is_404_without_partial_deletion(
    infra_api, cluster_app, cluster_minio
):
    keys_before = cluster_minio.object_keys()

    response = await infra_api.request(
        "DELETE", f"{SPACES}/never-created", headers=bearer("jwt-alice")
    )

    assert response.status_code == 404
    assert cluster_minio.object_keys() == keys_before


# ---------------------------------------------------------------------------
# REQ-011: lazy provisioning, last_used_at refresh, non-blocking on failure
# ---------------------------------------------------------------------------


async def test_first_scoped_use_lazy_creates_the_space(infra_api, cluster_app):
    upload = await infra_api.post(
        "/api/v1/files/upload",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-lazy"},
        data={"prefix": "docs/"},
        files={"file": ("hello.txt", b"scoped upload", "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    seen = await poll_until(
        lambda: _slug_listed(infra_api, "agent-lazy"),
    )
    assert seen, "the lazy space record must eventually exist (fire-and-forget)"


async def _slug_listed(infra_api, agent_slug: str) -> bool:
    """Whether ``agent_slug`` is listed in the caller's spaces."""
    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    assert listing.status_code == 200, listing.text
    return agent_slug in [entry.get("agent_slug") for entry in space_entries(listing)]


async def test_last_used_at_is_refreshed_by_a_second_scoped_use(infra_api, cluster_app):
    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-used"}
    )
    listing_before = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    before = [
        entry["last_used_at"]
        for entry in space_entries(listing_before)
        if entry.get("agent_slug") == "agent-used"
    ][0]

    upload = await infra_api.post(
        "/api/v1/files/upload",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-used"},
        data={"prefix": ""},
        files={"file": ("tick.txt", b"refresh", "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    # The background write is eventual: poll until the refresh is observable.
    async def refreshed() -> bool:
        listing_after = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
        after = [
            entry["last_used_at"]
            for entry in space_entries(listing_after)
            if entry.get("agent_slug") == "agent-used"
        ][0]
        return after > before

    assert await poll_until(refreshed), (
        "the scoped use must eventually refresh last_used_at (fire-and-forget)"
    )


async def test_space_record_write_failure_never_blocks_the_scoped_request(
    infra_api, cluster, cluster_app
):
    """Rename the rag_spaces table: record writes fail, uploads still succeed."""
    await cluster.pg_exec("ALTER TABLE rag_spaces RENAME TO rag_spaces_shadow")
    try:
        upload = await infra_api.post(
            "/api/v1/files/upload",
            headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-fail"},
            data={"prefix": "docs/"},
            files={"file": ("resilient.txt", b"still works", "text/plain")},
        )
        assert upload.status_code == 201, upload.text
    finally:
        await cluster.pg_exec("ALTER TABLE rag_spaces_shadow RENAME TO rag_spaces")


# ---------------------------------------------------------------------------
# Edge: concurrent first-use → exactly one record (unique (user_id, agent_slug))
# ---------------------------------------------------------------------------


async def test_concurrent_first_use_creates_exactly_one_record(infra_api, cluster_app):
    responses = await asyncio.gather(
        infra_api.post(
            SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-race"}
        ),
        infra_api.post(
            SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-race"}
        ),
    )

    for response in responses:
        assert response.status_code in (200, 201, 409), response.text

    listing = await infra_api.get(SPACES, headers=bearer("jwt-alice"))
    races = [
        entry
        for entry in space_entries(listing)
        if entry.get("agent_slug") == "agent-race"
    ]
    assert len(races) == 1


# ---------------------------------------------------------------------------
# Edge: delete + recreate = fresh empty space (documented no-rename semantics)
# ---------------------------------------------------------------------------


async def test_recreated_space_starts_empty(infra_api, cluster_app):
    await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-cycle"}
    )
    upload = await infra_api.post(
        "/api/v1/files/upload",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-cycle"},
        data={"prefix": ""},
        files={"file": ("old.txt", b"pre-recreate", "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    deleted = await infra_api.request(
        "DELETE", f"{SPACES}/agent-cycle", headers=bearer("jwt-alice")
    )
    assert deleted.status_code == 200

    recreated = await infra_api.post(
        SPACES, headers=bearer("jwt-alice"), json={"agent_slug": "agent-cycle"}
    )
    assert recreated.status_code in (200, 201, 409)

    listing = await infra_api.get(
        "/api/v1/files/list",
        headers={**bearer("jwt-alice"), "X-Composable-Agent": "agent-cycle"},
        params={"prefix": ""},
    )
    assert listing.status_code == 200
    assert listing.json() == []
