"""Unit contract for the coalesced background space tracking (review C6).

Reproduces the reviewer's executable-probe table against the module state
machine, at unit level because the predicate only exists in memory — the
behavioral chain (REST → use case → tracking → REAL ``rag_spaces``) is pinned
by ``test_spaces_api.py::test_indexing_a_new_project_records_its_working_dir``
and the cascading ``test_delete_cascades_objects_tracked_tables_and_record``.

Table (review iteration-4, C6):

case                                   | in-flight | wd seen | schedules write? | expected
working_dir X (first ever sighting)    |   False   |  False  | True             | True
working_dir X (first ever sighting)    |   True    |  False  | True             | True   (REQ-008: never lost)
working_dir X (already tracked)        |   True    |  True   | False            | False  (coalesced)
working_dir X (already tracked)        |   False   |  True   | False            | False  (cooldown)
pure refresh (no working_dir)          |   True    |   n/a   | False            | False
pure refresh (no working_dir)          |   False   |   n/a   | True             | True
"""

import pytest

import dependencies
from domain.entities.rag_space import RagSpace

pytestmark = pytest.mark.asyncio


class FakeSpaceRepository:
    """Recording RagSpaceRepository double (tracking assertions only)."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str, str | None]] = []

    async def ensure_user_space(
        self, user_id: str, agent_slug: str, working_dir: str | None = None
    ) -> RagSpace:
        self.writes.append((user_id, agent_slug, working_dir))
        return RagSpace(user_id=user_id, agent_slug=agent_slug)

    async def get(self, user_id: str, agent_slug: str) -> RagSpace | None:
        return None

    async def list_for_user(self, user_id: str) -> list[RagSpace]:
        return []

    async def delete(self, user_id: str, agent_slug: str) -> bool:
        return True


def _reset_tracking_state(monkeypatch) -> FakeSpaceRepository:
    """Start every probe from a pristine tracking state machine."""
    repository = FakeSpaceRepository()
    monkeypatch.setattr(dependencies, "_space_touch_inflight", set())
    monkeypatch.setattr(dependencies, "_seen_space_working_dirs", set())
    monkeypatch.setattr(dependencies, "_last_space_touch", {})
    monkeypatch.setattr(dependencies, "_space_touch_tasks", set())
    monkeypatch.setattr(dependencies, "rag_space_repository", repository)
    monkeypatch.setattr(
        dependencies,
        "get_request_agent_scope",
        lambda: dependencies.resolve_agent_scope("user-alice", "agent-x"),
    )
    return repository


def test_probe_table_new_working_dir_first_sighting_is_scheduled(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")

    # Row 1 — first-ever sighting, nothing in flight.
    assert dependencies._coalesce_space_touch(key, "user-alice/agent-x/p1/") is True


def test_probe_table_new_working_dir_while_in_flight_is_still_scheduled(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")
    dependencies._space_touch_inflight.add(key)  # a refresh write is running

    # Row 2 — the REQ-008 core: a NEW working_dir must be scheduled EVEN
    # while another write for the space is in flight (never dropped).
    assert dependencies._coalesce_space_touch(key, "user-alice/agent-x/p1/") is True


def test_probe_table_seen_working_dir_in_flight_is_coalesced(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")
    dependencies._seen_space_working_dirs.add(key + ("user-alice/agent-x/p1/",))
    dependencies._space_touch_inflight.add(key)

    # Row 3 — an already-tracked working_dir is absorbed by the in-flight
    # write (no per-request UPSERT).
    assert dependencies._coalesce_space_touch(key, "user-alice/agent-x/p1/") is False


def test_probe_table_seen_working_dir_is_throttled_by_the_cooldown(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")
    dependencies._seen_space_working_dirs.add(key + ("user-alice/agent-x/p1/",))

    # Row 4 — not in flight, but the cooldown recorded a recent write.
    now = dependencies.time.monotonic()
    dependencies._last_space_touch[key] = now
    assert dependencies._coalesce_space_touch(key, "user-alice/agent-x/p1/") is False

    # …and once the cooldown elapses, the refresh IS scheduled again.
    dependencies._last_space_touch[key] = now - 61.0
    assert dependencies._coalesce_space_touch(key, "user-alice/agent-x/p1/") is True


def test_probe_table_pure_refresh_in_flight_is_coalesced(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")
    dependencies._space_touch_inflight.add(key)

    # Row 5 — a pure refresh (no working_dir) is absorbed while in flight.
    assert dependencies._coalesce_space_touch(key, None) is False


def test_probe_table_pure_refresh_after_in_flight_is_scheduled(monkeypatch):
    _reset_tracking_state(monkeypatch)
    key = ("user-alice", "agent-x")

    # Row 6 — a pure refresh with nothing in flight schedules the write.
    assert dependencies._coalesce_space_touch(key, None) is True


async def test_track_space_usage_schedules_the_new_working_dir_write(monkeypatch):
    repository = _reset_tracking_state(monkeypatch)

    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")

    pending = list(dependencies._space_touch_tasks)
    assert pending, "the new working_dir must schedule a background write"
    for task in pending:
        await task
    assert repository.writes == [("user-alice", "agent-x", "user-alice/agent-x/p1/")]


async def test_track_space_usage_coalesces_a_repeat_seen_working_dir(monkeypatch):
    repository = _reset_tracking_state(monkeypatch)

    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    for task in list(dependencies._space_touch_tasks):
        await task
    total = len(repository.writes)

    # A burst of repeats within the cooldown window performs NO extra writes.
    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    dependencies.track_space_usage()  # pure refresh — coalesced as well
    assert len(repository.writes) == total, (
        "seen working_dirs / pure refreshes must be coalesced (no per-request UPSERT)"
    )


async def test_track_space_usage_skips_personal_scope(monkeypatch):
    repository = _reset_tracking_state(monkeypatch)
    # The personal tree carries NO space record (REQ-011/REQ-019).
    monkeypatch.setattr(
        dependencies,
        "get_request_agent_scope",
        lambda: dependencies.resolve_agent_scope("user-alice", "personal"),
    )

    dependencies.track_space_usage(working_dir="user-alice/personal/p1/")

    assert dependencies._space_touch_tasks == set()
    assert repository.writes == []


def test_forget_space_tracking_drops_only_that_space(monkeypatch):
    _reset_tracking_state(monkeypatch)
    dependencies._seen_space_working_dirs.update(
        {
            ("user-alice", "agent-x", "wd-x"),
            ("user-alice", "agent-y", "wd-y"),
            ("user-bob", "agent-x", "wd-bob"),
        }
    )

    dependencies.forget_space_tracking("user-alice", "agent-x")

    assert dependencies._seen_space_working_dirs == {
        ("user-alice", "agent-y", "wd-y"),
        ("user-bob", "agent-x", "wd-bob"),
    }


async def test_failed_write_re_arms_the_seen_marker(monkeypatch):
    repository = _reset_tracking_state(monkeypatch)

    async def failing_ensure(*_args, **_kwargs) -> RagSpace | None:
        return None  # the record write failed (best-effort contract)

    monkeypatch.setattr(dependencies.EnsureRagSpaceUseCase, "execute", failing_ensure)
    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    for task in list(dependencies._space_touch_tasks):
        await task

    # The failed write forgot the marker: the NEXT scoped use retries REQ-011.
    assert ("user-alice", "agent-x", "user-alice/agent-x/p1/") not in (
        dependencies._seen_space_working_dirs
    )

    writes_before_retry = len(repository.writes)

    # Restore a working ensure and verify the retry is scheduled as a NEW write.
    async def working_ensure(
        self, *, user_id, agent_slug, working_dir=None
    ) -> RagSpace:
        repository.writes.append((user_id, agent_slug, working_dir))
        return RagSpace(user_id=user_id, agent_slug=agent_slug)

    monkeypatch.setattr(dependencies.EnsureRagSpaceUseCase, "execute", working_ensure)
    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    for task in list(dependencies._space_touch_tasks):
        await task
    assert len(repository.writes) > writes_before_retry, (
        "a failed tracking write must be retried on the next scoped use (REQ-011)"
    )


async def test_drain_space_touch_tasks_awaits_pending_writes(monkeypatch):
    repository = _reset_tracking_state(monkeypatch)

    dependencies.track_space_usage(working_dir="user-alice/agent-x/p1/")
    await dependencies.drain_space_touch_tasks()

    assert repository.writes, "the drain must let the pending write complete"
    assert all(task.done() for task in dependencies._space_touch_tasks), (
        "no pending space-write task may survive the drain"
    )
