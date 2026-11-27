"""Behavioral contract for the scope-migration script (REQ-024) — TDD RED.

``scripts/migrate_scopes.py`` must (on fixture data seeded into REAL MinIO +
pgvector):

* (a) read each ``classical_rag_*`` table's chunk metadata to map indexed files
  to their ``user_id``;
* (b) move the corresponding MinIO objects to the ``personal`` scoped tree
  (``{user_id}/personal/...``);
* (c) drop the old tables and re-index the moved files through the existing
  indexing routes so chunks carry the new composed working_dir and file paths;
* (d) move objects with NO ``user_id`` mapping into ``_unscoped/`` and write a
  report listing them for manual triage;
* ignore ``chat_tmp/`` objects (ephemeral, already user-scoped);
* skip (report) objects whose MinIO object is missing without aborting;
* DEFAULT to dry-run (no mutation); ``--apply`` executes.

The script is invoked as a subprocess over the harness environment (the
``script_runner`` fixture injects ``DATABASE_URL``, ``MINIO_*`` plus
``MIGRATION_BASE_URL`` / ``MIGRATION_BEARER_TOKEN`` for the served application
when the re-indexing drives the HTTP surfaces) — see the infra conftest
docstring.

TDD RED: the script does not exist yet — the subprocess invocation fails and
every test below must fail for that reason.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_PATH = "scripts/migrate_scopes.py"

# Fixture data (seeded by ``migrated_corpus``):
LEGACY_TAGGED_WORKING_DIR = "docs/alice/"  # pre-feature working_dir of the chunks
TAGGED_FILE_A = "docs/alice/report.txt"
TAGGED_FILE_B = "docs/alice/notes.txt"
GHOST_FILE = "docs/alice/ghost.txt"  # chunk metadata with NO MinIO object
UNTAGGED_FILE = "docs/nobody/mystery.txt"  # never indexed → no user_id mapping
CHAT_TMP_FILE = "chat_tmp/user-alice/session/x.txt"  # must be ignored
COMPOSED_REINDEX_DIR = "user-alice/personal/docs/alice/"


@pytest.fixture
def script_runner(cluster):
    """Run the migration script as a subprocess over the harness environment."""

    def run(*arguments: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(
            {
                "DATABASE_URL": cluster.pg_dsn,
                "MINIO_HOST": cluster.minio_host,
                "MINIO_ACCESS": cluster.minio_access,
                "MINIO_SECRET": cluster.minio_secret,
                "MINIO_BUCKET": "raganything",
                "MIGRATION_BASE_URL": getattr(
                    cluster, "http_base_url", "http://127.0.0.1:0"
                ),
                "MIGRATION_BEARER_TOKEN": getattr(cluster, "http_bearer", "jwt-alice"),
                # The script writes its REQ-024 (d) triage report FILE; keep
                # test runs from polluting the repo worktree.
                "MIGRATION_REPORT_PATH": os.path.join(
                    tempfile.gettempdir(), "scope-migration-report.txt"
                ),
            }
        )
        return subprocess.run(
            [sys.executable, SCRIPT_PATH, *arguments],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

    return run


@pytest.fixture
async def migrated_corpus(cluster, cluster_app, cluster_minio, legacy_tables):
    """Seed the pre-feature world: indexed chunks + MinIO objects + chat_tmp."""
    tagged_table = await legacy_tables.create(
        LEGACY_TAGGED_WORKING_DIR,
        [
            {"file_path": TAGGED_FILE_A, "user_id": "user-alice"},
            {"file_path": TAGGED_FILE_B, "user_id": "user-alice"},
            {"file_path": GHOST_FILE, "user_id": "user-alice"},
        ],
    )
    # A second legacy table with chunks carrying NO user_id (untagged pool).
    untagged_table = await legacy_tables.create(
        "legacy-unscoped/",
        [{"file_path": UNTAGGED_FILE, "user_id": None}],
    )
    cluster_minio.put(TAGGED_FILE_A, b"alice report pre-migration")
    cluster_minio.put(TAGGED_FILE_B, b"alice notes pre-migration")
    # GHOST_FILE intentionally NOT seeded (missing-object edge).
    cluster_minio.put(UNTAGGED_FILE, b"nobody indexed me")
    cluster_minio.put(CHAT_TMP_FILE, b"ephemeral chat attachment")
    keys_before = cluster_minio.object_keys()
    yield {
        "tagged_table": tagged_table,
        "untagged_table": untagged_table,
        "keys_before": keys_before,
    }
    # Best-effort cleanup: the table seed helper may have left artifacts.
    await legacy_tables.drop(tagged_table)
    await legacy_tables.drop(untagged_table)


def combined_report(result: subprocess.CompletedProcess) -> str:
    return result.stdout + "\n" + result.stderr


def moved_personal_key(cluster_minio, original_key: str) -> str | None:
    """Find the migrated key for ``original_key`` (user/personal order-agnostic)."""
    candidates = [
        key
        for key in cluster_minio.object_keys()
        if key.endswith(original_key)
        and key != original_key
        and "user-alice" in key
        and "personal" in key
    ]
    return candidates[0] if candidates else None


async def chunk_rows(cluster, table: str) -> list:
    conn = await cluster.pg()
    try:
        return await conn.fetch(
            f"SELECT content, metadata, embedding::text AS embedding FROM {table}"
        )
    finally:
        await conn.close()


def assert_embeddings_carried(rows) -> None:
    """Every carried chunk must keep its vector (code-review C3).

    A NULL embedding can never satisfy a cosine-distance ordering, so a
    migrated row without its vector is silently unsearchable — the migration
    must carry the embedding through the recomposed tables.
    """
    non_null = [row for row in rows if row["embedding"] is not None]
    assert rows, "no chunks carried into the recomposed table"
    assert len(non_null) == len(rows), (
        f"the migration dropped the embedding of {len(rows) - len(non_null)} "
        f"chunk(s): {[(row['content'][:30]) for row in rows]}"
    )


# ---------------------------------------------------------------------------
# Dry-run (the DEFAULT): report everything, mutate nothing
# ---------------------------------------------------------------------------


async def test_dry_run_reports_mappings_and_mutates_nothing(
    cluster, cluster_app, script_runner, migrated_corpus, cluster_minio, legacy_tables
):
    result = script_runner()

    assert result.returncode == 0, combined_report(result)
    report = combined_report(result)
    # The tagged files are reported with their owning user (mapping review).
    assert TAGGED_FILE_A in report and "user-alice" in report
    assert TAGGED_FILE_B in report
    # The never-indexed objects are listed as _unscoped candidates.
    assert UNTAGGED_FILE in report and "_unscoped" in report
    # NOTHING was mutated: original keys, no personal/ target, tables intact.
    assert cluster_minio.object_keys() == migrated_corpus["keys_before"]
    assert not any("personal" in key for key in cluster_minio.object_keys())
    assert await legacy_tables.exists(migrated_corpus["tagged_table"])
    assert await legacy_tables.exists(migrated_corpus["untagged_table"])


async def test_dry_run_writes_the_triage_report_file(
    cluster, cluster_app, migrated_corpus, cluster_minio, tmp_path
):
    """REQ-024 (d, review iteration-4 🟡-5): the migration must WRITE the
    triage report FILE — not only print it — so an operator reviewing the
    dry-run keeps an artifact listing the ``_unscoped``/ghost candidates."""
    import sys as _sys

    report_path = tmp_path / "scope-migration-report.txt"

    def run(env_path: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(
            {
                "DATABASE_URL": cluster.pg_dsn,
                "MINIO_HOST": cluster.minio_host,
                "MINIO_ACCESS": cluster.minio_access,
                "MINIO_SECRET": cluster.minio_secret,
                "MINIO_BUCKET": "raganything",
                "MIGRATION_BASE_URL": getattr(
                    cluster, "http_base_url", "http://127.0.0.1:0"
                ),
                "MIGRATION_BEARER_TOKEN": getattr(cluster, "http_bearer", "jwt-alice"),
                "MIGRATION_REPORT_PATH": env_path,
            }
        )
        return subprocess.run(
            [_sys.executable, SCRIPT_PATH],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert bool(migrated_corpus), "the corpus fixture must have seeded data"

    result = run(str(report_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert report_path.exists(), "the REQ-024 (d) report file must be written"
    content = report_path.read_text(encoding="utf-8")
    assert UNTAGGED_FILE in content, (
        "the report file must list the never-indexed candidates"
    )
    assert "_unscoped" in content
    assert GHOST_FILE in content, (
        "the report file must list the mapped-but-missing ghost objects"
    )


# ---------------------------------------------------------------------------
# --apply: moves, drops, reindexes, reports — never aborts on a missing object
# ---------------------------------------------------------------------------


async def test_apply_moves_tagged_files_to_the_personal_scoped_tree(
    script_runner, migrated_corpus, cluster_minio, legacy_tables, cluster_app
):
    result = script_runner("--apply")

    assert result.returncode == 0, combined_report(result)
    keys = cluster_minio.object_keys()
    assert TAGGED_FILE_A not in keys
    assert TAGGED_FILE_B not in keys
    moved = moved_personal_key(cluster_minio, TAGGED_FILE_A)
    assert moved is not None, (
        f"no migrated key for {TAGGED_FILE_A} under the personal tree; keys={keys}"
    )
    assert moved_personal_key(cluster_minio, TAGGED_FILE_B) is not None
    # The old table is dropped; the re-index creates the composed table.
    assert not await legacy_tables.exists(migrated_corpus["tagged_table"])
    assert await legacy_tables.exists(legacy_tables.table_for(COMPOSED_REINDEX_DIR)), (
        "re-indexing must create the composed working_dir table"
    )


async def test_apply_reindexes_moved_files_with_user_tag_and_new_paths(
    cluster, script_runner, migrated_corpus, legacy_tables, cluster_app
):
    result = script_runner("--apply")

    assert result.returncode == 0, combined_report(result)
    table = legacy_tables.table_for(COMPOSED_REINDEX_DIR)
    rows = await chunk_rows(cluster, table)
    assert rows, f"no chunks re-indexed into {table}"
    metadata = (
        json.loads(rows[0]["metadata"])
        if isinstance(rows[0]["metadata"], str)
        else rows[0]["metadata"]
    )
    assert metadata.get("user_id") == "user-alice"
    assert "personal" in (metadata.get("file_path") or "")
    # code-review C3: the EMBEDDING must survive the migration — a carried
    # chunk without its vector is silently unsearchable.
    assert_embeddings_carried(rows)


async def test_apply_reports_missing_object_and_continues(
    script_runner, migrated_corpus, cluster_minio, legacy_tables, cluster_app
):
    result = script_runner("--apply")

    assert result.returncode == 0, combined_report(result)
    report = combined_report(result)
    # GHOST_FILE has chunk metadata but no MinIO object: reported + skipped.
    assert GHOST_FILE in report
    # The migration did NOT abort: the other file still migrated.
    assert TAGGED_FILE_A not in cluster_minio.object_keys()
    assert not await legacy_tables.exists(migrated_corpus["tagged_table"])


async def test_apply_moves_untagged_files_into_unscoped_with_report(
    script_runner, migrated_corpus, cluster_minio, cluster_app
):
    result = script_runner("--apply")

    assert result.returncode == 0, combined_report(result)
    keys = cluster_minio.object_keys()
    assert UNTAGGED_FILE not in keys
    unscoped = [
        key for key in keys if "_unscoped" in key and key.endswith(UNTAGGED_FILE)
    ]
    assert unscoped, f"untagged object was not moved to _unscoped; keys={keys}"


async def test_apply_ignores_chat_tmp_objects(
    script_runner, migrated_corpus, cluster_minio, cluster_app
):
    result = script_runner("--apply")

    assert result.returncode == 0, combined_report(result)
    assert CHAT_TMP_FILE in cluster_minio.object_keys(), (
        "chat_tmp objects must stay untouched"
    )
