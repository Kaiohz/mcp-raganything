#!/usr/bin/env python3
"""Scope migration: move pre-scope RAG data into per-user × agent spaces.

scope-rag-memories REQ-024 maintenance script. The production migration MUST
run with a DRY-RUN first and its report reviewed (decided), then be re-run
with ``--apply`` inside the deployment maintenance window.

What it does (on the ``classical_rag_*`` corpus + the whole MinIO bucket):

(a) reads every ``classical_rag_*`` table's chunk metadata and maps each
    indexed ``file_path`` to its owning ``user_id`` (the index-time tag);
(b) moves the corresponding MinIO objects into the composed PERSONAL tree
    ``{user_id}/personal/{key}`` (the pre-feature world was per-user only:
    every legacy file belongs to the caller's ``personal`` agent scope);
(c) carries each mapped chunk into the recomposed working_dir table
    ``classical_rag_<sha256("{user_id}/personal/{dir}/")[:16]>`` with the
    content preserved and the metadata rewritten to the NEW file path and the
    ``user_id`` tag RETAINED — the exact end-state the indexing pipeline
    produces (composed working_dir table + new file paths + per-user tag).
    Vector re-embedding is a follow-up full-re-embed task; the HTTP
    re-indexing route is deliberately NOT driven (it requires the users'
    live embedding credentials and its langchain-shaped table layout differs
    from the raw corpus tables), so the migration keeps the observed chunk
    schema (id/content/metadata/embedding);
(d) moves objects with NO ``user_id`` mapping (never indexed) into
    ``_unscoped/{key}`` and lists them in the report for manual triage;
    ``chat_tmp/`` objects are IGNORED (ephemeral, already user-scoped); a
    mapped file whose MinIO object disappeared is REPORTED and SKIPPED
    without aborting.

Dry-run (the DEFAULT) mutates nothing and only prints the report; ``--apply``
executes the moves, the chunk carry and the old-table drops. Environment
(infra harness contract): ``DATABASE_URL``, ``MINIO_HOST``, ``MINIO_ACCESS``,
``MINIO_SECRET``, ``MINIO_BUCKET`` and optionally ``MINIO_SECURE``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import posixpath
import sys
import time
from dataclasses import dataclass, field

import asyncpg

TABLE_PREFIX = "classical_rag_"
PERSONAL_SLUG = "personal"
UNSCOPED_PREFIX = "_unscoped/"
CHAT_TMP_PREFIX = "chat_tmp/"
DEFAULT_BUCKET = "raganything"
DEFAULT_EMBEDDING_DIM = 1536


@dataclass
class MigrationReport:
    """The human-readable migration plan/execution report."""

    tagged: list[tuple[str, str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unscoped: list[str] = field(default_factory=list)
    ignored_chat_tmp: list[str] = field(default_factory=list)
    already_scoped: list[str] = field(default_factory=list)
    tables: list[tuple[str, int, int]] = field(default_factory=list)
    dropped_tables: list[str] = field(default_factory=list)
    embeddings_carried: int = 0
    embeddings_missing: int = 0

    def render(self, *, apply: bool) -> str:
        """Render the report to the printable summary lines."""
        lines: list[str] = []
        mode = "APPLY" if apply else "DRY-RUN (no mutation)"
        lines.append(f"=== SCOPE MIGRATION REPORT — {mode} ===")
        lines.append(
            f"[tagged] {len(self.tagged)} file(s) mapped to a user "
            "(move target {user_id}/personal/<file>):"
        )
        for file_path, user, target in self.tagged:
            lines.append(f"  {file_path}  user_id={user}  ->  {target}")
        if self.missing:
            lines.append(
                f"[missing] {len(self.missing)} mapped object(s) NOT found in "
                "the bucket (reported and skipped, no abort):"
            )
            for key in self.missing:
                lines.append(f"  {key}")
        lines.append(
            f"[{UNSCOPED_PREFIX}] {len(self.unscoped)} object(s) with NO user_id "
            "mapping (never indexed) — move target _unscoped/<file> (manual triage):"
        )
        for key in self.unscoped:
            lines.append(f"  {key}")
        if self.ignored_chat_tmp:
            lines.append(
                f"[chat_tmp] {len(self.ignored_chat_tmp)} ephemeral object(s) "
                "ignored (already user-scoped): " + ", ".join(self.ignored_chat_tmp)
            )
        if self.already_scoped:
            lines.append(
                f"[already-scoped] {len(self.already_scoped)} object(s)/table(s) "
                "already living under a composed scope (previous run) — left "
                "untouched."
            )
        for table, total, carried in self.tables:
            lines.append(
                f"[table] {table}: {carried}/{total} chunk(s) carried into the "
                "recomposed working_dir tables"
            )
        if apply:
            lines.append(
                f"[embedding] {self.embeddings_carried} vector(s) carried into "
                f"the recomposed tables ({self.embeddings_missing} source "
                "row(s) without an embedding carried as NULL — schedule a "
                "full re-embed pass for those)"
            )
        if self.dropped_tables:
            lines.append("[dropped] old tables: " + ", ".join(self.dropped_tables))
        lines.append("=== END OF REPORT ===")
        return "\n".join(lines)


def classical_table(working_dir: str) -> str:
    """Resolve the recomposed working_dir's pgvector table name."""
    digest = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
    return f"{TABLE_PREFIX}{digest}"


def scoped_personal_working_dir(user_id: str, file_path: str) -> str:
    """Compose the personal working_dir for a legacy file path.

    The legacy file path's directory prefix becomes the recomposed
    working_dir inside the user's ``personal`` scope
    (``{user_id}/personal/<dir>/``); root-level files compose to the user's
    personal root namespace.
    """
    directory = posixpath.dirname(file_path)
    if directory:
        return f"{user_id}/{PERSONAL_SLUG}/{directory}/"
    return f"{user_id}/{PERSONAL_SLUG}/"


def is_scoped_personal_path(path: str) -> bool:
    """Whether a chunk ``file_path``/bucket key is ALREADY scope-composed.

    The recomposed tree shape is ``{user_id}/personal/{rest}`` — produced by
    THIS script (or the scoped pipeline). Idempotent re-runs must never treat
    those keys/rows as pre-scope data again.
    """
    segments = path.split("/")
    return len(segments) >= 3 and segments[1] == PERSONAL_SLUG


@dataclass
class ClassicalCorpus:
    """The pre-scope corpus state read from PostgreSQL."""

    dimensions: dict[str, int] = field(default_factory=dict)
    chunks: dict[str, list[tuple[str, str | None, str | None, dict]]] = field(
        default_factory=dict
    )
    file_users: dict[str, str] = field(default_factory=dict)
    recomposed_tables: set[str] = field(default_factory=set)


async def load_classical_corpus(conn) -> ClassicalCorpus:
    """Read every classical table: embedding dims, chunks, file→user mapping.

    Tables whose rows are ALL already scope-composed (a previous run's
    recomposed output) are excluded from the corpus: re-running ``--apply``
    is idempotent — they are neither carried nor dropped again.
    """
    corpus = ClassicalCorpus()
    table_records = await conn.fetch(
        """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public' AND tablename LIKE 'classical_rag_%'
        ORDER BY tablename;
        """
    )
    for record in table_records:
        table = record["tablename"]
        rows = await conn.fetch(
            f"SELECT content, metadata::text, embedding::text FROM {table}"  # noqa: S608
        )
        parsed: list[tuple[str, str | None, str | None, dict]] = []
        legacy_rows = 0
        scoped_rows = 0
        for row in rows:
            metadata = _parse_metadata(row[1])
            file_path = metadata.get("file_path")
            if isinstance(file_path, str) and is_scoped_personal_path(file_path):
                scoped_rows += 1
                continue
            raw_user = metadata.get("user_id")
            user_id = raw_user if isinstance(raw_user, str) and raw_user else None
            # code-review C3: the EMBEDDING is part of the carried chunk — a
            # row carried without its vector is unsearchable in the recomposed
            # table (a NULL vector never satisfies a cosine ordering), so the
            # migration must never silently lose retrievability.
            embedding = str(row[2]) if row[2] is not None else None
            parsed.append((str(row[0] or ""), user_id, embedding, metadata))
            legacy_rows += 1
            if file_path and user_id:
                corpus.file_users.setdefault(str(file_path), user_id)
        if scoped_rows and not legacy_rows:
            # All rows already scope-composed: this table is a PREVIOUS run's
            # recomposed output — excluded from the corpus entirely so re-runs
            # neither carry it again nor drop the composed table.
            corpus.recomposed_tables.add(table)
            continue
        corpus.dimensions[table] = await _table_dimension(conn, table)
        corpus.chunks[table] = parsed
    return corpus


def _parse_metadata(raw: object) -> dict:
    """Parse one chunk's metadata text (malformed metadata → empty dict)."""
    if isinstance(raw, dict):
        return raw
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


async def _table_dimension(conn, table: str) -> int:
    """Parse the table's ``vector(N)`` dimension (default when unknown)."""
    record = await conn.fetchrow(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS dims
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = $1
          AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped
        LIMIT 1;
        """,
        table,
    )
    if record is None:
        return DEFAULT_EMBEDDING_DIM
    raw = str(record["dims"]).replace("vector(", "").rstrip(")")
    with contextlib.suppress(ValueError):
        return int(raw)
    return DEFAULT_EMBEDDING_DIM


def object_keys(client, bucket: str) -> list[str]:
    """List every object key in the bucket (recursive)."""
    return sorted(
        obj.object_name for obj in client.list_objects(bucket, recursive=True)
    )


def object_exists(client, bucket: str, key: str) -> bool:
    """Whether ``key`` is present (a missing object is reported, not fatal)."""
    try:
        client.stat_object(bucket, key)
        return True
    except Exception:  # noqa: BLE001 — S3Error/network: treated as missing
        return False


async def _create_embedding_index(conn, table: str) -> None:
    """Create the HNSW ANN index on a recomposed table (best-effort).

    The pgvector backend of a recomposed working_dir is queried with cosine
    similarity by the classical pipeline; without the ANN index every query
    degenerates to a sequential scan (code-review 🟡-2). Best-effort on
    purpose: a backend without HNSW support must never abort the migration.
    """
    with contextlib.suppress(Exception):
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"  # noqa: S608
        )


def move_object(client, bucket: str, key: str, target: str) -> None:
    """Copy the object to ``target`` then remove the original key."""
    response = client.get_object(bucket, key)
    try:
        data = response.read()
        content_type = response.headers.get("Content-Type", "application/octet-stream")
    finally:
        response.close()
        response.release_conn()
    client.put_object(bucket, target, io.BytesIO(data), len(data), content_type)
    client.remove_object(bucket, key)


async def run_migration(
    conn,
    client,
    bucket: str,
    corpus: ClassicalCorpus,
    report: MigrationReport,
    apply: bool,
) -> None:
    """Map bucket objects, (optionally) move them, carry chunks, drop tables."""
    moved_targets: set[str] = set()
    keys = object_keys(client, bucket)
    key_set = set(keys)

    # (a)+(b) Mapped files first — EVERY chunk-mapped file_path is reported,
    # including the ones whose MinIO object DISAPPEARED (reported as missing
    # and skipped without aborting, REQ-024 (d)).
    for file_path, user_id in corpus.file_users.items():
        if file_path.startswith(CHAT_TMP_PREFIX):
            report.ignored_chat_tmp.append(file_path)
            continue
        target = f"{user_id}/{PERSONAL_SLUG}/{file_path}"
        report.tagged.append((file_path, user_id, target))
        if file_path not in key_set:
            report.missing.append(file_path)
        elif apply:
            move_object(client, bucket, file_path, target)
            moved_targets.add(target)

    # Untagged / never-indexed bucket objects → _unscoped/ (manual triage).
    # Already-scoped objects (a previous run's recomposed tree) and previously
    # triaged _unscoped objects are left untouched (idempotent re-runs).
    for key in keys:
        if key.startswith(CHAT_TMP_PREFIX):
            report.ignored_chat_tmp.append(key)
        elif is_scoped_personal_path(key) or key.startswith(UNSCOPED_PREFIX):
            report.already_scoped.append(key)
        elif key in corpus.file_users:
            continue  # handled (moved or reported missing) in the mapped pass
        else:
            report.unscoped.append(key)
            if apply and object_exists(client, bucket, key):
                move_object(client, bucket, key, f"{UNSCOPED_PREFIX}{key}")

    if not apply:
        for table, chunks in corpus.chunks.items():
            report.tables.append((table, len(chunks), 0))
        return

    for table, chunks in corpus.chunks.items():
        carried = 0
        for content, user_id, embedding, metadata in chunks:
            file_path = str(metadata.get("file_path") or "")
            if not user_id or not file_path:
                continue  # untagged pool: object moved to _unscoped/, no carry
            target_key = f"{user_id}/{PERSONAL_SLUG}/{file_path}"
            if target_key not in moved_targets:
                continue  # ghost object: reported above, chunk skipped
            new_table = classical_table(scoped_personal_working_dir(user_id, file_path))
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {new_table} (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    content text,
                    metadata jsonb,
                    embedding vector({corpus.dimensions.get(table, DEFAULT_EMBEDDING_DIM)})
                )
                """
            )
            # code-review 🟡-2: langchain-postgres ships an ANN index for every
            # table it creates; the recomposed corpus must not fall back to
            # sequential scans. Best-effort: an unavailable HNSW index never
            # aborts the migration (the chunks stay queryable either way).
            await _create_embedding_index(conn, new_table)
            new_metadata = {**metadata, "file_path": target_key, "user_id": user_id}
            await conn.execute(
                f"""
                INSERT INTO {new_table} (content, metadata, embedding)
                VALUES ($1, $2::jsonb, $3::text::vector)
                """,  # noqa: S608
                content,
                json.dumps(new_metadata),
                embedding,
            )
            if embedding is not None:
                report.embeddings_carried += 1
            else:
                report.embeddings_missing += 1
            carried += 1
        report.tables.append((table, len(chunks), carried))
        report.dropped_tables.append(table)

    for table in corpus.chunks:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608


async def async_main(apply: bool) -> int:
    """Entry point: read env, load the corpus, run the (dry) migration."""
    dsn = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "")
    host = os.environ.get("MINIO_HOST", "")
    access = os.environ.get("MINIO_ACCESS", "")
    secret = os.environ.get("MINIO_SECRET", "")
    bucket = os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET)
    secure = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
    if not dsn:
        print("FATAL: DATABASE_URL is not set.")
        return 1
    if not host or not access:
        print("FATAL: MINIO_HOST / MINIO_ACCESS are not set.")
        return 1

    from minio import Minio  # noqa: PLC0415 — script-local import

    conn = await asyncpg.connect(dsn=dsn)
    try:
        corpus = await load_classical_corpus(conn)
        client = Minio(host, access_key=access, secret_key=secret, secure=secure)
        report = MigrationReport()
        await run_migration(conn, client, bucket, corpus, report, apply)
        rendered = report.render(apply=apply)
        print(rendered)
        report_path = write_report_file(rendered, apply)
        if report_path:
            print(f"report written to {report_path}")
        if apply and corpus.chunks and report.missing:
            print(
                f"NOTE: {len(report.missing)} mapped object(s) were missing and "
                "skipped — reviewed in the report above (no abort)."
            )
        return 0
    finally:
        await conn.close()


def write_report_file(rendered: str, apply: bool) -> str | None:
    """Persist the report to a file (REQ-024 (d) — manual triage artifact).

    ``MIGRATION_REPORT_PATH`` overrides the default
    ``scope-migration-<timestamp>.report.txt`` (written next to the console
    output). Return the written path, or ``None`` when the write failed — a
    report file must never abort a half-applied maintenance run.
    """
    override = os.environ.get("MIGRATION_REPORT_PATH", "").strip()
    report_path = override or (
        "scope-migration-"
        + time.strftime("%Y%m%dT%H%M%S")
        + (".apply" if apply else "")
        + ".report.txt"
    )
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        return report_path
    except OSError as exc:
        print(f"NOTE: the report file {report_path!r} could not be written: {exc}")
        return None


def main_cli() -> int:
    """Parse CLI arguments and run the migration (dry-run by default)."""
    parser = argparse.ArgumentParser(
        description=(
            "Migrate pre-scope RAG data (classical_rag_* tables + MinIO "
            "objects) into per-user × agent scopes. Dry-run by default; "
            "pass --apply to execute."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the migration (default: dry-run report only).",
    )
    args = parser.parse_args()
    return asyncio.run(async_main(apply=args.apply))


if __name__ == "__main__":
    sys.exit(main_cli())
