"""Behavioral contract for file scoping (REQ-003, REQ-004, REQ-005) — TDD RED.

Every file operation — ``upload_file``, ``list_files``, ``list_folders``,
``read_file``, ``create_folder``, ``delete_file``, ``delete_folder`` (MCP tools
and REST routes) — must be prefixed with ``{user_id}/{agent_slug}/``, treating
the client-supplied path/prefix as relative to the scope; ``list_*`` returns
only objects under the caller's scope prefix (intra-user agent isolation AND
cross-user isolation). The temporary chat-attachment policy
(``chat_tmp/<user_id>/...``) is preserved unchanged (REQ-004).

All tests go through the real MCP tools (mounted ``/files/mcp``) and REST
routes; only the outbound MinIO boundary is a recording double.
"""

import base64
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

REST = "/api/v1/files"
FILES_MOUNT = "/files/mcp"
AGENT_HEADER = "X-Composable-Agent"
ALICE = {"Authorization": "Bearer jwt-alice"}
BOB = {"Authorization": "Bearer jwt-bob"}
WITH_AGENT_X = {**ALICE, AGENT_HEADER: "agent-x"}
WITH_AGENT_Y = {**ALICE, AGENT_HEADER: "agent-y"}

CROSS_SCOPE_OBJECTS = {
    "user-alice/personal/notes.txt": b"alice personal",
    "user-alice/agent-x/plan.md": b"alice agent-x",
    "user-alice/agent-y/other.md": b"alice agent-y",
    "user-alice/agent-x/docs/report.txt": b"alice agent-x docs",
    "user-bob/personal/own.txt": b"bob personal",
    "user-bob/agent-x/task.md": b"bob agent-x",
}


def error_text(result) -> str:
    return "\n".join(getattr(item, "text", "") for item in (result.content or []))


def tool_rows(result) -> list[dict]:
    assert result.is_error is False, error_text(result)
    data = result.data
    if data is None:
        return []
    if not isinstance(data, list):
        data = [data]
    return [item if isinstance(item, dict) else dict(vars(item)) for item in data]


# ---------------------------------------------------------------------------
# upload_file (REST) — REQ-003
# ---------------------------------------------------------------------------


async def test_rest_upload_stores_object_under_the_caller_scope_prefix(
    api, storage_external
):
    response = await api.post(
        f"{REST}/upload",
        headers=WITH_AGENT_X,
        data={"prefix": "docs/"},
        files={"file": ("hello.txt", b"hello scoped world", "text/plain")},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["object_name"] == "user-alice/agent-x/docs/hello.txt"
    stored = storage_external.calls_for("put_object")
    assert stored and stored[-1]["object_path"] == "user-alice/agent-x/docs/hello.txt"


async def test_rest_upload_without_header_lands_in_personal_scope(
    api, storage_external
):
    response = await api.post(
        f"{REST}/upload",
        headers=ALICE,
        data={"prefix": ""},
        files={"file": ("notes.txt", b"personal note", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["object_name"] == "user-alice/personal/notes.txt"


async def test_rest_upload_rejects_traversal_even_inside_the_scope(
    api, storage_external
):
    response = await api.post(
        f"{REST}/upload",
        headers=WITH_AGENT_X,
        data={"prefix": "../../escape"},
        files={"file": ("bad.txt", b"x", "text/plain")},
    )

    assert response.status_code == 422
    assert storage_external.calls_for("put_object") == []


# ---------------------------------------------------------------------------
# list_files / list_folders (MCP) — isolation matrix (REQ-005)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("credential", "agent", "expected"),
    [
        (
            "jwt-alice",
            "agent-x",
            ["user-alice/agent-x/docs/report.txt", "user-alice/agent-x/plan.md"],
        ),
        ("jwt-alice", "agent-y", ["user-alice/agent-y/other.md"]),
        ("jwt-alice", None, ["user-alice/personal/notes.txt"]),
        ("jwt-bob", "agent-x", ["user-bob/agent-x/task.md"]),
    ],
)
async def test_mcp_list_files_isolation_matrix_user_x_agent(
    mcp_call, storage_external, credential, agent, expected
):
    storage_external.seed(CROSS_SCOPE_OBJECTS)
    headers = {"Authorization": f"Bearer {credential}"}
    if agent:
        headers[AGENT_HEADER] = agent

    result = await mcp_call(FILES_MOUNT, "list_files", {"prefix": ""}, headers=headers)

    rows = tool_rows(result)
    assert [row["object_name"] for row in rows] == sorted(expected)


async def test_mcp_list_folders_is_scoped(mcp_call, storage_external):
    storage_external.seed(
        {
            "user-alice/agent-x/docs/report.txt": b"a",
            "user-alice/agent-x/photos/img.png": b"b",
            "user-bob/agent-x/private.txt": b"c",
        }
    )

    result = await mcp_call(
        FILES_MOUNT, "list_folders", {"prefix": ""}, headers=WITH_AGENT_X
    )

    assert result.is_error is False
    assert list(result.data) == ["docs/", "photos/"]


# ---------------------------------------------------------------------------
# list_files / list_folders (REST) — isolation matrix (REQ-005)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (WITH_AGENT_X, ["user-alice/agent-x/plan.md"]),
        (WITH_AGENT_Y, []),
        (ALICE, ["user-alice/personal/notes.txt"]),
        # Bob with agent-x sees ONLY his own fresh scope: his own object is
        # visible, none of Alice's data ever is (same contract as the MCP
        # isolation matrix above and the "fresh empty scope" scope-resolution
        # test — the original RED row here expected [] which contradicted
        # both; the cross-user isolation itself is covered by the agent-y row
        # and by test_user_id_is_never_taken_from_a_client_supplied_header).
        ({**BOB, AGENT_HEADER: "agent-x"}, ["user-bob/agent-x/task.md"]),
    ],
)
async def test_rest_list_files_isolated_by_caller_scope(
    api, storage_external, headers, expected
):
    storage_external.seed(
        {
            "user-alice/personal/notes.txt": b"alice personal",
            "user-alice/agent-x/plan.md": b"alice agent-x",
            "user-bob/agent-x/task.md": b"bob agent-x",
        }
    )

    response = await api.get(f"{REST}/list", headers=headers, params={"prefix": ""})

    assert response.status_code == 200
    assert [row["object_name"] for row in response.json()] == expected


async def test_rest_list_folders_returns_only_caller_scope_folders(
    api, storage_external
):
    storage_external.seed(
        {
            "user-alice/agent-x/docs/report.txt": b"a",
            "user-bob/agent-x/private/task.txt": b"b",
        }
    )

    response = await api.get(
        f"{REST}/folders", headers=WITH_AGENT_X, params={"prefix": ""}
    )

    assert response.status_code == 200
    assert response.json() == ["docs/"]


# ---------------------------------------------------------------------------
# read_file (MCP + REST) — REQ-003/REQ-005
# ---------------------------------------------------------------------------


async def test_mcp_read_file_composes_the_scoped_object_path(
    mcp_call, storage_external
):
    storage_external.seed({"user-alice/agent-x/docs/report.txt": b"# scoped report"})

    result = await mcp_call(
        FILES_MOUNT, "read_file", {"file_path": "docs/report.txt"}, headers=WITH_AGENT_X
    )

    assert result.is_error is False
    reads = storage_external.calls_for("get_object")
    assert reads and reads[-1]["object_path"] == "user-alice/agent-x/docs/report.txt"


async def test_rest_read_file_missing_object_inside_scope_answers_404(
    api, storage_external
):
    storage_external.seed(
        {
            "user-alice/agent-x/plan.md": b"present",
            "user-bob/agent-x/task.md": b"other user",
        }
    )

    response = await api.post(
        f"{REST}/read", headers=WITH_AGENT_X, json={"file_path": "docs/absent.txt"}
    )

    assert response.status_code == 404


async def test_rest_read_file_of_another_users_object_is_not_reachable(
    api, storage_external
):
    """The historical vulnerability: guessing an absolute prefix must fail."""
    storage_external.seed(
        {
            "user-alice/agent-x/plan.md": b"alice secret",
            "user-bob/agent-x/task.md": b"bob secret",
        }
    )

    await api.post(
        f"{REST}/read",
        headers={**BOB, AGENT_HEADER: "agent-x"},
        json={"file_path": "user-alice/agent-x/plan.md"},
    )

    # user_id is composed in front of the client path; the guessed key is
    # unreachable from user-bob's scope.
    reads = storage_external.calls_for("get_object")
    assert reads and reads[-1]["object_path"] == (
        "user-bob/agent-x/user-alice/agent-x/plan.md"
    )


# ---------------------------------------------------------------------------
# create_folder / delete_file / delete_folder (REST) — REQ-003
# ---------------------------------------------------------------------------


async def test_rest_create_folder_is_scoped(api, storage_external):
    response = await api.post(
        f"{REST}/folders", headers=WITH_AGENT_X, json={"prefix": "docs/archive"}
    )

    assert response.status_code == 201, response.text
    assert response.json()["prefix"] == "user-alice/agent-x/docs/archive/"


async def test_rest_create_folder_rejects_traversal_before_scope_composition(
    api, storage_external
):
    """Security review FINDING-1: create_folder routes its relative prefix
    through the SAME validation as every other file op (traversal/absolute
    rejected 422) BEFORE scope composition — no residual-``..`` marker key
    may ever be composed."""
    for bad_prefix in ("../escape", "/etc/passwd"):
        response = await api.post(
            f"{REST}/folders", headers=WITH_AGENT_X, json={"prefix": bad_prefix}
        )

        assert response.status_code == 422, (bad_prefix, response.text)
    # No marker object may have been stored for either rejected prefix.
    assert storage_external.calls_for("put_object") == []


async def test_rest_delete_file_composes_object_and_working_dir(
    api, storage_external, rag_boundaries
):
    response = await api.request(
        "DELETE",
        REST,
        headers=WITH_AGENT_X,
        params={"object_name": "docs/report.txt", "working_dir": "p1"},
    )

    assert response.status_code == 200
    removed = storage_external.calls_for("remove_object")
    assert (
        removed and removed[-1]["object_path"] == "user-alice/agent-x/docs/report.txt"
    )
    cascades = rag_boundaries.calls_for("delete_documents")
    assert cascades and cascades[-1] == (
        "user-alice/agent-x/p1/",
        "user-alice/agent-x/docs/report.txt",
    )


async def test_rest_delete_file_with_explicitly_empty_working_dir_cascades_to_scope_root(
    api, storage_external, rag_boundaries
):
    """"""
    # Code-review iteration-5 cross-repo contract: an EXPLICITLY-EMPTY
    # working_dir ("") composes the caller's SCOPE ROOT namespace — the
    # identical normalization the query/index paths apply to an empty
    # working_dir — and cascades there.
    response = await api.request(
        "DELETE",
        REST,
        headers=WITH_AGENT_X,
        params={"object_name": "docs/report.txt", "working_dir": ""},
    )

    assert response.status_code == 200
    removed = storage_external.calls_for("remove_object")
    assert (
        removed and removed[-1]["object_path"] == "user-alice/agent-x/docs/report.txt"
    )
    cascades = rag_boundaries.calls_for("delete_documents")
    assert cascades and cascades[-1] == (
        "user-alice/agent-x/",
        "user-alice/agent-x/docs/report.txt",
    )


async def test_rest_delete_file_without_working_dir_keeps_the_scoped_default_fallback(
    api, storage_external, rag_boundaries
):
    """"""
    # Code-review iteration-5: an ABSENT working_dir keeps the legacy
    # ``default-composable`` fallback, composed under the caller's scope
    # (old REST clients — the deployment order requirement).
    response = await api.request(
        "DELETE",
        REST,
        headers=WITH_AGENT_X,
        params={"object_name": "docs/report.txt"},
    )

    assert response.status_code == 200
    cascades = rag_boundaries.calls_for("delete_documents")
    assert cascades and cascades[-1] == (
        "user-alice/agent-x/default-composable/",
        "user-alice/agent-x/docs/report.txt",
    )


async def test_rest_delete_folder_composes_the_prefix(
    api, storage_external, rag_boundaries
):
    response = await api.request(
        "DELETE",
        f"{REST}/folders",
        headers=WITH_AGENT_X,
        params={"prefix": "p1/docs"},
    )

    assert response.status_code == 200
    removals = storage_external.calls_for("remove_prefix")
    assert removals and removals[-1]["prefix"] == "user-alice/agent-x/p1/docs/"
    cascades = rag_boundaries.calls_for("delete_by_prefix")
    assert cascades and cascades[-1] == (
        "user-alice/agent-x/p1/docs/",
        "user-alice/agent-x/p1/docs/",
    )


# ---------------------------------------------------------------------------
# MCP file-tool surface stays intact (guards against accidental renaming)
# ---------------------------------------------------------------------------


async def test_files_mount_still_exposes_the_public_file_tools(mcp_list):
    tools = await mcp_list(FILES_MOUNT)
    names = [tool.name for tool in tools]
    for required in ("list_files", "list_folders", "read_file"):
        assert required in names


# ---------------------------------------------------------------------------
# REQ-004: the temporary chat-attachment policy is unchanged
# ---------------------------------------------------------------------------


async def test_temporary_upload_composes_the_caller_scope_before_chat_tmp(
    mcp_call, storage_external
):
    """chat_tmp rides the CALLER'S scope (write == read of the agent-facing
    file tools, scope-rag-memories REQ-003): the scoped composition applies
    in front of the RELATIVE chat_tmp prefix."""
    result = await mcp_call(
        FILES_MOUNT,
        "temporary_upload_file",
        {
            "file_name": "attachment.txt",
            "content_base64": "aGVsbG8=",
            "content_type": "text/plain",
            "prefix": "chat_tmp/user-alice/session-1",
        },
        headers={**{"Authorization": "Bearer jwt-alice"}, AGENT_HEADER: "agent-x"},
    )

    assert result.is_error is False, error_text(result)
    stored = storage_external.calls_for("put_object")
    assert (
        stored
        and stored[-1]["object_path"]
        == "user-alice/agent-x/chat_tmp/user-alice/session-1/attachment.txt"
    )


async def test_temporary_upload_still_rejects_another_users_chat_tmp_prefix(
    mcp_call, storage_external
):
    result = await mcp_call(
        FILES_MOUNT,
        "temporary_upload_file",
        {
            "file_name": "attachment.txt",
            "content_base64": "aGVsbG8=",
            "content_type": "text/plain",
            "prefix": "chat_tmp/user-bob/evil",
        },
        headers={**{"Authorization": "Bearer jwt-alice"}, AGENT_HEADER: "agent-x"},
    )

    assert result.is_error is True
    assert storage_external.calls_for("put_object") == []


async def test_temporary_delete_composes_the_caller_scope_like_the_write_side(
    mcp_call, storage_external
):
    scoped_name = "user-alice/agent-x/chat_tmp/user-alice/session-1/attachment.txt"
    storage_external.seed({scoped_name: b"hello"})

    result = await mcp_call(
        FILES_MOUNT,
        "temporary_delete_file",
        {"object_path": "chat_tmp/user-alice/session-1/attachment.txt"},
        headers={**{"Authorization": "Bearer jwt-alice"}, AGENT_HEADER: "agent-x"},
    )

    assert result.is_error is False, error_text(result)
    removals = storage_external.calls_for("remove_object")
    assert removals and removals[-1]["object_path"] == scoped_name


async def test_temporary_upload_of_disallowed_type_still_rejected(
    mcp_call, storage_external
):
    result = await mcp_call(
        FILES_MOUNT,
        "temporary_upload_file",
        {
            "file_name": "malware.exe",
            "content_base64": "aGVsbG8=",
            "content_type": "application/x-msdownload",
            "prefix": "chat_tmp/user-alice/x",
        },
    )

    assert result.is_error is True
    assert storage_external.calls_for("put_object") == []


# ---------------------------------------------------------------------------
# streamable-HTTP request-body cap (413 on oversized MCP POSTs) — REQ-092
# ---------------------------------------------------------------------------


async def test_mcp_transport_accepts_mcp_post_bodies_above_the_sdk_4mib_default(
    mcp_call, storage_external
):
    """The transport cap lifts from the 4 MiB SDK default to the FILES budget:
    an 8 MiB payload (base64-inflated past the SDK line) reaches the tool and
    is stored instead of answering 413 Content Too Large."""
    payload = b"X" * (8 * 1024 * 1024)

    result = await mcp_call(
        FILES_MOUNT,
        "temporary_upload_file",
        {
            "file_name": "big.pdf",
            "content_base64": base64.b64encode(payload).decode(),
            "content_type": "application/pdf",
            "prefix": "chat_tmp/user-alice/session-1",
        },
        headers=WITH_AGENT_X,
    )

    assert result.is_error is False, error_text(result)
    stored = storage_external.calls_for("put_object")
    assert stored and stored[-1]["size"] == 8 * 1024 * 1024
    assert stored[-1]["object_path"].endswith("chat_tmp/user-alice/session-1/big.pdf")


def test_apply_mcp_max_request_body_size_patches_the_fastmcp_lifespan_symbol():
    """The fastmcp lifespan builds its session manager through the patched
    module symbol: after ``apply_mcp_max_request_body_size`` every mount
    instantiates the LargeBody subclass (and defaulting reads the FILES
    budget at call time)."""
    import fastmcp.server.http as fastmcp_http
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from application.api.mcp_transport_security import (
        LargeBodyStreamableHTTPSessionManager,
        mcp_max_request_body_size,
        apply_mcp_max_request_body_size,
    )

    assert mcp_max_request_body_size() > 4 * 1024 * 1024
    previous = fastmcp_http.StreamableHTTPSessionManager
    try:
        apply_mcp_max_request_body_size()
        assert fastmcp_http.StreamableHTTPSessionManager is (
            LargeBodyStreamableHTTPSessionManager
        )
        assert issubclass(
            LargeBodyStreamableHTTPSessionManager, StreamableHTTPSessionManager
        )
    finally:
        fastmcp_http.StreamableHTTPSessionManager = previous
