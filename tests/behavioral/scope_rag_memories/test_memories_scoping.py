"""Behavioral contract for scoped memories (REQ-012, REQ-013, REQ-014) — TDD RED.

* REQ-012 — agent memories live at ``/memories/{agent_slug}/{name}.md`` in the
  composable-agents store; personal memories keep the flat
  ``/memories/{name}.md`` shape; exactly ONE controlled nesting segment (the
  agent slug, validated with the REQ-002 charset) is allowed — deeper nesting
  stays rejected.
* REQ-013 — the Memories MCP server exposes ``list_memories()`` (names +
  metadata), ``get_memory(name)`` (content) and ``delete_memory(name)`` next to
  ``save_memory``, all resolved per REQ-001 and forwarded with the caller's OWN
  credential.
* REQ-014 — ``save_memory`` honors the scope (the adapter composes the store
  path); upstream failures surface as sanitized ``MemoryUpstreamError``-mapped
  tool errors with no credential leakage.

All tests go through the real mounted ``/memories/mcp`` app; the composable-
agents Store File API is the session ``memory_external`` HTTP-boundary double
(extended with listing/DELETE handlers by the feature conftest).
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

MEMORY_MOUNT = "/memories/mcp"
AGENT_HEADER = "X-Composable-Agent"
AUTH_ALICE = {"Authorization": "Bearer jwt-alice"}
WITH_AGENT_X = {**AUTH_ALICE, AGENT_HEADER: "agent-x"}
WITH_AGENT_Y = {**AUTH_ALICE, AGENT_HEADER: "agent-y"}


def error_text(result) -> str:
    return "\n".join(getattr(item, "text", "") for item in (result.content or []))


def tool_data(result) -> dict:
    assert result.is_error is False, error_text(result)
    data = result.data
    if isinstance(data, dict):
        return data
    if hasattr(data, "__dataclass_fields__"):
        return dict(vars(data))
    return data


async def save(mcp_call, name="prefs", content="remember this", **kwargs):
    arguments = {"name": name, "content": content, **kwargs}
    headers = arguments.pop("headers", None)
    return await mcp_call(MEMORY_MOUNT, "save_memory", arguments, headers=headers)


# ---------------------------------------------------------------------------
# The dedicated Memories server exposes the four scoped tools (REQ-013)
# ---------------------------------------------------------------------------


async def test_memories_mount_exposes_save_list_get_delete(mcp_list):
    tools = await mcp_list(MEMORY_MOUNT)

    names = sorted(tool.name for tool in tools)
    assert names == ["delete_memory", "get_memory", "list_memories", "save_memory"]
    schemas = {tool.name: tool.inputSchema for tool in tools}
    # No scope/user parameters anywhere (REQ-002: never client-chosen).
    for name, schema in schemas.items():
        properties = set(schema.get("properties", {}))
        for forbidden in (
            "user_id",
            "agent",
            "agent_slug",
            "scope",
            "namespace",
            "path",
        ):
            assert forbidden not in properties, (name, forbidden)


# ---------------------------------------------------------------------------
# save_memory honors the scope (REQ-012 / REQ-014)
# ---------------------------------------------------------------------------


async def test_save_with_agent_header_composes_the_agent_namespace_path(
    mcp_call, memory_external
):
    result = await save(
        mcp_call, name="prefs", content="agent prefs", headers=WITH_AGENT_X
    )

    body = tool_data(result)
    assert body == {
        "path": "/memories/agent-x/prefs.md",
        "action": "created",
        "chars": len("agent prefs"),
    }
    puts = [request for request in memory_external.requests if request.method == "PUT"]
    assert puts and puts[0].url.path == "/api/v1/store/files/memories/agent-x/prefs.md"


async def test_save_without_header_keeps_the_flat_personal_path(
    mcp_call, memory_external
):
    result = await save(mcp_call, name="prefs", content="personal prefs")

    body = tool_data(result)
    assert body["path"] == "/memories/prefs.md"
    puts = [request for request in memory_external.requests if request.method == "PUT"]
    assert puts and puts[0].url.path == "/api/v1/store/files/memories/prefs.md"


async def test_same_memory_name_coexists_in_personal_and_agent_scopes(
    mcp_call, memory_external
):
    personal = await save(mcp_call, name="prefs", content="personal taste")
    agent = await save(
        mcp_call, name="prefs", content="agent taste", headers=WITH_AGENT_X
    )

    assert tool_data(personal)["action"] == "created"
    assert tool_data(agent)["action"] == "created"
    assert memory_external.stored("jwt-alice", "prefs") == "personal taste"
    assert memory_external.files[
        ("Bearer jwt-alice", "/memories/agent-x/prefs.md")
    ] == ("agent taste")


async def test_agent_scoped_save_appends_within_the_agent_namespace_only(
    mcp_call, memory_external, memory_store_ext
):
    memory_store_ext.seed(
        "jwt-alice",
        "agent-x/prefs",
        "existing agent note",
    )

    result = await save(
        mcp_call, name="prefs", content="more", if_exists="append", headers=WITH_AGENT_X
    )

    assert tool_data(result)["action"] == "appended"
    assert (
        memory_store_ext.files[("Bearer jwt-alice", "/memories/agent-x/prefs.md")]
        == "existing agent note\n\nmore"
    )
    assert memory_store_ext.stored("jwt-alice", "prefs") is None


@pytest.mark.parametrize(
    "name",
    ["nested/deep/name", "a/b", "../escape", "agent-x/prefs.md/extra"],
)
async def test_deeper_nesting_is_still_rejected_even_inside_an_agent_scope(
    mcp_call, memory_external, name
):
    result = await save(mcp_call, name=name, headers=WITH_AGENT_X)

    assert result.is_error is True
    assert memory_external.requests == []


@pytest.mark.parametrize(
    "agent_value",
    ["Bad_Name!", "Agent-X", "-agent", "agent/x", "%41"],
)
async def test_invalid_agent_header_rejected_before_any_upstream_request(
    mcp_call, memory_external, agent_value
):
    result = await save(
        mcp_call,
        name="prefs",
        content="never stored",
        headers={**AUTH_ALICE, AGENT_HEADER: agent_value},
    )

    assert result.is_error is True
    assert memory_external.requests == []


# ---------------------------------------------------------------------------
# list_memories / get_memory / delete_memory (REQ-013)
# ---------------------------------------------------------------------------


async def test_list_memories_returns_only_the_agent_scoped_names(
    mcp_call, memory_external, memory_store_ext
):
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent prefs")
    memory_store_ext.seed("jwt-alice", "agent-x/todo", "agent todo")
    memory_store_ext.seed("jwt-alice", "prefs", "personal prefs")
    memory_store_ext.seed("jwt-bob", "agent-x/foreign", "someone else")

    result = await mcp_call(MEMORY_MOUNT, "list_memories", {}, headers=WITH_AGENT_X)

    rows = tool_data(result)
    listing = rows if isinstance(rows, list) else rows.get("memories", [])
    names = [entry["name"] if isinstance(entry, dict) else entry for entry in listing]
    assert sorted(names) == ["prefs", "todo"]


async def test_list_memories_on_an_empty_scope_answers_an_empty_listing(
    mcp_call, memory_store_ext
):
    """An empty scope lists ZERO memories — the real listing route
    (``GET /api/v1/store/files?prefix=<scope>``, QA BUG-002) answers
    ``200 + []`` for a scope that holds no memories (no 404 is involved)."""
    # No seed at all: the double's prefix query filters everything out → [].
    result = await mcp_call(MEMORY_MOUNT, "list_memories", {}, headers=WITH_AGENT_X)

    assert result.is_error is False, error_text(result)
    listing_request = [
        request
        for request in memory_store_ext.requests
        if request.url.path == "/api/v1/store/files"
    ]
    assert listing_request, "the listing must hit the prefix-query route"
    assert listing_request[-1].url.params.get("prefix") == "/memories/agent-x/", (
        "the listing must filter on the caller's own scope prefix"
    )
    rows = tool_data(result)
    listing = rows if isinstance(rows, list) else rows.get("memories", [])
    assert listing == []


async def test_list_memories_surfaces_an_upstream_listing_anomaly_as_sanitized_error(
    mcp_call, memory_store_ext
):
    """QA BUG-002 regression pin (inverted): a 404 on the LISTING route is an
    upstream anomaly now — swallowing it as an "empty scope" was exactly how
    the misrouted file-read call hid the bug. It surfaces as a sanitized
    ToolError (never a success, never a leak)."""
    memory_store_ext.respond_next("GET", 404, "raw-upstream-secret Bearer jwt-alice")

    result = await mcp_call(MEMORY_MOUNT, "list_memories", {}, headers=WITH_AGENT_X)

    assert result.is_error is True
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "jwt-alice" not in message


async def test_list_memories_on_the_personal_scope_lists_flat_memories(
    mcp_call, memory_external, memory_store_ext
):
    memory_store_ext.seed("jwt-alice", "prefs", "personal prefs")
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent prefs")

    result = await mcp_call(MEMORY_MOUNT, "list_memories", {}, headers=None)

    rows = tool_data(result)
    listing = rows if isinstance(rows, list) else rows.get("memories", [])
    names = [entry["name"] if isinstance(entry, dict) else entry for entry in listing]
    assert names == ["prefs"]


async def test_get_memory_returns_the_agent_scoped_content(mcp_call, memory_store_ext):
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent secret 🍵")

    result = await mcp_call(
        MEMORY_MOUNT, "get_memory", {"name": "prefs"}, headers=WITH_AGENT_X
    )

    data = tool_data(result)
    if isinstance(data, dict):
        assert data.get("name") == "prefs"
        assert str(data.get("content")) == "agent secret 🍵"
    else:
        assert data == "agent secret 🍵"
    gets = [
        request
        for request in memory_store_ext.requests
        if request.method == "GET" and request.url.path.endswith(".md")
    ]
    assert gets and gets[-1].url.path == "/api/v1/store/files/memories/agent-x/prefs.md"


async def test_get_memory_is_isolated_across_scopes(mcp_call, memory_store_ext):
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent secret")

    personal = await mcp_call(
        MEMORY_MOUNT, "get_memory", {"name": "prefs"}, headers=None
    )
    other_agent = await mcp_call(
        MEMORY_MOUNT, "get_memory", {"name": "prefs"}, headers=WITH_AGENT_Y
    )

    assert personal.is_error is True
    assert other_agent.is_error is True
    for result in (personal, other_agent):
        message = error_text(result)
        assert "agent secret" not in message
    assert "jwt-alice" not in (error_text(personal) + error_text(other_agent))


async def test_delete_memory_removes_only_the_scoped_file(mcp_call, memory_store_ext):
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent prefs")
    memory_store_ext.seed("jwt-alice", "prefs", "personal prefs")

    result = await mcp_call(
        MEMORY_MOUNT, "delete_memory", {"name": "prefs"}, headers=WITH_AGENT_X
    )

    assert result.is_error is False, error_text(result)
    deletes = [
        request for request in memory_store_ext.requests if request.method == "DELETE"
    ]
    assert (
        deletes
        and deletes[0].url.path == "/api/v1/store/files/memories/agent-x/prefs.md"
    )
    assert (
        memory_store_ext.files.get(("Bearer jwt-alice", "/memories/agent-x/prefs.md"))
        is None
    )
    assert memory_store_ext.stored("jwt-alice", "prefs") == "personal prefs"


async def test_delete_memory_of_an_absent_scoped_file_is_a_sanitized_error(
    mcp_call, memory_store_ext
):
    """QA BUG-004: the store's DELETE answers 204 for an absent file exactly
    like for a real deletion — the adapter verifies the file EXISTS before
    deleting, so an absent memory surfaces the documented not-found error
    and never an ambiguous "success" (and no DELETE is even issued)."""
    result = await mcp_call(
        MEMORY_MOUNT, "delete_memory", {"name": "ghost"}, headers=WITH_AGENT_X
    )

    assert result.is_error is True
    assert "Memory not found" in error_text(result)
    assert "jwt-alice" not in error_text(result)
    assert [
        request for request in memory_store_ext.requests if request.method == "DELETE"
    ] == [], "the pre-check must catch the absent file before any DELETE is issued"
    prechecks = [
        request for request in memory_store_ext.requests if request.method == "GET"
    ]
    assert (
        prechecks
        and prechecks[-1].url.path == "/api/v1/store/files/memories/agent-x/ghost.md"
    )


# ---------------------------------------------------------------------------
# Credential forwarding and sanitized upstream failures (REQ-013/REQ-014)
# ---------------------------------------------------------------------------


async def test_new_tools_forward_only_the_callers_own_credential(
    mcp_call, memory_store_ext
):
    memory_store_ext.seed("jwt-alice", "agent-x/prefs", "agent prefs")

    listed = await mcp_call(MEMORY_MOUNT, "list_memories", {}, headers=WITH_AGENT_X)
    fetched = await mcp_call(
        MEMORY_MOUNT, "get_memory", {"name": "prefs"}, headers=WITH_AGENT_X
    )
    removed = await mcp_call(
        MEMORY_MOUNT, "delete_memory", {"name": "prefs"}, headers=WITH_AGENT_X
    )

    assert listed.is_error is False
    assert fetched.is_error is False
    assert removed.is_error is False
    for request in memory_store_ext.requests:
        assert request.headers.get("authorization") == "Bearer jwt-alice"
        assert request.headers.get("x-api-key") is None


async def test_api_key_callers_forward_the_api_key_header(mcp_call, memory_store_ext):
    memory_store_ext.seed("key-alice", "agent-x/prefs", "key user prefs")

    result = await mcp_call(
        MEMORY_MOUNT,
        "get_memory",
        {"name": "prefs"},
        headers={"X-API-Key": "key-alice", AGENT_HEADER: "agent-x"},
    )

    assert result.is_error is False, error_text(result)
    for request in memory_store_ext.requests:
        assert request.headers.get("x-api-key") == "key-alice"
        assert request.headers.get("authorization") is None


@pytest.mark.parametrize(
    ("tool", "method", "status"),
    [
        ("list_memories", "GET", 500),
        ("get_memory", "GET", 503),
        ("delete_memory", "DELETE", 503),
    ],
)
async def test_upstream_failures_surface_as_sanitized_tool_errors(
    mcp_call, memory_store_ext, tool, method, status
):
    if tool == "delete_memory":
        # QA BUG-004 fix: the deletion pre-checks the file via a GET, so a
        # scripted DELETE-503 lands on the pre-check request — seed the file
        # so the pre-check succeeds upstream and reaches the scripted failure.
        memory_store_ext.seed("jwt-alice", "agent-x/prefs", "deletable")
        memory_store_ext.respond_next("GET", status, "")
    memory_store_ext.respond_next(
        method, status, "raw-upstream-secret Bearer jwt-alice"
    )

    result = await mcp_call(MEMORY_MOUNT, tool, {"name": "prefs"}, headers=WITH_AGENT_X)

    assert result.is_error is True
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "jwt-alice" not in message
    assert "Traceback" not in message


async def test_save_memory_upstream_failure_still_sanitized_with_agent_scope(
    mcp_call, memory_external
):
    memory_external.respond_next("PUT", 503, "raw-upstream-secret Bearer jwt-alice")

    result = await save(
        mcp_call, name="prefs", content="must not persist", headers=WITH_AGENT_X
    )

    assert result.is_error is True
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "jwt-alice" not in message


async def test_unauthenticated_call_never_reaches_the_upstream(
    mcp_call, memory_external
):
    # The HTTP guard rejects before any MCP method runs; the client seam
    # surfaces that as the credential error contract (see global conftest).
    result = await mcp_call(
        MEMORY_MOUNT,
        "list_memories",
        {},
        headers={AGENT_HEADER: "agent-x"},
        credential=None,
    )

    assert result.is_error is True
    assert "credentials" in error_text(result).lower()
    assert memory_external.requests == []
