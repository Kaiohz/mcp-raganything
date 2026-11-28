"""Behavioral contract for scope resolution (REQ-001, REQ-002) — TDD RED.

The caller scope ``(user_id, agent_slug)`` must be resolved on EVERY
authenticated MCP tool call and REST route of mcp-raganything:

* REQ-001: the user id comes from the authenticated identity (the RLS
  contextvar established by ``verify_credentials`` / the MCP guard), the agent
  slug from the ``X-Composable-Agent`` header, defaulting to ``personal`` when
  the header is absent.
* REQ-002: a value that does not match ``^[a-z0-9][a-z0-9-]{0,99}$`` is
  rejected with a validation error (422 on REST, ``ToolError`` on MCP) BEFORE
  any storage operation, and the user id is never derived from a
  client-supplied header or parameter (cross-user access impossible by
  construction).

All tests go through the real surfaces: REST routes over the real app
(httpx ``ASGITransport``) and MCP tools over the real mounted apps.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

AGENT_HEADER = "X-Composable-Agent"
CLASSICAL_MOUNT = "/classical/mcp"
FILES_MOUNT = "/files/mcp"
REST_LIST = "/api/v1/files/list"


def bearer(credential: str = "jwt-alice") -> dict[str, str]:
    """Credential headers as the real clients present them."""
    if credential.startswith("jwt-"):
        return {"Authorization": f"Bearer {credential}"}
    return {"X-API-Key": credential}


def error_text(result) -> str:
    """Extract the text content of an MCP error result."""
    return "\n".join(getattr(item, "text", "") for item in (result.content or []))


def tool_rows(result) -> list[dict]:
    """Serialise a FastMCP structured result into plain dict rows."""
    assert result.is_error is False, error_text(result)
    data = result.data
    if data is None:
        return []
    if not isinstance(data, list):
        data = [data]
    return [item if isinstance(item, dict) else dict(vars(item)) for item in data]


#: Header values that violate ^[a-z0-9][a-z0-9-]{0,99}$ (REQ-002).
INVALID_AGENT_HEADERS = [
    "Bad_Name!",
    "Agent-X",
    "-agent",
    "agent/x",
    "%41",
    "agent_x",
    "9" + "9" * 100,  # 101 characters (one over the boundary)
    "agent..x",
]

#: A valid 100-character value (the regex's length boundary).
VALID_100_CHARS = ("a0-" * 33) + "a"  # exactly 100 characters, all in [a-z0-9-]


# ---------------------------------------------------------------------------
# REQ-001: no header → scope (user_id, "personal")
# ---------------------------------------------------------------------------


async def test_dual_auth_wired_requests_are_always_scoped_f2_verification(
    api, storage_external
):
    """Security review FINDING-2 (verification pin): while dual-auth IS wired,
    an authenticated REST file operation ALWAYS composes its scope prefix —
    the unscoped legacy fallback (the deployment-controlled escape hatch
    documented on ``security.verify_credentials``) is unreachable for an
    authenticated request, because the dual-auth path always sets the
    ``current_user_id`` contextvar the scoping branches read."""
    response = await api.get(REST_LIST, headers=bearer("jwt-alice"))

    assert response.status_code == 200
    listing = storage_external.calls_for("list_files_metadata")
    assert listing and listing[-1]["prefix"] == "user-alice/personal/", (
        "an authenticated request must be scoped (never the unscoped legacy path)"
    )


async def test_rest_list_without_agent_header_returns_only_personal_scope(
    api, storage_external
):
    """No ``X-Composable-Agent`` header → prefix ``{user_id}/personal/``."""
    # Arrange — objects across two users and two scopes of user A.
    storage_external.seed(
        {
            "user-alice/personal/notes.txt": b"alice personal",
            "user-alice/agent-x/plan.md": b"alice agent",
            "user-bob/personal/own.txt": b"bob personal",
            "user-bob/agent-x/task.md": b"bob agent",
        }
    )

    # Act
    response = await api.get(REST_LIST, headers=bearer("jwt-alice"))

    # Assert — only the caller's personal scope is visible.
    assert response.status_code == 200
    object_names = [row["object_name"] for row in response.json()]
    assert object_names == ["user-alice/personal/notes.txt"]


async def test_mcp_list_files_without_agent_header_returns_only_personal_scope(
    mcp_call, storage_external
):
    """MCP variant of the personal fallback (REQ-001)."""
    storage_external.seed(
        {
            "user-alice/personal/notes.txt": b"alice personal",
            "user-alice/agent-x/plan.md": b"alice agent",
            "user-bob/personal/own.txt": b"bob personal",
        }
    )

    result = await mcp_call(FILES_MOUNT, "list_files", {"prefix": ""})

    rows = tool_rows(result)
    assert [row["object_name"] for row in rows] == ["user-alice/personal/notes.txt"]


# ---------------------------------------------------------------------------
# REQ-002: invalid header → validation error, no storage operation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", INVALID_AGENT_HEADERS)
async def test_rest_rejects_invalid_agent_header_with_422_without_storage_call(
    api, storage_external, value
):
    storage_external.seed({"user-alice/personal/notes.txt": b"content"})

    response = await api.get(
        REST_LIST, headers={**bearer("jwt-alice"), AGENT_HEADER: value}
    )

    assert response.status_code == 422
    # No storage operation may occur on an invalid scope (REQ-002).
    assert storage_external.calls == []


async def test_mcp_classical_rejects_invalid_agent_header_before_query(
    mcp_call, rag_boundaries
):
    result = await mcp_call(
        CLASSICAL_MOUNT,
        "classical_query",
        {"working_dir": "p1", "query": "question", "enable_llm_judge": False},
        headers={**bearer("jwt-alice"), AGENT_HEADER: "Agent X"},
    )

    assert result.is_error is True
    assert rag_boundaries.calls_for("similarity_search") == []


async def test_rest_classical_query_rejects_invalid_agent_header_with_422(
    api, rag_boundaries
):
    response = await api.post(
        "/api/v1/classical/query",
        json={"working_dir": "p1", "query": "question", "enable_llm_judge": False},
        headers={**bearer("jwt-alice"), AGENT_HEADER: "Bad_Name!"},
    )

    assert response.status_code == 422
    assert rag_boundaries.calls_for("similarity_search") == []


async def test_100_char_agent_header_value_is_accepted(api, storage_external):
    """Boundary of the regex: exactly 100 characters is a VALID slug."""
    storage_external.seed({})

    response = await api.get(
        REST_LIST,
        headers={**bearer("jwt-alice"), AGENT_HEADER: VALID_100_CHARS},
    )

    assert response.status_code == 200
    listing = storage_external.calls_for("list_files_metadata")
    assert listing and listing[-1]["prefix"] == f"user-alice/{VALID_100_CHARS}/"


# ---------------------------------------------------------------------------
# REQ-001/REQ-002: the user id always comes from the credential
# ---------------------------------------------------------------------------


async def test_user_id_is_never_taken_from_a_client_supplied_header(
    api, storage_external
):
    """A decoy identity header must never grant another user's scope."""
    storage_external.seed(
        {
            "user-alice/personal/notes.txt": b"alice personal",
            "user-alice/agent-x/plan.md": b"alice agent-x",
            "user-bob/personal/own.txt": b"bob personal",
            "user-bob/agent-x/task.md": b"bob agent",
        }
    )

    response = await api.get(
        REST_LIST,
        headers={
            **bearer("jwt-alice"),
            AGENT_HEADER: "agent-x",
            "X-User-Id": "user-bob",
            "X-Composable-User": "user-bob",
        },
    )

    assert response.status_code == 200
    object_names = [row["object_name"] for row in response.json()]
    assert object_names == ["user-alice/agent-x/plan.md"]


async def test_caller_scope_is_composed_from_the_authenticated_identity(
    api, storage_external
):
    """JWT resolves to user-alice; the agent header only adds the slug."""
    storage_external.seed({})

    response = await api.get(
        REST_LIST,
        headers={**bearer("jwt-alice"), AGENT_HEADER: "agent-x"},
        params={"prefix": "docs/"},
    )

    assert response.status_code == 200
    listing = storage_external.calls_for("list_files_metadata")
    assert listing and listing[-1]["prefix"] == "user-alice/agent-x/docs/"


async def test_api_key_credential_resolves_the_same_scoping_rules(
    api, storage_external
):
    """A per-user API key is an equally valid identity source (REQ-001)."""
    storage_external.seed({"user-bob/agent-x/task.md": b"bob agent"})

    response = await api.get(
        REST_LIST,
        headers={**bearer("key-bob"), AGENT_HEADER: "agent-x"},
    )

    assert response.status_code == 200
    object_names = [row["object_name"] for row in response.json()]
    assert object_names == ["user-bob/agent-x/task.md"]
    listing = storage_external.calls_for("list_files_metadata")
    assert listing and listing[-1]["prefix"] == "user-bob/agent-x/"


# ---------------------------------------------------------------------------
# Cross-user access is impossible by construction (REQ-001 acceptance)
# ---------------------------------------------------------------------------


async def test_bob_with_agent_x_header_gets_a_fresh_empty_scope(api, storage_external):
    """User B sending user A's agent slug sees only B's own fresh scope."""
    storage_external.seed(
        {
            "user-alice/agent-x/secret-plan.md": b"alice secret",
            "user-bob/agent-x/own.txt": b"bob own",
        }
    )

    response = await api.get(
        REST_LIST, headers={**bearer("jwt-bob"), AGENT_HEADER: "agent-x"}
    )

    assert response.status_code == 200
    object_names = [row["object_name"] for row in response.json()]
    assert object_names == ["user-bob/agent-x/own.txt"]


async def test_mcp_prefix_is_relative_to_the_caller_scope_not_the_bucket(
    mcp_call, storage_external
):
    """A client-supplied prefix naming another user's tree stays inside the scope."""
    storage_external.seed(
        {
            "user-alice/agent-x/here.txt": b"mine",
            "user-bob/personal/other-user-tree.txt": b"not mine",
        }
    )

    result = await mcp_call(
        FILES_MOUNT,
        "list_files",
        {"prefix": "user-bob/personal/"},
        headers={**bearer("jwt-alice"), AGENT_HEADER: "agent-x"},
    )

    rows = tool_rows(result)
    # The requested prefix is relative to user-alice/agent-x/ — nothing from
    # user-bob's tree leaks through the real response.
    assert [row["object_name"] for row in rows] == []


# ---------------------------------------------------------------------------
# Tool contracts never expose the scope to the model (REQ-002)
# ---------------------------------------------------------------------------


async def test_files_tools_schema_never_accepts_user_or_agent_parameters(mcp_list):
    tools = await mcp_list(FILES_MOUNT)
    for tool in tools:
        properties = tool.inputSchema.get("properties", {})
        for forbidden in ("user_id", "agent", "agent_slug", "scope", "namespace"):
            assert forbidden not in properties, (tool.name, forbidden)


async def test_scoped_namespace_is_stable_across_repeated_calls(
    mcp_call, rag_boundaries
):
    """Same identity + header → identical composed namespace (determinism)."""
    arguments = {"working_dir": "p1", "query": "q", "enable_llm_judge": False}
    headers = {**bearer("jwt-alice"), AGENT_HEADER: "agent-x"}

    await mcp_call(CLASSICAL_MOUNT, "classical_query", arguments, headers=headers)
    await mcp_call(CLASSICAL_MOUNT, "classical_query", arguments, headers=headers)

    searches = rag_boundaries.calls_for("similarity_search")
    assert len(searches) >= 2
    working_dirs = {payload[0] for payload in searches}
    assert working_dirs == {"user-alice/agent-x/p1/"}
