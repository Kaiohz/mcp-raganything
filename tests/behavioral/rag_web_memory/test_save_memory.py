"""Behavioral contract for the public ``save_memory`` MCP tool."""

import httpx
import pytest
from mcp.shared.exceptions import McpError

pytestmark = pytest.mark.asyncio(loop_scope="session")

MEMORY_MOUNT = "/memories/mcp"
UPSTREAM_PATH = "/api/v1/store/files/memories/preferences.md"


def response_body(result) -> dict:
    assert result.is_error is False, result.content
    data = result.data
    if isinstance(data, dict):
        return data
    # FastMCP validates the tool's advertised output schema into a generated
    # dataclass type; serialise it into the advertised response contract
    # (fields match REQ-011 field-for-field).
    assert getattr(data, "__dataclass_fields__", None) is not None
    return dict(vars(data))


async def save(mcp_call, name="preferences", content="remember this", **kwargs):
    arguments = {"name": name, "content": content, **kwargs}
    credential = arguments.pop("credential", "jwt-alice")
    headers = arguments.pop("headers", None)
    return await mcp_call(
        MEMORY_MOUNT,
        "save_memory",
        arguments,
        credential=credential,
        headers=headers,
    )


async def test_memories_mount_exposes_only_the_approved_tool(mcp_list):
    tools = await mcp_list(MEMORY_MOUNT)

    assert [tool.name for tool in tools] == ["save_memory"]
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {"name", "content", "if_exists"}
    assert set(schema["required"]) == {"name", "content"}
    assert schema["properties"]["if_exists"]["default"] == "fail"
    assert set(schema["properties"]["if_exists"]["enum"]) == {
        "fail",
        "append",
        "replace",
    }
    for forbidden in ("user_id", "email", "tenant", "account", "namespace", "path"):
        assert forbidden not in schema["properties"]
    # The documented response contract must be advertised, not a loose
    # additionalProperties object (REQ-011).
    output_properties = tools[0].outputSchema["properties"]
    assert set(output_properties) == {"path", "action", "chars"}
    assert output_properties["action"]["enum"] == [
        "created",
        "appended",
        "replaced",
    ]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer invalid-jwt"},
        {"X-API-Key": "invalid-key"},
    ],
)
async def test_memory_discovery_rejects_missing_or_invalid_credentials(
    mcp_list, headers
):
    with pytest.raises(McpError, match="Invalid or missing credentials"):
        await mcp_list(MEMORY_MOUNT, headers=headers)


async def test_memory_invocation_rejects_missing_credentials_without_upstream_call(
    mcp_call, memory_external
):
    result = await save(mcp_call, credential=None)

    assert result.is_error is True
    assert memory_external.requests == []


async def test_jwt_creates_memory_with_normalized_path_exact_chars_and_forwarding(
    mcp_call, memory_external
):
    content = "I prefer tea 🍵\nAlways use metric units."

    result = await save(mcp_call, name="preferences.md", content=content)

    assert response_body(result) == {
        "path": "/memories/preferences.md",
        "action": "created",
        "chars": len(content),
    }
    assert memory_external.stored("jwt-alice", "preferences") == content
    assert [request.method for request in memory_external.requests] == ["GET", "PUT"]
    get_request, put_request = memory_external.requests
    assert get_request.url.path == UPSTREAM_PATH
    assert put_request.url.path == UPSTREAM_PATH
    for request in memory_external.requests:
        assert request.headers.get("authorization") == "Bearer jwt-alice"
        assert request.headers.get("x-api-key") is None
    import json

    assert json.loads(put_request.content) == {"content": content}
    assert any(
        kwargs.get("timeout") == 15 for kwargs in memory_external.client_kwargs_history
    )


async def test_api_key_creates_memory_and_forwards_only_x_api_key(
    mcp_call, memory_external
):
    result = await mcp_call(
        MEMORY_MOUNT,
        "save_memory",
        {"name": "preferences", "content": "API key user"},
        credential="key-alice",
    )

    assert response_body(result)["action"] == "created"
    assert memory_external.stored("key-alice", "preferences") == "API key user"
    assert all(
        request.headers.get("x-api-key") == "key-alice"
        and request.headers.get("authorization") is None
        for request in memory_external.requests
    )


async def test_valid_jwt_takes_precedence_and_never_forwards_both_schemes(
    mcp_call, memory_external
):
    result = await save(
        mcp_call,
        headers={"Authorization": "Bearer jwt-alice", "X-API-Key": "key-bob"},
    )

    assert result.is_error is False
    assert memory_external.stored("jwt-alice", "preferences") == "remember this"
    assert memory_external.stored("key-bob", "preferences") is None
    assert all(
        request.headers.get("authorization") == "Bearer jwt-alice"
        and request.headers.get("x-api-key") is None
        for request in memory_external.requests
    )


async def test_name_variants_share_one_md_suffix_and_default_fail_does_not_overwrite(
    mcp_call, memory_external
):
    first = await save(mcp_call, name="preferences", content="original")

    assert response_body(first)["path"] == "/memories/preferences.md"
    memory_external.requests.clear()
    collision = await save(mcp_call, name="preferences.md", content="replacement")

    assert collision.is_error is True
    assert [request.method for request in memory_external.requests] == ["GET"]
    assert memory_external.stored("jwt-alice", "preferences") == "original"


async def test_existing_memory_fail_mode_issues_no_put(mcp_call, memory_external):
    memory_external.seed("jwt-alice", "preferences", "do not overwrite")

    result = await save(mcp_call, content="replacement")

    assert result.is_error is True
    assert [request.method for request in memory_external.requests] == ["GET"]
    assert memory_external.stored("jwt-alice", "preferences") == "do not overwrite"


async def test_append_to_absent_memory_creates_it(mcp_call, memory_external):
    result = await save(mcp_call, content="first", if_exists="append")

    assert response_body(result) == {
        "path": "/memories/preferences.md",
        "action": "created",
        "chars": 5,
    }
    assert memory_external.stored("jwt-alice", "preferences") == "first"


async def test_replace_to_absent_memory_creates_it(mcp_call, memory_external):
    result = await save(mcp_call, content="first", if_exists="replace")

    assert response_body(result)["action"] == "created"
    assert memory_external.stored("jwt-alice", "preferences") == "first"


@pytest.mark.parametrize(
    ("existing", "supplied", "expected"),
    [
        ("", "new", "new"),
        ("existing", "new", "existing\n\nnew"),
        ("existing\n", "new", "existing\n\nnew"),
        ("existing\n\n", "new", "existing\n\nnew"),
        ("existing\n\n\n", "new", "existing\n\n\nnew"),
        ("🙂", "新しい🍵", "🙂\n\n新しい🍵"),
    ],
)
async def test_append_preserves_existing_content_with_exact_markdown_spacing(
    mcp_call, memory_external, existing, supplied, expected
):
    memory_external.seed("jwt-alice", "preferences", existing)

    result = await save(mcp_call, content=supplied, if_exists="append")

    assert response_body(result) == {
        "path": "/memories/preferences.md",
        "action": "appended",
        "chars": len(expected),
    }
    assert memory_external.stored("jwt-alice", "preferences") == expected
    assert len(memory_external.requests) == 2
    import json

    assert json.loads(memory_external.requests[1].content) == {"content": expected}


async def test_replace_existing_stores_only_new_content(mcp_call, memory_external):
    memory_external.seed("jwt-alice", "preferences", "old content")

    result = await save(mcp_call, content="new 🍵", if_exists="replace")

    assert response_body(result) == {
        "path": "/memories/preferences.md",
        "action": "replaced",
        "chars": len("new 🍵"),
    }
    assert memory_external.stored("jwt-alice", "preferences") == "new 🍵"


async def test_non_empty_whitespace_content_is_stored_verbatim(
    mcp_call, memory_external
):
    content = " \n\t "

    result = await save(mcp_call, content=content)

    assert response_body(result)["chars"] == len(content)
    assert memory_external.stored("jwt-alice", "preferences") == content


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        ".md",
        "../secret",
        "..\\secret",
        "nested/name",
        "nested\\name",
        "/etc/passwd",
        "C:\\secret",
        "%2e%2e%2fsecret",
        "preferences%00.md",
        "bad\nname",
        "bad\x00name",
        "/memories/preferences",
        "agents/alice/memories/preferences",
    ],
)
async def test_invalid_memory_names_fail_before_any_upstream_request(
    mcp_call, memory_external, name
):
    result = await save(mcp_call, name=name)

    assert result.is_error is True
    assert memory_external.requests == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"content": "missing name"},
        {"name": "preferences"},
        {"name": "preferences", "content": ""},
        {"name": "preferences", "content": "x", "if_exists": "merge"},
        {"name": "preferences", "content": "x", "user_id": "user-bob"},
        {"name": "preferences", "content": "x", "namespace": "filesystem"},
        {"name": "preferences", "content": "x", "path": "/etc/passwd"},
    ],
)
async def test_invalid_memory_arguments_and_caller_identity_are_rejected(
    mcp_call, memory_external, arguments
):
    result = await mcp_call(MEMORY_MOUNT, "save_memory", arguments)

    assert result.is_error is True
    assert memory_external.requests == []


async def test_legacy_master_key_fails_closed_without_identity_or_upstream_call(
    mcp_call, memory_external, monkeypatch
):
    import main

    monkeypatch.setattr(main.mcp_middleware, "_auth_service", None)
    monkeypatch.setattr(main.mcp_middleware, "_master_key", "legacy-master")

    result = await save(mcp_call, credential="legacy-master")

    assert result.is_error is True
    assert memory_external.requests == []


async def test_authentication_disabled_mode_also_fails_closed(
    mcp_call, memory_external, monkeypatch
):
    import main

    monkeypatch.setattr(main.mcp_middleware, "_auth_service", None)
    monkeypatch.setattr(main.mcp_middleware, "_master_key", "")

    result = await save(mcp_call, credential=None)

    assert result.is_error is True
    assert memory_external.requests == []


async def test_two_users_can_use_the_same_name_without_cross_user_content(
    mcp_call, memory_external
):
    alice = await save(mcp_call, content="Alice private", credential="jwt-alice")
    bob = await save(mcp_call, content="Bob private", credential="jwt-bob")
    assert response_body(alice)["action"] == "created"
    assert response_body(bob)["action"] == "created"

    appended = await save(
        mcp_call, content="Bob preference", if_exists="append", credential="jwt-bob"
    )
    replaced = await save(
        mcp_call,
        content="Alice replacement",
        if_exists="replace",
        credential="jwt-alice",
    )

    assert response_body(appended)["action"] == "appended"
    assert response_body(replaced)["action"] == "replaced"
    assert memory_external.stored("jwt-bob", "preferences") == (
        "Bob private\n\nBob preference"
    )
    assert memory_external.stored("jwt-alice", "preferences") == "Alice replacement"


@pytest.mark.parametrize("status", [401, 403, 500, 503])
async def test_get_failures_other_than_404_never_mean_absent_and_never_put(
    mcp_call, memory_external, error_text, status
):
    memory_external.respond_next("GET", status, "raw-upstream-secret Bearer jwt-alice")

    result = await save(mcp_call, content="must not persist")

    assert result.is_error is True
    assert [request.method for request in memory_external.requests] == ["GET"]
    assert memory_external.stored("jwt-alice", "preferences") is None
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "jwt-alice" not in message
    assert "Traceback" not in message
    if status in (401, 403):
        assert any(
            word in message.lower() for word in ("auth", "credential", "permission")
        )


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
async def test_put_failures_never_return_success_or_claim_durability(
    mcp_call, memory_external, error_text, status
):
    memory_external.respond_next(
        "PUT", status, "raw-upstream-secret X-API-Key: static-service-key"
    )

    result = await save(mcp_call, content="must not persist")

    assert result.is_error is True
    assert [request.method for request in memory_external.requests] == ["GET", "PUT"]
    assert memory_external.stored("jwt-alice", "preferences") is None
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "static-service-key" not in message
    assert "Traceback" not in message


@pytest.mark.parametrize("method", ["GET", "PUT"])
async def test_upstream_timeouts_are_safe_tool_errors_not_success(
    mcp_call, memory_external, error_text, method
):
    memory_external.fail_next(
        method,
        httpx.TimeoutException("timeout at 10.0.0.5 with jwt-alice"),
    )

    result = await save(mcp_call, content="not durable")

    assert result.is_error is True
    message = error_text(result).lower()
    assert "time" in message
    assert "jwt-alice" not in message
    assert "10.0.0.5" not in message
    assert memory_external.stored("jwt-alice", "preferences") is None


@pytest.mark.parametrize("method", ["GET", "PUT"])
async def test_malformed_upstream_success_responses_are_rejected(
    mcp_call, memory_external, error_text, method
):
    memory_external.respond_next(method, 200, "not-json raw-upstream-secret")

    result = await save(mcp_call, content="content")

    assert result.is_error is True
    message = error_text(result)
    assert "raw-upstream-secret" not in message
    assert "response" in message.lower()
