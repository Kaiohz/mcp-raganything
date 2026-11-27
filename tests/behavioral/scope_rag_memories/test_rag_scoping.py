"""Behavioral contract for classical-RAG scoping (REQ-006, REQ-007, REQ-008) — TDD RED.

The effective classical-RAG namespace is ``{user_id}/{agent_slug}/{working_dir}``:

* REQ-006 — for indexing (``POST /api/v1/classical/file/index``,
  ``POST /api/v1/classical/folder/index``) and querying (MCP
  ``classical_query``, ``POST /api/v1/classical/query``), where ``working_dir``
  is the client-supplied, project-level relative value.
* REQ-007 — the ``user_id`` metadata tag at index time and the ``user_id``
  metadata filter at query time are retained (defense-in-depth).
* REQ-008 — one pgvector table per composed working_dir (existing
  ``classical_rag_<hash>`` naming); a space tracks every working_dir it created.

Isolation is behavioral: the namespace-aware vector double emulates the
per-table semantics of pgvector (chunks live in exactly one composed
namespace), so "agent-y querying working_dir `p1` gets ZERO chunks from
agent-x's corpus" is asserted through the REAL MCP tool / REST route responses.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

REST_QUERY = "/api/v1/classical/query"
REST_FILE_INDEX = "/api/v1/classical/file/index"
REST_FOLDER_INDEX = "/api/v1/classical/folder/index"

AGENT_HEADER = "X-Composable-Agent"
ALICE = {"Authorization": "Bearer jwt-alice"}
WITH_AGENT_X = {**ALICE, AGENT_HEADER: "agent-x"}
WITH_AGENT_Y = {**ALICE, AGENT_HEADER: "agent-y"}

COMPOSED_AGENT_X_P1 = "user-alice/agent-x/p1/"
COMPOSED_AGENT_Y_P1 = "user-alice/agent-y/p1/"
COMPOSED_PERSONAL_ROOT = "user-alice/personal/"
COMPOSED_AGENT_X_ROOT = "user-alice/agent-x/"


def error_text(result) -> str:
    return "\n".join(getattr(item, "text", "") for item in (result.content or []))


def rag_chunks(result) -> list[dict]:
    """Extract the chunk rows from an MCP classical_query result."""
    assert result.is_error is False, error_text(result)
    data = result.data
    payload = data if isinstance(data, dict) else vars(data)
    items = payload.get("rag_response") or []
    return [item if isinstance(item, dict) else dict(vars(item)) for item in items]


def similarity_searches(vector_double) -> list[tuple]:
    return vector_double.calls_for("similarity_search")


async def await_background_index(vector_double, timeout_seconds: float = 10.0) -> None:
    """Wait for an accepted (202) background indexing task to reach the boundary."""
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if vector_double.calls_for("add_documents"):
            return
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# REQ-006 + REQ-007: MCP classical_query composes namespace and keeps the filter
# ---------------------------------------------------------------------------


async def test_mcp_query_composes_working_dir_with_scope_and_user_filter(
    mcp_call, rag_boundaries
):
    result = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": "p1", "query": "what is scoped?", "enable_llm_judge": False},
        headers=WITH_AGENT_X,
    )

    assert result.is_error is False
    searches = similarity_searches(rag_boundaries)
    assert searches, "classical_query never reached the vector store"
    for working_dir, _query, _top_k, _threshold, metadata_filter in searches:
        assert working_dir == COMPOSED_AGENT_X_P1
        assert metadata_filter == {"user_id": "user-alice"}


async def test_mcp_query_without_header_queries_the_personal_root_namespace(
    mcp_call, rag_boundaries
):
    result = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": "", "query": "personal docs", "enable_llm_judge": False},
        headers=ALICE,
    )

    assert result.is_error is False
    searches = similarity_searches(rag_boundaries)
    assert searches and {payload[0] for payload in searches} == {COMPOSED_PERSONAL_ROOT}


# ---------------------------------------------------------------------------
# Isolation matrix: zero chunks across agents (composed namespaces)
# ---------------------------------------------------------------------------


async def test_agent_y_querying_p1_gets_zero_chunks_from_agent_x_corpus(
    mcp_call, rag_boundaries
):
    vector = rag_boundaries
    vector.seed_namespace(
        COMPOSED_AGENT_X_P1,
        [
            (
                "agent-x internal roadmap",
                "docs/roadmap.txt",
                {"user_id": "user-alice"},
            )
        ],
    )
    # Pre-feature corpora live directly under the raw working_dir; in the
    # scoped world they are unreachable from any agent scope (post-migration).
    vector.seed_namespace(
        "p1/",
        [
            (
                "agent-x internal roadmap (pre-feature)",
                "docs/roadmap.txt",
                {"user_id": "user-alice"},
            )
        ],
    )

    cross = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": "p1", "query": "roadmap", "enable_llm_judge": False},
        headers=WITH_AGENT_Y,
    )
    own = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": "p1", "query": "roadmap", "enable_llm_judge": False},
        headers=WITH_AGENT_X,
    )

    cross_chunks = rag_chunks(cross)
    own_chunks = rag_chunks(own)
    assert cross_chunks == [], (
        f"agent-y must see zero chunks from agent-x's corpus, got: {cross_chunks}"
    )
    assert [chunk["content"] for chunk in own_chunks] == ["agent-x internal roadmap"]


async def test_personal_scope_cannot_reach_an_agent_corpus(mcp_call, rag_boundaries):
    vector = rag_boundaries
    vector.seed_namespace(
        COMPOSED_AGENT_X_P1,
        [("agent-x only", "docs/a.txt", {"user_id": "user-alice"})],
    )
    # Pre-feature chunks live under the raw working_dir; the personal scope
    # must not reach them after scoping either.
    vector.seed_namespace(
        "p1/", [("pre-feature chunk", "docs/a.txt", {"user_id": "user-alice"})]
    )
    vector.seed_namespace(
        "user-alice/personal/p1/",
        [("personal p1 chunk", "docs/p.txt", {"user_id": "user-alice"})],
    )

    result = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": "p1", "query": "a", "enable_llm_judge": False},
        headers=ALICE,
    )

    chunks = rag_chunks(result)
    assert [chunk["content"] for chunk in chunks] == ["personal p1 chunk"]


async def test_rest_query_isolated_across_agents(api, rag_boundaries):
    vector = rag_boundaries
    vector.seed_namespace(
        COMPOSED_AGENT_X_P1,
        [("agent-x chunk", "docs/a.txt", {"user_id": "user-alice"})],
    )

    response = await api.post(
        REST_QUERY,
        json={"working_dir": "p1", "query": "anything", "enable_llm_judge": False},
        headers=WITH_AGENT_Y,
    )

    assert response.status_code == 200, response.text
    assert response.json()["chunks"] == []
    searches = similarity_searches(vector)
    assert searches and searches[-1][0] == COMPOSED_AGENT_Y_P1
    assert searches[-1][4] == {"user_id": "user-alice"}


# ---------------------------------------------------------------------------
# REQ-006/REQ-007: REST indexing composes the namespace
# ---------------------------------------------------------------------------


async def test_rest_file_index_composes_working_dir_and_keeps_user_tag(
    api, storage_external, rag_boundaries
):
    storage_external.seed(
        {
            "user-alice/agent-x/docs/report.txt": b"scoped indexed content for the project."
        }
    )

    response = await api.post(
        REST_FILE_INDEX,
        json={"file_name": "docs/report.txt", "working_dir": "p1"},
        headers=WITH_AGENT_X,
    )

    assert response.status_code == 202, response.text
    await await_background_index(rag_boundaries)

    reads = storage_external.calls_for("get_object")
    assert reads and reads[-1]["object_path"] == "user-alice/agent-x/docs/report.txt"
    tables = rag_boundaries.calls_for("ensure_table")
    assert tables and tables[-1][0] == COMPOSED_AGENT_X_P1
    additions = rag_boundaries.calls_for("add_documents")
    assert additions, "indexing never added documents"
    working_dir, documents = additions[-1]
    assert working_dir == COMPOSED_AGENT_X_P1
    for _content, _file_path, metadata in documents:
        assert metadata.get("user_id") == "user-alice"


async def test_rest_file_index_without_header_indexes_the_personal_composition(
    api, storage_external, rag_boundaries
):
    storage_external.seed(
        {"user-alice/personal/docs/guide.txt": b"personal guide content."}
    )

    response = await api.post(
        REST_FILE_INDEX,
        json={"file_name": "docs/guide.txt", "working_dir": ""},
        headers=ALICE,
    )

    assert response.status_code == 202
    await await_background_index(rag_boundaries)

    additions = rag_boundaries.calls_for("add_documents")
    assert additions and additions[-1][0] == COMPOSED_PERSONAL_ROOT


async def test_rest_folder_index_schedules_the_scoped_prefix(
    api, storage_external, rag_boundaries
):
    storage_external.seed(
        {
            "user-alice/agent-x/p1/readme.txt": b"project readme content.",
            "user-bob/agent-x/p1/leak.txt": b"must not be listed",
        }
    )

    response = await api.post(
        REST_FOLDER_INDEX,
        json={"working_dir": "p1"},
        headers=WITH_AGENT_X,
    )

    assert response.status_code == 202
    await await_background_index(rag_boundaries)

    listings = storage_external.calls_for("list_objects")
    assert listings and listings[-1]["prefix"] == COMPOSED_AGENT_X_P1


# ---------------------------------------------------------------------------
# REQ-008 edge: namespace normalization of empty / slashed working_dirs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supplied", "composed"),
    [
        ("", COMPOSED_AGENT_X_ROOT),
        ("/", COMPOSED_AGENT_X_ROOT),
        ("p1", COMPOSED_AGENT_X_P1),
        ("p1/", COMPOSED_AGENT_X_P1),
    ],
)
async def test_working_dir_edge_forms_collapse_identically_per_space(
    mcp_call, rag_boundaries, supplied, composed
):
    result = await mcp_call(
        "/classical/mcp",
        "classical_query",
        {"working_dir": supplied, "query": "q", "enable_llm_judge": False},
        headers=WITH_AGENT_X,
    )

    assert result.is_error is False
    searches = similarity_searches(rag_boundaries)
    assert searches and searches[-1][0] == composed


async def test_same_supplied_working_dir_under_different_agents_targets_different_tables(
    mcp_call, rag_boundaries
):
    arguments = {"working_dir": "p1", "query": "q", "enable_llm_judge": False}

    await mcp_call("/classical/mcp", "classical_query", arguments, headers=WITH_AGENT_X)
    await mcp_call("/classical/mcp", "classical_query", arguments, headers=WITH_AGENT_Y)
    await mcp_call("/classical/mcp", "classical_query", arguments, headers=ALICE)

    working_dirs = {payload[0] for payload in similarity_searches(rag_boundaries)}
    assert working_dirs == {
        COMPOSED_AGENT_X_P1,
        COMPOSED_AGENT_Y_P1,
        "user-alice/personal/p1/",
    }
