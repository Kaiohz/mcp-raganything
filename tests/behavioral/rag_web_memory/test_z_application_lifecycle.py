"""Application-lifecycle supplement after exercising both public MCP tools.

This file intentionally sorts last within the feature: it closes process-wide
resources and verifies that a repeated shutdown is harmless.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_shutdown_closes_crawler_search_executor_and_memory_client_idempotently(
    running_app,
    mcp_call,
    web_external,
    memory_external,
    monkeypatch,
):
    web_external.search_results = [
        {"title": "page", "href": "https://example.com/page", "body": "page"}
    ]
    web_result = await mcp_call(
        "/web/mcp", "web_search", {"query": "lifecycle", "explore_top_n": 1}
    )
    memory_result = await mcp_call(
        "/memories/mcp",
        "save_memory",
        {"name": "lifecycle", "content": "resource check"},
    )
    assert web_result.is_error is False
    assert memory_result.is_error is False
    assert web_external.crawler_count == 1
    assert memory_external.clients

    shutdowns = []
    original_shutdown = ThreadPoolExecutor.shutdown

    def track_shutdown(executor, *args, **kwargs):
        shutdowns.append(executor)
        return original_shutdown(executor, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", track_shutdown)
    initial_crawler_closes = web_external.crawler_close_count

    # The resource lifespan is the non-HTTP lifecycle entry point specified for
    # these process-wide resources; both public tools were exercised above.
    from main import db_lifespan

    async with db_lifespan(running_app):
        pass
    async with db_lifespan(running_app):
        pass

    assert web_external.crawler_close_count == initial_crawler_closes + 1
    assert shutdowns
    assert all(client.is_closed for client in memory_external.clients)
