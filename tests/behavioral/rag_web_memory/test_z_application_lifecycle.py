"""Application-lifecycle supplement after exercising the public MCP tools.

This file intentionally sorts last within the feature: it closes process-wide
resources and verifies that a repeated shutdown is harmless.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_shutdown_closes_crawler_and_search_executor_idempotently(
    running_app,
    mcp_call,
    web_external,
    monkeypatch,
):
    web_external.search_results = [
        {"title": "page", "href": "https://example.com/page", "body": "page"}
    ]
    web_result = await mcp_call(
        "/web/mcp", "web_search", {"query": "lifecycle", "explore_top_n": 1}
    )
    assert web_result.is_error is False
    assert web_external.crawler_count == 1

    shutdowns = []
    original_shutdown = ThreadPoolExecutor.shutdown

    def track_shutdown(executor, *args, **kwargs):
        shutdowns.append(executor)
        return original_shutdown(executor, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", track_shutdown)
    initial_crawler_closes = web_external.crawler_close_count

    # The resource lifespan is the non-HTTP lifecycle entry point specified for
    # these process-wide resources; the public tool was exercised above.
    from main import db_lifespan

    async with db_lifespan(running_app):
        pass
    async with db_lifespan(running_app):
        pass

    assert web_external.crawler_close_count == initial_crawler_closes + 1
    assert shutdowns
