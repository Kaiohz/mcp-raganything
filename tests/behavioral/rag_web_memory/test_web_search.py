"""Behavioral contract for the public ``web_search`` MCP tool."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import McpError

from infrastructure.web_search.ddgs_adapter import DDGSSearchAdapter

pytestmark = pytest.mark.asyncio(loop_scope="session")

WEB_MOUNT = "/web/mcp"


def provider_results(count: int = 3) -> list[dict[str, str]]:
    return [
        {
            "title": f"Result {index}",
            "href": f"https://example.com/{index}",
            "body": f"Snippet {index}",
        }
        for index in range(1, count + 1)
    ]


def response_body(result) -> dict:
    assert result.is_error is False, result.content
    data = result.data
    if isinstance(data, dict):
        return data
    # FastMCP validates the tool's advertised output schema into a generated
    # dataclass type; serialise it recursively into the advertised response
    # contract (fields match REQ-006 field-for-field).
    assert getattr(data, "__dataclass_fields__", None) is not None
    return {key: _response_value(value) for key, value in vars(data).items()}


def _response_value(value):
    if (
        isinstance(value, list)
        and value
        and getattr(value[0], "__dataclass_fields__", None) is not None
    ):
        return [_response_value(item) for item in value]
    if getattr(value, "__dataclass_fields__", None) is not None:
        return {key: _response_value(item) for key, item in vars(value).items()}
    return value


async def wait_until(predicate, timeout: float = 2.0) -> None:
    """Poll ``predicate`` until truthy, within ``timeout`` seconds.

    The ceiling is deliberately set above CI-runner scheduling jitter (0.12s
    failed twice on ubuntu runners while passing locally in < 1s). The spec
    assertions themselves (ceiling ≤ 2, fresh dispatchers, shared limiter)
    remain unchanged; the poll exits as soon as the predicate is satisfied.
    """
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_web_mount_exposes_only_the_approved_tool_without_starting_chromium(
    mcp_list, web_external
):
    tools = await mcp_list(WEB_MOUNT)

    assert [tool.name for tool in tools] == ["web_search"]
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {"query", "max_results", "explore_top_n"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["max_results"]["default"] == 5
    assert schema["properties"]["explore_top_n"]["default"] == 3
    assert "max_chars_per_page" not in schema["properties"]
    # The documented response contract must be advertised, not a loose
    # additionalProperties object: the crawl status enum is part of REQ-006.
    output_properties = tools[0].outputSchema["properties"]
    assert set(output_properties) == {"query", "results"}
    crawl_properties = output_properties["results"]["items"]["properties"]["crawl"][
        "properties"
    ]
    assert set(crawl_properties) == {
        "status",
        "final_url",
        "markdown",
        "truncated",
        "error",
    }
    assert crawl_properties["status"]["enum"] == [
        "not_explored",
        "success",
        "failed",
    ]
    assert web_external.crawler_count == 0
    assert web_external.crawler_start_count == 0


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer invalid-jwt"},
        {"X-API-Key": "invalid-key"},
    ],
)
async def test_web_discovery_rejects_missing_or_invalid_credentials(mcp_list, headers):
    with pytest.raises(McpError, match="Invalid or missing credentials"):
        await mcp_list(WEB_MOUNT, headers=headers)


async def test_web_invocation_rejects_missing_credentials_without_calling_ddgs(
    mcp_call, web_external
):
    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "private"}, credential=None
    )

    assert result.is_error is True
    assert web_external.search_calls == []


async def test_concurrent_first_crawls_share_one_lazy_crawler(mcp_call, web_external):
    web_external.search_results = provider_results(1)

    first, second = await asyncio.gather(
        mcp_call(WEB_MOUNT, "web_search", {"query": "first", "explore_top_n": 1}),
        mcp_call(WEB_MOUNT, "web_search", {"query": "second", "explore_top_n": 1}),
    )

    assert response_body(first)["query"] == "first"
    assert response_body(second)["query"] == "second"
    assert web_external.crawler_count == 1
    assert web_external.crawler_start_count == 1
    assert len(web_external.crawl_batches) == 2


async def test_search_uses_duckduckgo_maps_results_and_streams_one_batch(
    mcp_call, web_external
):
    web_external.search_results = provider_results(4)
    for index in range(1, 5):
        web_external.markdown_by_url[f"https://example.com/{index}"] = (
            f"# Complete page {index}\n"
        )

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "  RAG news  "})

    body = response_body(result)
    assert body["query"] == "RAG news"
    assert web_external.search_calls[0]["query"] == "RAG news"
    assert web_external.search_calls[0]["kwargs"] == {
        "backend": "duckduckgo",
        "max_results": 5,
    }
    assert [item["title"] for item in body["results"]] == [
        "Result 1",
        "Result 2",
        "Result 3",
        "Result 4",
    ]
    assert [item["url"] for item in body["results"]] == [
        f"https://example.com/{index}" for index in range(1, 5)
    ]
    assert [item["snippet"] for item in body["results"]] == [
        "Snippet 1",
        "Snippet 2",
        "Snippet 3",
        "Snippet 4",
    ]
    assert len(web_external.crawl_batches) == 1
    batch = web_external.crawl_batches[0]
    assert batch.urls == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert batch.config.stream is True
    assert batch.config.cache_mode.name == "BYPASS"
    assert batch.config.check_robots_txt is True
    assert batch.config.page_timeout == 20_000
    assert batch.config.session_id is None
    assert batch.config.deep_crawl_strategy is None
    assert body["results"][3]["crawl"] == {
        "status": "not_explored",
        "final_url": None,
        "markdown": None,
        "truncated": False,
        "error": None,
    }


async def test_custom_limits_bound_search_and_safe_exploration_count(
    mcp_call, web_external
):
    web_external.search_results = provider_results(3) + [
        {"title": "No URL", "body": "ignored"},
        {"title": "Blank URL", "href": " ", "body": "ignored"},
    ]

    result = await mcp_call(
        WEB_MOUNT,
        "web_search",
        {"query": "bounded", "max_results": 2, "explore_top_n": 5},
    )

    body = response_body(result)
    assert web_external.search_calls[0]["kwargs"]["max_results"] == 2
    assert len(body["results"]) == 3
    assert web_external.crawl_batches[0].urls == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]


async def test_search_only_never_creates_or_invokes_the_crawler(mcp_call, web_external):
    web_external.search_results = provider_results(2)
    initial_crawlers = web_external.crawler_count

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "metadata", "explore_top_n": 0}
    )

    body = response_body(result)
    assert all(item["crawl"]["status"] == "not_explored" for item in body["results"])
    assert web_external.crawl_batches == []
    assert web_external.crawler_count == initial_crawlers


async def test_empty_search_returns_empty_response_without_crawler(
    mcp_call, web_external
):
    initial_crawlers = web_external.crawler_count

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "no results"})

    assert response_body(result) == {"query": "no results", "results": []}
    assert web_external.crawl_batches == []
    assert web_external.crawler_count == initial_crawlers


async def test_backend_fallback_uses_first_non_empty_backend(
    mcp_call, web_external, monkeypatch
):
    """The ordered backend chain rides out a single provider's challenge: a
    cleanly-empty first backend is skipped and the next one supplies results."""
    monkeypatch.setattr(
        "dependencies.web_search_adapter",
        DDGSSearchAdapter(
            timeout_seconds=5,
            max_concurrency=2,
            executor_workers=2,
            backends=["google", "duckduckgo"],
        ),
    )
    web_external.backends_results = {"google": []}
    web_external.search_results = provider_results(2)

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "fallback", "explore_top_n": 0}
    )

    body = response_body(result)
    assert result.is_error is False
    assert [call["kwargs"]["backend"] for call in web_external.search_calls] == [
        "google",
        "duckduckgo",
    ]
    assert [item["url"] for item in body["results"]] == [
        "https://example.com/1",
        "https://example.com/2",
    ]


async def test_backend_fallback_continues_after_a_provider_error(
    mcp_call, web_external, monkeypatch
):
    """A hard backend failure is logged and skipped, never surfaced while a
    later backend still supplies entries."""
    monkeypatch.setattr(
        "dependencies.web_search_adapter",
        DDGSSearchAdapter(
            timeout_seconds=5,
            max_concurrency=2,
            executor_workers=2,
            backends=["google", "duckduckgo"],
        ),
    )
    web_external.backends_results = {"google": RuntimeError("connect boom")}
    web_external.search_results = provider_results(1)

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "resilience", "explore_top_n": 0}
    )

    body = response_body(result)
    assert result.is_error is False
    assert [call["kwargs"]["backend"] for call in web_external.search_calls] == [
        "google",
        "duckduckgo",
    ]
    assert [item["title"] for item in body["results"]] == ["Result 1"]


async def test_all_backends_clean_empty_maps_to_empty_results(
    mcp_call, web_external, monkeypatch
):
    """When the whole chain answers cleanly-but-empty the genuine-empty
    contract (AC-22) still maps to an empty response, not an error."""
    monkeypatch.setattr(
        "dependencies.web_search_adapter",
        DDGSSearchAdapter(
            timeout_seconds=5,
            max_concurrency=2,
            executor_workers=2,
            backends=["google", "duckduckgo"],
        ),
    )
    web_external.backends_results = {"google": [], "duckduckgo": []}

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "genuinely empty", "explore_top_n": 0}
    )

    assert result.is_error is False
    assert response_body(result)["results"] == []
    assert [call["kwargs"]["backend"] for call in web_external.search_calls] == [
        "google",
        "duckduckgo",
    ]


async def test_all_backends_failing_surface_sanitized_provider_error(
    mcp_call, web_external, error_text, monkeypatch
):
    """No clean backend run at all: the LAST backend failure propagates and is
    sanitized like the single-backend provider failure it replaces."""
    monkeypatch.setattr(
        "dependencies.web_search_adapter",
        DDGSSearchAdapter(
            timeout_seconds=5,
            max_concurrency=2,
            executor_workers=2,
            backends=["google", "duckduckgo"],
        ),
    )
    web_external.backends_results = {
        "google": RuntimeError("private backend detail"),
        "duckduckgo": TimeoutError("socket 10.0.0.8"),
    }

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "outage", "explore_top_n": 0}
    )

    message = error_text(result)
    assert result.is_error is True
    assert len(message) < 500
    assert "private backend detail" not in message
    assert "10.0.0.8" not in message
    assert [call["kwargs"]["backend"] for call in web_external.search_calls] == [
        "google",
        "duckduckgo",
    ]


async def test_crawler_page_cap_truncates_markdown_and_flags_it(
    mcp_call, web_external, monkeypatch
):
    """A configured ``WEB_SEARCH_PAGE_MAX_CHARS`` cap cuts every successful
    page at the bound and flags the outcome ``truncated`` (context
    protection); 0 keeps the complete-Markdown contract (see the exact-markdown
    test above). The REAL prod crawl singleton is used — only its cap
    attribute is overridden, so the session-unique lazy-crawler contract
    (TestShutdown above) is preserved."""
    import dependencies

    adapter = dependencies.get_web_crawler_adapter()
    monkeypatch.setattr(adapter, "_max_page_chars", 50)

    web_external.search_results = provider_results(1)
    markdown = "x" * 200
    web_external.markdown_by_url["https://example.com/1"] = markdown

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "capped", "explore_top_n": 1}
    )

    crawl = response_body(result)["results"][0]["crawl"]
    assert crawl["status"] == "success"
    assert crawl["markdown"] == "x" * 50
    assert crawl["truncated"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "   \t\n"},
        {"query": "valid", "max_results": 0},
        {"query": "valid", "max_results": 11},
        {"query": "valid", "explore_top_n": -1},
        {"query": "valid", "explore_top_n": 6},
    ],
)
async def test_invalid_web_inputs_fail_before_any_external_call(
    mcp_call, web_external, arguments
):
    initial_crawlers = web_external.crawler_count

    result = await mcp_call(WEB_MOUNT, "web_search", arguments)

    assert result.is_error is True
    assert web_external.search_calls == []
    assert web_external.crawler_count == initial_crawlers


async def test_out_of_order_streaming_restores_provider_result_order(
    mcp_call, web_external
):
    web_external.search_results = provider_results(3)

    async def complete_out_of_order(urls):
        assert urls == [f"https://example.com/{i}" for i in range(1, 4)]
        for index in (3, 1, 2):
            url = f"https://example.com/{index}"
            web_external.markdown_by_url[url] = f"markdown-{index}"
            yield web_external.crawl_result(url)

    web_external.stream_handler = complete_out_of_order

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "ordered"})

    body = response_body(result)
    assert [item["url"] for item in body["results"]] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert [item["crawl"]["markdown"] for item in body["results"]] == [
        "markdown-1",
        "markdown-2",
        "markdown-3",
    ]


async def test_partial_page_failure_preserves_successes_and_search_metadata(
    mcp_call, web_external
):
    web_external.search_results = provider_results(3)
    web_external.markdown_by_url["https://example.com/1"] = "first full markdown"
    web_external.markdown_by_url["https://example.com/3"] = "third full markdown"
    web_external.crawl_failures["https://example.com/2"] = (
        "provider token=super-secret\nTraceback: private stack"
    )

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "partial"})

    assert result.is_error is False
    body = response_body(result)
    assert body["results"][0]["crawl"]["markdown"] == "first full markdown"
    assert body["results"][2]["crawl"]["markdown"] == "third full markdown"
    failed = body["results"][1]
    assert failed["title"] == "Result 2"
    assert failed["snippet"] == "Snippet 2"
    assert failed["crawl"]["status"] == "failed"
    assert failed["crawl"]["markdown"] is None
    assert failed["crawl"]["error"]
    assert "super-secret" not in failed["crawl"]["error"]
    assert "Traceback" not in failed["crawl"]["error"]


async def test_successful_empty_markdown_is_not_converted_to_failure(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    web_external.markdown_by_url["https://example.com/1"] = ""

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "empty page"})

    crawl = response_body(result)["results"][0]["crawl"]
    assert crawl["status"] == "success"
    assert crawl["markdown"] == ""
    assert crawl["error"] is None


async def test_long_unicode_markdown_is_exact_and_untruncated_without_a_cap(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    markdown = "# 完整研究 🐍\n" + ("é🙂漢字 markdownline\n" * 700) + "END-MARKER"
    web_external.markdown_by_url["https://example.com/1"] = markdown

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "full markdown"})

    body = response_body(result)
    item = body["results"][0]
    assert item["crawl"]["markdown"] == markdown
    assert item["crawl"]["markdown"].endswith("END-MARKER")
    assert item["crawl"]["truncated"] is False
    assert set(item) == {"title", "url", "snippet", "crawl"}
    assert set(item["crawl"]) == {
        "status",
        "final_url",
        "markdown",
        "truncated",
        "error",
    }
    assert "chars" not in str(body)


@pytest.mark.parametrize(
    "provider_error",
    [
        RuntimeError("DDGS secret=provider-secret\nTraceback: hidden"),
        TimeoutError("socket 10.0.0.8 secret=provider-secret"),
    ],
)
async def test_search_provider_errors_are_bounded_and_sanitized(
    mcp_call, web_external, error_text, provider_error
):
    web_external.search_error = provider_error

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "failure"})

    assert result.is_error is True
    message = error_text(result)
    assert len(message) < 500
    assert "provider-secret" not in message
    assert "10.0.0.8" not in message
    assert "Traceback" not in message


async def test_blocking_ddgs_uses_dedicated_executor_and_leaves_loop_responsive(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    web_external.search_gate = threading.Event()

    pending = asyncio.create_task(
        mcp_call(WEB_MOUNT, "web_search", {"query": "blocking", "explore_top_n": 0})
    )
    await wait_until(lambda: len(web_external.search_calls) == 1)
    await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.05)
    assert not pending.done()

    web_external.search_gate.set()
    result = await pending

    assert result.is_error is False
    thread_name = web_external.search_calls[0]["thread"]
    assert not thread_name.startswith("asyncio_")


async def test_outer_search_timeouts_do_not_start_work_above_process_capacity(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    web_external.search_gate = threading.Event()

    first_calls = [
        asyncio.create_task(
            mcp_call(
                WEB_MOUNT,
                "web_search",
                {"query": f"blocked-{index}", "explore_top_n": 0},
            )
        )
        for index in range(4)
    ]
    await wait_until(lambda: len(web_external.search_calls) == 4)
    timed_out = await asyncio.gather(*first_calls)
    assert all(result.is_error for result in timed_out)
    assert web_external.active_searches == 4
    # I-1: the timeout must surface the dedicated, sanitized timeout wording
    # (not the generic failure message and not a raw traceback).
    for result in timed_out:
        message = "\n".join(
            getattr(item, "text", "") for item in (result.content or [])
        )
        assert "did not complete in time" in message.lower()

    fifth = asyncio.create_task(
        mcp_call(WEB_MOUNT, "web_search", {"query": "fifth", "explore_top_n": 0})
    )
    await asyncio.sleep(0.03)
    assert len(web_external.search_calls) == 4

    web_external.search_gate.set()
    fifth_result = await fifth
    assert fifth_result.is_error is False
    assert web_external.max_active_searches <= 4


async def test_overall_crawl_timeout_preserves_completed_pages_only(
    mcp_call, web_external
):
    web_external.search_results = provider_results(3)
    web_external.markdown_by_url["https://example.com/1"] = "completed in full"

    async def one_page_then_stall(_urls):
        yield web_external.crawl_result("https://example.com/1")
        await asyncio.sleep(2)
        yield web_external.crawl_result("https://example.com/2")

    web_external.stream_handler = one_page_then_stall

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "deadline", "explore_top_n": 2}
    )

    body = response_body(result)
    assert body["results"][0]["crawl"]["markdown"] == "completed in full"
    assert body["results"][1]["crawl"]["status"] == "failed"
    assert body["results"][1]["crawl"]["markdown"] is None
    assert "time" in body["results"][1]["crawl"]["error"].lower()
    assert body["results"][2]["crawl"]["status"] == "not_explored"


async def test_overlapping_batches_use_fresh_dispatchers_shared_rate_limiter_and_ceiling(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    release = asyncio.Event()

    async def blocked_batch(urls):
        await release.wait()
        for url in urls:
            yield web_external.crawl_result(url)

    web_external.stream_handler = blocked_batch
    calls = [
        asyncio.create_task(
            mcp_call(WEB_MOUNT, "web_search", {"query": f"batch-{index}"})
        )
        for index in range(3)
    ]
    await wait_until(lambda: len(web_external.crawl_batches) == 2)
    await asyncio.sleep(0.02)

    assert len(web_external.crawl_batches) == 2
    assert web_external.active_crawl_batches == 2
    dispatchers = [batch.dispatcher for batch in web_external.crawl_batches]
    assert dispatchers[0] is not dispatchers[1]
    assert dispatchers[0].rate_limiter is dispatchers[1].rate_limiter
    assert dispatchers[0].max_session_permit == 2
    assert dispatchers[0].memory_threshold_percent == 80

    release.set()
    results = await asyncio.gather(*calls)
    assert [response_body(result)["query"] for result in results] == [
        "batch-0",
        "batch-1",
        "batch-2",
    ]
    assert web_external.max_active_crawl_batches <= 2


@pytest.mark.parametrize(
    ("url", "dns"),
    [
        ("file:///etc/passwd", {}),
        ("ftp://example.com/file", {}),
        ("https:///missing-host", {}),
        ("https://user:password@example.com/", {}),
        ("https://example.com:bad-port/", {}),
        ("https://127.0.0.1/private", {}),
        ("https://[::1]/private", {}),
        ("https://[::ffff:127.0.0.1]/private", {}),
        ("https://169.254.169.254/metadata", {}),
        ("https://localhost/private", {}),
        ("https://private.example/", {"private.example": ["10.0.0.4"]}),
        ("https://multicast.example/", {"multicast.example": ["224.0.0.1"]}),
        (
            "https://mixed.example/",
            {"mixed.example": ["8.8.8.8", "192.168.1.10"]},
        ),
    ],
)
async def test_unsafe_urls_and_non_global_dns_answers_are_never_crawled(
    mcp_call, web_external, url, dns
):
    web_external.search_results = [{"title": "unsafe", "href": url, "body": "x"}]
    web_external.dns_answers = dns
    initial_crawlers = web_external.crawler_count

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "ssrf"})

    crawl = response_body(result)["results"][0]["crawl"]
    assert crawl["status"] == "failed"
    assert crawl["markdown"] is None
    assert crawl["error"]
    assert web_external.crawl_batches == []
    assert web_external.crawler_count == initial_crawlers


async def test_unsafe_result_is_skipped_while_later_safe_result_is_explored(
    mcp_call, web_external
):
    unsafe = "https://private.example/secret"
    safe = "https://example.com/safe"
    web_external.search_results = [
        {"title": "unsafe", "href": unsafe, "body": "unsafe"},
        {"title": "safe", "href": safe, "body": "safe"},
    ]
    web_external.dns_answers = {"private.example": ["10.1.2.3"]}

    result = await mcp_call(
        WEB_MOUNT, "web_search", {"query": "mixed", "explore_top_n": 1}
    )

    body = response_body(result)
    assert body["results"][0]["crawl"]["status"] == "failed"
    assert body["results"][1]["crawl"]["status"] == "success"
    assert web_external.crawl_batches[0].urls == [safe]


async def test_public_to_private_redirect_cannot_return_successful_markdown(
    mcp_call, web_external
):
    original = "https://example.com/start"
    web_external.search_results = [
        {"title": "redirect", "href": original, "body": "redirect"}
    ]
    web_external.final_urls[original] = "http://127.0.0.1/admin"
    web_external.markdown_by_url[original] = "private response must not escape"

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "redirect"})

    crawl = response_body(result)["results"][0]["crawl"]
    assert crawl["status"] == "failed"
    assert crawl["markdown"] is None
    assert crawl["final_url"] in (None, original)
    assert crawl["error"]
    # The failure must be attributed to the redirect itself, proving the
    # FINAL destination (not the requested URL) was the one re-validated.
    assert "redirect" in crawl["error"].lower()


async def test_robots_denial_returns_no_partial_markdown_or_raw_html(
    mcp_call, web_external
):
    web_external.search_results = provider_results(1)
    web_external.crawl_failures["https://example.com/1"] = "robots.txt denied"
    web_external.markdown_by_url["https://example.com/1"] = "partial secret"

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "robots"})

    item = response_body(result)["results"][0]
    assert item["crawl"]["status"] == "failed"
    assert item["crawl"]["markdown"] is None
    assert "partial secret" not in str(item)
    assert "html" not in item


async def test_web_search_has_no_rag_indexing_side_effects(
    mcp_call, web_external, monkeypatch
):
    web_external.search_results = provider_results(1)
    embedding = AsyncMock(side_effect=AssertionError("web search embedded a page"))
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings.aembed_documents", embedding)
    monkeypatch.setattr(
        "asyncpg.Connection.execute",
        AsyncMock(side_effect=AssertionError("web search wrote to Postgres")),
    )
    monkeypatch.setattr(
        "asyncpg.Connection.executemany",
        AsyncMock(side_effect=AssertionError("web search wrote to Postgres")),
    )

    result = await mcp_call(WEB_MOUNT, "web_search", {"query": "no indexing"})

    assert result.is_error is False
    embedding.assert_not_awaited()
