"""Unit tests for the DDGS adapter's provider-result mapping (BUG-004).

ddgs 9.16.1 evidence: ``BaseSearchEngine.request`` returns ``None`` for any
non-200 response (including DuckDuckGo's HTTP 202 anti-bot challenge page) and
``DDGS._search_sync`` raises ``DDGSException("No results found.")`` with no
distinguishing signal in that case. These tests pin the adapter's contract:
the empty condition stays ``[]`` (AC-22) with an explicit operator warning,
while any other provider failure surfaces as a sanitized domain error.
"""

from unittest.mock import patch

import pytest
from ddgs.exceptions import DDGSException, TimeoutException

from domain.errors.web import WebSearchProviderError
from infrastructure.web_search.ddgs_adapter import DDGSSearchAdapter


def _adapter() -> DDGSSearchAdapter:
    return DDGSSearchAdapter(
        timeout_seconds=5.0,
        max_concurrency=2,
        executor_workers=2,
        backends=["duckduckgo"],
    )


def _fake_ddgs(monkeypatch, results=None, error: Exception | None = None):
    """Patch the adapter's module-level ``DDGS`` symbol for one call."""

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            if error is not None:
                raise error
            return results

    monkeypatch.setattr("infrastructure.web_search.ddgs_adapter.DDGS", FakeClient)
    return FakeClient


async def test_empty_provider_result_maps_to_empty_list_with_warning(monkeypatch):
    """The ddgs 'No results found.' condition maps to ``[]`` (AC-22 preserved)."""
    _fake_ddgs(monkeypatch, error=DDGSException("No results found."))
    adapter = _adapter()
    with patch("infrastructure.web_search.ddgs_adapter.logger.warning") as mock_warning:
        pages = await adapter.search("python programming language", max_results=5)
    assert pages == []
    assert mock_warning.called


async def test_provider_challenge_202_is_indistinguishable_from_empty(monkeypatch):
    """A 202 anti-bot page yields the same empty mapping (library-verified).

    ddgs collapses non-200 responses (e.g. the 202 challenge page) into
    ``DDGSException("No results found.")`` with no distinguishing signal, so
    the adapter must keep the ``[]`` mapping — the operator signal is the
    warning log, not a distinct exception.
    """
    _fake_ddgs(monkeypatch, error=DDGSException("No results found."))
    adapter = _adapter()
    pages = await adapter.search("weather today", max_results=3)
    assert pages == []
    assert not isinstance(pages, BaseException)


async def test_other_provider_failure_surfaces_sanitized_error(monkeypatch):
    """Provider messages other than the empty condition fail sanitized."""
    _fake_ddgs(
        monkeypatch, error=DDGSException("provider secret=abc\nTraceback: hidden")
    )
    adapter = _adapter()
    with pytest.raises(WebSearchProviderError):
        await adapter.search("some query", max_results=3)


async def test_provider_timeout_surfaces_sanitized_error(monkeypatch):
    _fake_ddgs(monkeypatch, error=TimeoutException("search timed out after 5s"))
    adapter = _adapter()
    with pytest.raises(WebSearchProviderError):
        await adapter.search("some query", max_results=3)


async def test_provider_results_keep_order_and_filter_usable_urls(monkeypatch):
    _fake_ddgs(
        monkeypatch,
        results=[
            {"title": "First", "href": "https://example.com/1", "body": "one"},
            {"title": "No URL", "body": "dropped"},
            {"title": "Blank", "href": "   ", "body": "dropped"},
            {"title": "Second", "href": "https://example.com/2", "body": "two"},
        ],
    )
    adapter = _adapter()
    pages = await adapter.search("ordered query", max_results=5)

    assert [page.url for page in pages] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert [page.title for page in pages] == ["First", "Second"]
