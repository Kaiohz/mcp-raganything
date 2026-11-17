"""Response models for the ``web_search`` MCP tool (REQ-006 contract).

These models mirror the domain report 1:1 so the thin tool layer maps a
domain entity with a direct ``**model_dump()`` transformation. They carry no
``chars``/``truncated`` output-size fields by design: complete Crawl4AI
Markdown is always returned.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CrawlResult(BaseModel):
    """Per-page exploration outcome."""

    status: Literal["not_explored", "success", "failed"] = Field(
        description="Exploration status of the page"
    )
    final_url: str | None = Field(
        default=None, description="Last validated URL when known, otherwise null"
    )
    markdown: str | None = Field(
        default=None,
        description="Complete Crawl4AI Markdown for successful pages; never truncated",
    )
    error: str | None = Field(default=None, description="Safe, concise failure reason")


class WebSearchResult(BaseModel):
    """One search result and its exploration outcome."""

    title: str = Field(description="Provider result title")
    url: str = Field(description="Provider result URL")
    snippet: str = Field(description="Provider result snippet")
    crawl: CrawlResult = Field(description="Exploration outcome")


class WebSearchResponse(BaseModel):
    """Complete web-search response in stable DuckDuckGo order."""

    query: str = Field(description="Trimmed search query")
    results: list[WebSearchResult] = Field(
        default_factory=list, description="Results in provider order"
    )
