"""
MCP tools for RAGAnything.
These tools are registered with FastMCP for Claude Desktop integration.
"""

from dependencies import get_lightrag_proxy_use_case
from domain.entities.lightrag_proxy_entities import LightRAGProxyRequest
from fastmcp import FastMCP


# MCP instance will be configured in dependencies.py
mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(
    query: str, mode: str = "naive", top_k: int = 10, only_need_context: bool = True
) -> str:
    """
    Search the RAGAnything knowledge base for relevant document chunks.

    Default Strategy (use this first):
    - mode="naive" with top_k=10 for fast, focused results
    - This works well for most queries

    Fallback Strategy (if no relevant results):
    - Ask user if they want a broader search
    - Use mode="hybrid" with top_k=20 for comprehensive search
    - This casts a wider net and combines multiple search strategies

    Args:
        query: The user's question or search query (e.g., "What are the main findings?")
        mode: Search mode - "naive" (default, recommended), "local" (context-aware),
              "global" (document-level), or "hybrid" (comprehensive)
        top_k: Number of chunks to retrieve (default 10, use 20 for broader search)
        only_need_context: If True, returns only context chunks without LLM answer (default True)

    Returns:
        JSON string containing the query response from LightRAG
    """
    use_case = await get_lightrag_proxy_use_case()

    # Build request body for LightRAG /query endpoint
    request_body = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "only_need_context": only_need_context,
    }

    proxy_request = LightRAGProxyRequest(
        method="POST",
        path="query",
        body=request_body,
    )

    response = await use_case.execute(proxy_request)

    return response.content.decode("utf-8")


@mcp.tool()
async def query_knowledge_base_multimodal(
    query: str,
    mode: str = "naive",
    top_k: int = 10,
    image_path: str = None,
    image_base64: str = None,
    table_data: str = None,
    table_caption: str = None,
    equation_latex: str = None,
    equation_caption: str = None,
) -> str:
    """
    Query the knowledge base with multimodal content support (images, tables, equations).

    This tool extends the standard query with support for visual and structured data.
    Use this when you need to include images, tables, or equations in your query.

    Multimodal Query Strategy:
    - Use mode="hybrid" for best results with multimodal content
    - Provide descriptive captions for better context understanding
    - Combine multiple content types if needed (e.g., image + table)

    Args:
        query: The user's question or search query
        mode: Search mode - "naive", "local", "global", or "hybrid" (recommended for multimodal)
        top_k: Number of chunks to retrieve (default 10)
        image_path: Path to an image file to include in the query
        image_base64: Base64-encoded image data (alternative to image_path)
        table_data: CSV-formatted table data (e.g., "col1,col2\\nval1,val2")
        table_caption: Optional caption describing the table
        equation_latex: LaTeX equation (e.g., "E=mc^2")
        equation_caption: Optional caption describing the equation

    Returns:
        JSON string containing the query response with multimodal analysis

    Examples:
        # Query with an image
        result = await query_knowledge_base_multimodal(
            query="What does this diagram show?",
            image_path="/path/to/diagram.png",
            mode="hybrid"
        )

        # Query with a table
        result = await query_knowledge_base_multimodal(
            query="Compare these performance metrics with the document",
            table_data="Method,Accuracy,Speed\\nRAG,95%,120ms\\nBaseline,87%,180ms",
            table_caption="Performance comparison",
            mode="hybrid"
        )

        # Query with an equation
        result = await query_knowledge_base_multimodal(
            query="Explain this formula in context of the document",
            equation_latex="P(d|q) = \\\\frac{P(q|d) \\\\cdot P(d)}{P(q)}",
            equation_caption="Document relevance probability",
            mode="hybrid"
        )
    """
    use_case = await get_lightrag_proxy_use_case()

    # Build multimodal content list
    multimodal_content = []

    if image_path or image_base64:
        image_item = {"type": "image"}
        if image_path:
            image_item["img_path"] = image_path
        if image_base64:
            image_item["image_data"] = image_base64
        multimodal_content.append(image_item)

    if table_data:
        table_item = {"type": "table", "table_data": table_data}
        if table_caption:
            table_item["table_caption"] = table_caption
        multimodal_content.append(table_item)

    if equation_latex:
        equation_item = {"type": "equation", "latex": equation_latex}
        if equation_caption:
            equation_item["equation_caption"] = equation_caption
        multimodal_content.append(equation_item)

    # Build request based on whether multimodal content exists
    if multimodal_content:
        # Use multimodal query format
        request_body = {
            "query": query,
            "multimodal_content": multimodal_content,
            "mode": mode,
            "top_k": top_k,
        }
    else:
        # Fallback to standard query if no multimodal content provided
        request_body = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "only_need_context": True,
        }

    proxy_request = LightRAGProxyRequest(
        method="POST",
        path="query",
        body=request_body,
    )

    response = await use_case.execute(proxy_request)
    return response.content.decode("utf-8")
