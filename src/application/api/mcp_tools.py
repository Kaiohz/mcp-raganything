"""
MCP tools for RAGAnything.
These tools are registered with FastMCP for Claude Desktop integration.
"""
from fastmcp import FastMCP
import json


# MCP instance will be configured in dependencies.py
mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """
    Query the RAGAnything knowledge base and retrieve relevant document chunks.
    
    Args:
        query: The question or query to search for in the knowledge base
    
    Returns:
        str: JSON string containing relevant chunks from the knowledge base
    """
    # This is a placeholder - actual implementation will be injected via dependencies
    # The real query logic will be wired through dependency injection
    from dependencies import get_query_use_case
    from application.requests.query_request import QueryRequest
    
    try:
        use_case = await get_query_use_case()
        request = QueryRequest(
            query=query,
            mode="naive",
            only_need_context=True,
            chunk_top_k=10,
            include_references=True,
            enable_rerank=False
        )
        
        result = await use_case.execute(request)
        
        if not result.chunks:
            return "No relevant chunks found for your query."
        
        return json.dumps({
            "chunks": result.chunks,
            "count": len(result.chunks)
        }, indent=2)
        
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"
