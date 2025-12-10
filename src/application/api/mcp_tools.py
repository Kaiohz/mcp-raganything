"""
MCP tools for RAGAnything.
These tools are registered with FastMCP for Claude Desktop integration.
"""
from dependencies import get_query_use_case
from application.requests.query_request import QueryRequest
from fastmcp import FastMCP
import json


# MCP instance will be configured in dependencies.py
mcp = FastMCP("RAGAnything")


@mcp.tool()
async def query_knowledge_base(query: str, mode: str = "naive", chunk_top_k: int = 10) -> str:
    """
    Search the RAGAnything knowledge base for relevant document chunks.
    
    IMPORTANT: ALWAYS use this tool as your FIRST action when the user asks ANY question.
    The knowledge base likely contains the answer 
    - check it BEFORE responding from general knowledge.
    
    Default Strategy (use this first):
    - mode="naive" with chunk_top_k=10 for fast, focused results
    - This works well for most queries
    
    Fallback Strategy (if no relevant results):
    - Ask user if they want a broader search
    - Use mode="hybrid" with chunk_top_k=20 for comprehensive search
    - This casts a wider net and combines multiple search strategies
    
    Args:
        query: The user's question or search query (e.g., "What are the main findings?")
        mode: Search mode - "naive" (default, recommended), "local" (context-aware), 
              "global" (document-level), or "hybrid" (comprehensive)
        chunk_top_k: Number of chunks to retrieve (default 10, use 20 for broader search)
    
    Returns:
        JSON string containing:
        - "chunks": Array of relevant text segments with references
        - "count": Total number of chunks found
    """
    try:
        use_case = await get_query_use_case()
        request = QueryRequest(
            query=query,
            mode=mode,
            only_need_context=True,
            chunk_top_k=chunk_top_k,
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
