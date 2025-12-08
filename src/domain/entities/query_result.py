from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class QueryResult(BaseModel):
    """
    Represents the result of a RAG query.
    """

    query: str = Field(..., description="The original query string")
    answer: Optional[str] = Field(
        default=None, description="Generated answer from the LLM"
    )
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved document chunks"
    )
    entities: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Extracted entities from the knowledge graph"
    )
    relationships: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Extracted relationships from the knowledge graph"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the query"
    )
