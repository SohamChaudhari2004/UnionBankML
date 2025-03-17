from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class SourceType(str, Enum):
    DOCUMENT = "document"
    WEB = "web"


class RetrievalResult(BaseModel):
    """Model for retrieval results"""
    source_type: SourceType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = None


class DocumentChunk(BaseModel):
    """Model for document chunks from vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class WebSearchResult(BaseModel):
    """Model for web search results"""
    title: str
    snippet: str
    url: str
    score: Optional[float] = None


class RetrievalRequest(BaseModel):
    """Request model for retrieval"""
    query: str
    top_k: int = 3
    use_web_search: bool = False