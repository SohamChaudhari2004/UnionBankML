from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """Request model for text-based chat"""
    query: str = Field(..., description="User query text")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    use_web_search: Optional[bool] = Field(False, description="Whether to enable active retrieval")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the current mortgage rates?",
                "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "use_web_search": True
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat"""
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session ID")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="Sources used for generating the response"
    )
    query_type: QueryType = Field(QueryType.TEXT, description="Type of query processed")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatHistoryRequest(BaseModel):
    """Request model for retrieving chat history"""
    session_id: str = Field(..., description="Chat session ID")
    limit: Optional[int] = Field(None, description="Maximum number of messages to retrieve")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history"""
    session_id: str
    messages: List[Message]
    created_at: datetime
    last_active: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NewSessionResponse(BaseModel):
    """Response model for creating a new chat session"""
    session_id: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }