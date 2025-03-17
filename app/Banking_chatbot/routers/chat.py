from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Optional, List

from app.models.chat import (
    ChatRequest, ChatResponse, ChatHistoryRequest, 
    ChatHistoryResponse, NewSessionResponse, QueryType
)
from app.services.chat_service import ChatService
from app.core.chat_manager import ChatManager

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


def get_chat_service(request: Request) -> ChatService:
    """Dependency to get ChatService instance"""
    return request.app.state.chat_service


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Process a text chat query and return a response
    """
    try:
        response, session_id, sources = await chat_service.process_query(
            query=request.query,
            session_id=request.session_id,
            use_web_search=request.use_web_search,
            query_type=QueryType.TEXT
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources,
            query_type=QueryType.TEXT
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat query: {str(e)}"
        )


@router.post("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    request: ChatHistoryRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Get chat history for a session
    """
    chat_manager = chat_service.chat_manager
    session = chat_manager.get_session(request.session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found with ID: {request.session_id}"
        )
    
    messages = session.get_messages(limit=request.limit)
    
    return ChatHistoryResponse(
        session_id=session.session_id,
        messages=messages,
        created_at=session.created_at,
        last_active=session.last_active
    )


@router.post("/session", response_model=NewSessionResponse)
async def create_chat_session(
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Create a new chat session
    """
    session = chat_service.chat_manager.create_session()
    
    return NewSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Delete a chat session
    """
    result = chat_service.chat_manager.delete_session(session_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found with ID: {session_id}"
        )


@router.delete("/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Clear chat history for a session
    """
    result = chat_service.clear_chat_history(session_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found with ID: {session_id}"
        )