from typing import List, Dict, Any, Optional, Tuple
from services.retrieval import RetrievalService
from services.web_search import WebSearchService
from services.llm import LLMService
from core.chat_manager import ChatManager, ChatSession
from models.retrieval import RetrievalResult
from models.chat import QueryType


class ChatService:
    def __init__(self, retrieval_service: RetrievalService, web_search_service: WebSearchService, 
                 llm_service: LLMService, chat_manager: ChatManager):
        self.retrieval_service = retrieval_service
        self.web_search_service = web_search_service
        self.llm_service = llm_service
        self.chat_manager = chat_manager
        
    async def process_query(self, query: str, session_id: Optional[str] = None, 
                         use_web_search: bool = False, query_type: QueryType = QueryType.TEXT
                        ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Process a user query and generate a response
        
        Args:
            query: User query text
            session_id: Optional chat session ID
            use_web_search: Whether to use web search if local retrieval fails
            query_type: Type of query (text or audio)
            
        Returns:
            Tuple of (response text, session ID, sources used)
        """
        # Get or create chat session
        session = self._get_or_create_session(session_id)
        
        # Add user message to history
        session.add_message("user", query, {"query_type": query_type.value})
        
        # Retrieve relevant information from documents
        retrieved_results = await self.retrieval_service.retrieve(query)
        
        # Check if document retrieval was successful
        has_good_results = self._has_good_results(retrieved_results)
        
        # If no good results and web search is enabled, try web search
        sources = []
        if not has_good_results and use_web_search:
            web_results = await self.web_search_service.retrieve(query)
            if web_results:
                retrieved_results = web_results
                has_good_results = True
        
        # Prepare sources for response
        if retrieved_results:
            sources = self._prepare_sources(retrieved_results)
        
        # Get chat history for context
        chat_history = session.get_messages(limit=10)
        
        # Generate response
        response = await self.llm_service.generate_response(
            query, 
            retrieved_context=retrieved_results if has_good_results else None,
            chat_history=chat_history
        )
        
        # Add assistant message to history
        session.add_message("assistant", response)
        
        return response, session.session_id, sources
    
    def _get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get an existing session or create a new one"""
        if session_id:
            session = self.chat_manager.get_session(session_id)
            if session:
                return session
        
        # Create new session if none exists or session_id is invalid
        return self.chat_manager.create_session()
    
    def _has_good_results(self, results: List[RetrievalResult]) -> bool:
        """Determine if retrieved results are good enough"""
        # Simple implementation - check if we have any results
        if not results:
            return False
        
        # More sophisticated implementations could check relevance scores
        # or content quality
        return True
    
    def _prepare_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Prepare sources information for response"""
        sources = []
        
        for i, result in enumerate(results):
            source = {
                "source_type": result.source_type,
                "content_preview": result.content[:150] + "..." if len(result.content) > 150 else result.content
            }
            
            # Add metadata
            if result.metadata:
                source.update(result.metadata)
                
            # Add relevance score if available
            if result.relevance_score is not None:
                source["relevance_score"] = result.relevance_score
                
            sources.append(source)
            
        return sources
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        session = self.chat_manager.get_session(session_id)
        if session:
            return session.get_messages(limit=limit)
        return []
    
    def clear_chat_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        session = self.chat_manager.get_session(session_id)
        if session:
            session.clear_history()
            return True
        return False