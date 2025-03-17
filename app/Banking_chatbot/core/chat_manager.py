from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Any, Optional

class ChatSession:
    def __init__(self, session_id: str = None, user_id: str = None):
        self.session_id = session_id or str(uuid4())
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.messages = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the chat history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
        }
        
        if metadata:
            message["metadata"] = metadata
            
        self.messages.append(message)
        self.last_active = datetime.now()
        return message
    
    def get_messages(self, limit: int = None):
        """Get the most recent messages from the chat history"""
        if limit is None:
            return self.messages
        return self.messages[-limit:]
    
    def clear_history(self):
        """Clear the chat history"""
        self.messages = []
        self.last_active = datetime.now()


class ChatManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def create_session(self, user_id: Optional[str] = None) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(user_id=user_id)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID"""
        return self.sessions.get(session_id)
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[ChatSession]:
        """List all chat sessions, optionally filtered by user ID"""
        if user_id:
            return [session for session in self.sessions.values() if session.user_id == user_id]
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session by ID"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def clean_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than the specified age"""
        now = datetime.now()
        to_delete = []
        
        for session_id, session in self.sessions.items():
            age = (now - session.last_active).total_seconds() / 3600
            if age > max_age_hours:
                to_delete.append(session_id)
        
        for session_id in to_delete:
            del self.sessions[session_id]
        
        return len(to_delete)