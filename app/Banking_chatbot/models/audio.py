from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription"""
    session_id: Optional[str] = Field(
        None, description="Chat session ID for continuous conversation"
    )
    language: Optional[str] = Field(
        None, description="Language code (if known) to improve transcription"
    )


class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription"""
    transcription: str = Field(..., description="Transcribed text from audio")
    session_id: str = Field(..., description="Chat session ID")
    confidence: Optional[float] = Field(None, description="Confidence score for transcription")
    language: Optional[str] = Field(None, description="Detected language")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }