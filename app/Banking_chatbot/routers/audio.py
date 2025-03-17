from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, status
from typing import Optional

from models.audio import AudioTranscriptionRequest, AudioTranscriptionResponse
from models.chat import QueryType, ChatResponse
from services.transcription import TranscriptionService
from services.chat_service import ChatService

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
    responses={404: {"description": "Not found"}},
)


def get_transcription_service(request: Request) -> TranscriptionService:
    """Dependency to get TranscriptionService instance"""
    return request.app.state.transcription_service


def get_chat_service(request: Request) -> ChatService:
    """Dependency to get ChatService instance"""
    return request.app.state.chat_service


@router.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    request_data: str = Form(...),
    transcription_service: TranscriptionService = Depends(get_transcription_service)
):
    """
    Transcribe an audio file to text
    """
    import json
    
    try:
        # Parse request data
        data = json.loads(request_data)
        request = AudioTranscriptionRequest(**data)
        
        # Validate audio file format
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File must be an audio file"
            )
        
        # Transcribe the audio
        result = await transcription_service.transcribe_audio(
            audio_file=audio_file,
            language=request.language
        )
        
        # Return response
        return AudioTranscriptionResponse(
            transcription=result["transcription"],
            session_id=request.session_id or "new",  # Will be replaced if a new session is created
            confidence=result.get("confidence"),
            language=result.get("language"),
            processing_time=result.get("processing_time")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}"
        )


@router.post("/query", response_model=ChatResponse)
async def audio_chat_query(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    use_web_search: bool = Form(False),
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Process an audio chat query and return a response
    """
    try:
        # Validate audio file format
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File must be an audio file"
            )
        
        # Transcribe the audio
        result = await transcription_service.transcribe_audio(
            audio_file=audio_file,
            language=language
        )
        
        transcribed_text = result["transcription"]
        
        # Process the transcribed text as a chat query
        response, session_id, sources = await chat_service.process_query(
            query=transcribed_text,
            session_id=session_id,
            use_web_search=use_web_search,
            query_type=QueryType.AUDIO
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources,
            query_type=QueryType.AUDIO
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio query: {str(e)}"
        )