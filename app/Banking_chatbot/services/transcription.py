import os
import tempfile
import time
from groq import Groq
from core.config import settings

class TranscriptionService:
    def __init__(self):
        self.client = Groq(api_key='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')
        self.model = 'whisper-large-v3-turbo'
    
    async def transcribe_audio(self, audio_file, language=None):
        """
        Transcribe audio using Groq's Whisper model
        
        Args:
            audio_file: UploadFile object containing audio data
            language: Optional language code to improve transcription
            
        Returns:
            Dict containing transcription results
        """
        start_time = time.time()
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            # Write the file content
            content = await audio_file.read()
            temp_file.write(content)
        
        try:
            # Get file info
            file_size = os.path.getsize(temp_file_path)
            
            # Prepare parameters for transcription
            params = {
                "file": open(temp_file_path, "rb"),
                "model": self.model,
            }
            
            if language:
                params["language"] = language
            
            # Perform transcription
            response = self.client.audio.transcriptions.create(**params)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format response
            result = {
                "transcription": response.text,
                "processing_time": processing_time,
                "file_size_bytes": file_size,
            }
            
            if hasattr(response, "language"):
                result["language"] = response.language
            
            if hasattr(response, "confidence"):
                result["confidence"] = response.confidence
            
            return result
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)