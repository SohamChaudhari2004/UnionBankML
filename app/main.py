import threading
import time
from fastapi import HTTPException , FastAPI , UploadFile , File, Depends,Form, BackgroundTasks
from FaceModel.utils import save_upload_file, verify_faces, recognize_face, get_embedding, UPLOAD_FOLDER , MODEL_NAME
from Chatbot.voice_auth import authenticate_user
from Chatbot.audio_processing import extract_embedding , convert_to_wav
from Chatbot.text_to_speech import generate_tts
from pydantic import BaseModel
import os
from fastapi.responses import JSONResponse
import base64
from groq import Groq
from pydantic import BaseModel
import uvicorn
from typing import Optional
import imghdr
import io
from PIL import Image
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from typing import Dict, Optional, List
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Import utils
from BankChatbot.chatbot import LoanChatbot
from BankChatbot.voice import process_voice_input
from BankChatbot.config import GROQ_API_KEY, LOAN_DATA_PATH, TRANSCRIPTION_MODEL, EMBEDDING_MODEL, LLM_MODEL
import os
import requests
import threading
from fastapi import HTTPException, UploadFile, File
from pydub import AudioSegment
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
import os
import requests
import numpy as np
from fastapi import HTTPException
import threading
from sklearn.metrics.pairwise import cosine_similarity

# Create FastAPI app

app = FastAPI()

PORT = int(os.getenv("PORT", 8080))  # Default to 8080 for Render

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"

# Store chatbot instances by session_id
chatbot_instances = {}

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    session_id: str

class TranscriptionResponse(BaseModel):
    text: str
    session_id: str

# Helper function to get or create a chatbot instance
def get_chatbot(session_id: Optional[str] = None):
    """Get or create a chatbot instance for the given session ID."""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in chatbot_instances:
        print(f"Creating new chatbot instance for session {session_id}")
        chatbot_instances[session_id] = LoanChatbot(
            api_key=GROQ_API_KEY, 
            data_path=LOAN_DATA_PATH,
            session_id=session_id
        )
    
    return chatbot_instances[session_id], session_id

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Loan Chatbot API",
        "documentation": "/docs",
        "endpoints": {
            "query": "/query - Process a text query",
            "voice": "/voice - Process voice input",
            "clear": "/clear - Clear conversation history",
            "health": "/health - Health check"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a text query through the conversational RAG system."""
    chatbot, session_id = get_chatbot(request.session_id)
    result = chatbot.process_query(request.query)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": session_id
    }

# @app.post("/voice", response_model=TranscriptionResponse)
# async def handle_voice(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     session_id: Optional[str] = Form(None)
# ):
#     """Process voice input and return transcribed text."""
#     chatbot, session_id = get_chatbot(session_id)
    
#     # Save uploaded file to temporary location
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     try:
#         temp_file.write(await file.read())
#         temp_file.close()
        
#         # Process the voice file
#         transcribed_text = process_voice_input(
#             groq_client=chatbot.groq_client,
#             audio_file_path=temp_file.name,
#             model=TRANSCRIPTION_MODEL
#         )
        
#         if not transcribed_text:
#             raise HTTPException(status_code=400, detail="Failed to transcribe audio")
            
#         # Delete temp file in the background after response is sent
#         background_tasks.add_task(os.unlink, temp_file.name)

#         response = chatbot.process_query(transcribed_text)
#         return {
#             "text": transcribed_text,
#             "session_id": session_id,
#             "response" : response,
#         }
        
#     except Exception as e:
#         # Make sure to clean up temp file in case of an error
#         if os.path.exists(temp_file.name):
#             os.unlink(temp_file.name)
#         raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")



@app.post("/voice", response_model=TranscriptionResponse)
async def handle_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Process voice input and return transcribed text along with chatbot response."""
    chatbot, session_id = get_chatbot(session_id)

    # Save uploaded file to temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(await file.read())
        temp_file.close()

        # Process the voice file
        transcribed_text = process_voice_input(
            groq_client=chatbot.groq_client,
            audio_file_path=temp_file.name,
            model=TRANSCRIPTION_MODEL
        )

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        # Process the transcribed text as a query
        response = chatbot.process_query(transcribed_text)

        # Delete temp file in the background after response is sent
        background_tasks.add_task(os.unlink, temp_file.name)

        return {
            "text": transcribed_text,
            "session_id": session_id,
            "response": {
                "answer": response["answer"],
                "sources": response["sources"]
            }
        }

    except Exception as e:
        # Make sure to clean up temp file in case of an error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")
    


# @app.post("/trans_voice")
async def Transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Process voice input and return transcribed text along with chatbot response."""
    chatbot, session_id = get_chatbot(session_id)

    # Save uploaded file to temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(await file.read())
        temp_file.close()

        # Process the voice file
        transcribed_text = process_voice_input(
            groq_client=chatbot.groq_client,
            audio_file_path=temp_file.name,
            model=TRANSCRIPTION_MODEL
        )

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")


        # Delete temp file in the background after response is sent
        background_tasks.add_task(os.unlink, temp_file.name)
        res = {
            "text": transcribed_text,
            "session_id": session_id,
        }
        return res

    except Exception as e:
        # Make sure to clean up temp file in case of an error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")


# @app.post("/query_voice_direct")
# async def query_from_voice_direct(
#     background_tasks: BackgroundTasks,
#     file_path: str,  # Accept the file path directly
#     session_id: Optional[str] = None
# ):
#     """Process voice input from a direct file path and return both transcription and chatbot answer."""
#     chatbot, session_id = get_chatbot(session_id)

#     try:
#         # Process the voice file directly from the given path
#         transcribed_text = process_voice_input(
#             groq_client=chatbot.groq_client,
#             audio_file_path=file_path,
#             model=TRANSCRIPTION_MODEL
#         )

#         if not transcribed_text:
#             raise HTTPException(status_code=400, detail="Failed to transcribe audio")

#         # Process the transcribed text as a query
#         result = chatbot.process_query(transcribed_text)

#         return {
#             "transcription": transcribed_text,
#             "answer": result["answer"],
#             "sources": result["sources"],
#             "session_id": session_id
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")


@app.post("/clear")
async def clear_history(request: QueryRequest):
    """Clear conversation history for a session."""
    chatbot, session_id = get_chatbot(request.session_id)
    result = chatbot.clear_memory()
    return {
        **result,
        "session_id": session_id
    }


# working
@app.post("/query_voice")
async def query_from_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Process voice input and return both transcription and chatbot answer."""
    chatbot, session_id = get_chatbot(session_id)
    
    # Save uploaded file to temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(await file.read())
        temp_file.close()
        
        # Process the voice file
        transcribed_text = process_voice_input(
            groq_client=chatbot.groq_client,
            audio_file_path=temp_file.name,
            model=TRANSCRIPTION_MODEL
        )
        
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
            
        # Process the transcribed text as a query
        result = chatbot.process_query(transcribed_text)
        
        # Delete temp file in the background after response is sent
        background_tasks.add_task(os.unlink, temp_file.name)
        
        return {
            "transcription": transcribed_text,
            "answer": result["answer"],
            "sources": result["sources"],
            "session_id": session_id
        }
        
    except Exception as e:
        # Make sure to clean up temp file in case of an error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")


@app.post("/generate-tts/")
async def generate_tts_api(request: TTSRequest):
    try:    
        file_name = await generate_tts(request.text, request.voice)
        return {"message": f"TTS generated and saved to {file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")  # Health check route
async def root():
    return {"message": "FastAPI is running!"}

# @app.post("/verify/")
# async def verify_image(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     img1_path = save_upload_file(file1, UPLOAD_FOLDER)
#     img2_path = save_upload_file(file2, UPLOAD_FOLDER)
#     return verify_faces(img1_path, img2_path)

class ImageURLs(BaseModel):
    url1: str
    url2: str

from deepface import DeepFace

def verify_faces_with_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> dict:
    try:
        # Calculate cosine similarity
        similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]

        # Define a threshold for verification
        threshold = 0.6  # You can adjust this threshold based on your requirements
        verified = similarity_score >= threshold

        return {"verified": verified, "similarity_score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_image/")
async def verify_image(image_urls: ImageURLs):
    try:
        # Download the images from the URLs
        response1 = requests.get(image_urls.url1)
        response1.raise_for_status()
        response2 = requests.get(image_urls.url2)
        response2.raise_for_status()

        # Save the images temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img2:
            temp_img1.write(response1.content)
            temp_img2.write(response2.content)
            temp_img1_path = temp_img1.name
            temp_img2_path = temp_img2.name

        # Verify faces using deepface verify func.
        result = DeepFace.verify(temp_img1_path, temp_img2_path, model_name=MODEL_NAME)

        return result
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading images: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")

@app.post("/recognize/")
async def recognize_image(file: UploadFile = File(...), db_path: str = "database"):
    img_path = save_upload_file(file, UPLOAD_FOLDER)
    return recognize_face(img_path, db_path)


import requests
from io import BytesIO
# @app.post("/embed-image/")
# async def embed_image(file: UploadFile = File(...)):
#     img_path = save_upload_file(file, UPLOAD_FOLDER)
#     return get_embedding(img_path)


class ImageURL(BaseModel):
    url: str

# def get_embedding(image_path: str):
#     # Your existing embedding logic here
#     pass

@app.post("/embed-image/")
async def embed_image(image_url: ImageURL):
    try:
        # Download the image from the URL
        response = requests.get(image_url.url)
        response.raise_for_status()

        # Save the image temporarily
        img_data = BytesIO(response.content)

        # Create a temporary file to save the image data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(img_data.getvalue())
            temp_img_path = temp_img.name

        # Get the embedding using the temporary file path
        embedding = get_embedding(temp_img_path)

        return embedding
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
def delete_file_after_delay(file_path, delay=300):
    """Deletes the file after a specified delay (in seconds)."""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

# @app.post("/extract-embedding/")
# async def extract_embedding_from_voice(file: UploadFile = File(...)):
#     """
#     Extracts and returns the speaker embedding from an uploaded audio file.
#     """
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     # Debugging information
#     print(f"File saved at: {file_path}")
#     print(f"File exists: {os.path.exists(file_path)}")
    
#     try:
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist.")

#         # Convert the file to WAV format
#         wav_file_path = convert_to_wav(file_path)
#         if wav_file_path is None:
#             raise ValueError("Failed to convert audio file to WAV format.")

#         # Extract embedding
#         embedding = extract_embedding(wav_file_path)
#         if embedding is None:
#             raise ValueError("Failed to extract embedding from the audio file.")

#         # Schedule file deletion after 5 minutes
#         threading.Thread(target=delete_file_after_delay, args=(file_path,)).start()
#         threading.Thread(target=delete_file_after_delay, args=(wav_file_path,)).start()

#         return {"embedding": embedding.tolist()}
#     except Exception as e:
#         # Clean up temp files in case of error
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         if wav_file_path and os.path.exists(wav_file_path):
#             os.remove(wav_file_path)
#         raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")




@app.post("/extract-embedding/")
async def extract_embedding_from_voice(url: str):
    """
    Extracts and returns the speaker embedding from an audio file at a given URL.
    """
    # Define the local file path
    file_path = "temp_audio_file"

    try:
        # Download the file from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file locally
        with open(file_path, "wb") as buffer:
            buffer.write(response.content)

        # Debugging information
        print(f"File saved at: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Extract embedding directly from the downloaded file
        embedding = extract_embedding(file_path)
        if embedding is None:
            raise ValueError("Failed to extract embedding from the audio file.")

        # Schedule file deletion after 5 minutes
        threading.Thread(target=delete_file_after_delay, args=(file_path,)).start()

        return {"embedding": embedding.tolist()}
    except Exception as e:
        # Clean up temp files in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")




@app.post("/authenticatevoice/")
async def verify_uservoice(saved_audio_url: str, input_audio_url: str):
    """
    Compares the embeddings from two audio file URLs.
    """
    saved_audio_path = "saved_audio_file"
    input_audio_path = "input_audio_file"

    try:
        # Download the saved audio file
        response = requests.get(saved_audio_url)
        response.raise_for_status()
        with open(saved_audio_path, "wb") as buffer:
            buffer.write(response.content)

        # Download the input audio file
        response = requests.get(input_audio_url)
        response.raise_for_status()
        with open(input_audio_path, "wb") as buffer:
            buffer.write(response.content)

        # Extract embeddings from both audio files
        saved_embedding = extract_embedding(saved_audio_path)
        input_embedding = extract_embedding(input_audio_path)

        if saved_embedding is None or input_embedding is None:
            raise ValueError("Failed to extract embedding from one or both audio files.")

        # Calculate cosine similarity
        similarity = cosine_similarity([saved_embedding], [input_embedding])[0][0]

        # Convert similarity to a native Python float
        similarity_float = float(similarity)

        # Schedule file deletion after 5 minutes
        threading.Thread(target=delete_file_after_delay, args=(saved_audio_path,)).start()
        threading.Thread(target=delete_file_after_delay, args=(input_audio_path,)).start()

        return {"similarity": similarity_float}
    except Exception as e:
        # Clean up temp files in case of error
        if os.path.exists(saved_audio_path):
            os.remove(saved_audio_path)
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)
        raise HTTPException(status_code=500, detail=f"Error during authentication: {str(e)}")

# -------------------------------------AADHAR OCR--------------------------------------------------------
client = Groq(
    api_key='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'
)

# Aadhaar-specific prompt template
AADHAAR_PROMPT = """
Please analyze this image and extract the following information in a Json format:

1. Mobile No (It is a 10 digit number only.)
Give me the output in json format.  
only return json format.
don't return any other format or text.
"""

# List of accepted image formats
ACCEPTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "bmp", "gif", "tiff", "webp"]

def validate_image_url(image_url: str):
    """
    Validate that the Cloudinary URL points to an acceptable image format
    """
    # Check if the URL has an extension we can verify
    ext = os.path.splitext(image_url)[1].lower().lstrip('.')
    
    if ext and ext in ACCEPTED_IMAGE_FORMATS:
        return True
    
    # If we can't determine from URL, try to fetch headers
    try:
        response = requests.head(image_url, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        
        if any(f"image/{fmt}" in content_type for fmt in ACCEPTED_IMAGE_FORMATS):
            return True
            
        # If content type doesn't help, we'll have to download a bit of the image
        if 'image/' in content_type:
            return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error validating image URL: {str(e)}")
    
    raise HTTPException(
        status_code=400,
        detail=f"Cannot verify if URL points to a valid image. Accepted formats: {', '.join(ACCEPTED_IMAGE_FORMATS)}"
    )

def download_image_from_url(image_url: str):
    """
    Download image from Cloudinary URL
    """
    try:
        response = requests.get(image_url, timeout=2000)
        response.raise_for_status()
        
        image_data = response.content
        
        # Verify it's a valid image
        file_type = imghdr.what(None, h=image_data)
        
        if not file_type or file_type not in ACCEPTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail="The URL does not point to a valid image."
            )
        
        # Verify image can be opened by PIL
        try:
            Image.open(io.BytesIO(image_data))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="The image cannot be processed."
            )
            
        return image_data
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

def encode_image_to_base64(image_data):
    """Convert image data to base64 encoding"""
    return base64.b64encode(image_data).decode('utf-8')

def process_aadhaar_image(image_data, custom_prompt=None):
    """Process the uploaded Aadhaar image"""
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_data)
        
        # Prepare the prompt
        prompt = custom_prompt if custom_prompt else AADHAAR_PROMPT
        
        # Create the message with the image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call the Groq API
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.1,  # Lower temperature for more accurate extraction
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        raw_result = completion.choices[0].message.content
        
        return {
            "aadhaar_details": raw_result,
            "model": completion.model,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Aadhaar: {str(e)}")

@app.post("/aadhaar/process-url/", response_class=JSONResponse)
async def process_aadhaar_url(image_url: str, custom_prompt: Optional[str] = None):
    """Endpoint to process an Aadhaar card image from a Cloudinary URL"""
    try:
        # Validate the URL points to an acceptable image
        validate_image_url(image_url)
        
        # Download the image
        image_data = download_image_from_url(image_url)
        
        # Process the image
        result = process_aadhaar_image(image_data, custom_prompt)
        
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")

# Keep the original file upload endpoint for backward compatibility
@app.post("/aadhaar/upload/", response_class=JSONResponse)
async def upload_aadhaar(
    file: UploadFile = Depends(validate_image_url), 
    custom_prompt: Optional[str] = None
):
    """Endpoint to upload an Aadhaar card image and perform OCR"""
    try:
        # Read the validated image file
        contents = await file.read()
        
        # Process the image
        result = process_aadhaar_image(contents, custom_prompt)
        
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
