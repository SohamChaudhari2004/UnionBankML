import threading
import time
from fastapi import HTTPException , FastAPI , UploadFile , File, Depends
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



# Create FastAPI app

app = FastAPI()

PORT = int(os.getenv("PORT", 8080))  # Default to 8080 for Render

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"



@app.post("/generate-tts/")
async def generate_tts_api(request: TTSRequest):
    try:
        file_name = await generate_tts(request.text, request.voice)
        return {"message": f"TTS generated and saved to {file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")  # Health check route
async def root():
    return {"message": "DeepFace FastAPI is running!"}

@app.post("/verify/")
async def verify_image(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_path = save_upload_file(file1, UPLOAD_FOLDER)
    img2_path = save_upload_file(file2, UPLOAD_FOLDER)
    return verify_faces(img1_path, img2_path)

@app.post("/recognize/")
async def recognize_image(file: UploadFile = File(...), db_path: str = "database"):
    img_path = save_upload_file(file, UPLOAD_FOLDER)
    return recognize_face(img_path, db_path)

@app.post("/embed-image/")
async def embed_image(file: UploadFile = File(...)):
    img_path = save_upload_file(file, UPLOAD_FOLDER)
    return get_embedding(img_path)


def delete_file_after_delay(file_path, delay=300):
    """Deletes the file after a specified delay (in seconds)."""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

@app.post("/extract-embedding/")
async def extract_embedding_from_voice(file: UploadFile = File(...)):
    """
    Extracts and returns the speaker embedding from an uploaded audio file.
    """
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Debugging information
    print(f"File saved at: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Convert the file to WAV format
        wav_file_path = convert_to_wav(file_path)
        if wav_file_path is None:
            raise ValueError("Failed to convert audio file to WAV format.")

        # Extract embedding
        embedding = extract_embedding(wav_file_path)
        if embedding is None:
            raise ValueError("Failed to extract embedding from the audio file.")

        # Schedule file deletion after 5 minutes
        threading.Thread(target=delete_file_after_delay, args=(file_path,)).start()
        threading.Thread(target=delete_file_after_delay, args=(wav_file_path,)).start()

        return {"embedding": embedding.tolist()}
    except Exception as e:
        # Clean up temp files in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        if wav_file_path and os.path.exists(wav_file_path):
            os.remove(wav_file_path)
        raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")

@app.post("/authenticate-voice/")
async def verify_user_voice(saved_embedding: list, file: UploadFile = File(...)):
    """
    Compares uploaded audio embedding with a previously saved embedding.
    """
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Convert the file to WAV format
        wav_file_path = convert_to_wav(file_path)
        if wav_file_path is None:
            raise ValueError("Failed to convert audio file to WAV format.")

        # Authenticate user
        result = authenticate_user(saved_embedding, wav_file_path)

        # Clean up temp files
        os.remove(file_path)
        os.remove(wav_file_path)

        return result
    except Exception as e:
        # Clean up temp files in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)
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
        response = requests.get(image_url, timeout=10)
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
