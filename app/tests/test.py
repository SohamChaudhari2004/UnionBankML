from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import base64
from groq import Groq
from pydantic import BaseModel
import uvicorn
from typing import Optional
import imghdr
import io
from PIL import Image

app = FastAPI(title="Aadhaar Card OCR API", description="API for extracting details from Aadhaar cards using Groq's vision model")

# Configure Groq client
# Make sure to set your GROQ_API_KEY environment variable
client = Groq(
    api_key='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'
)

# Aadhaar-specific prompt template
AADHAAR_PROMPT = """
Please analyze this image and extract the following information in a Json format:

1. Mobile No (It should be 10 digits)
Give me the output in json format.  
only return json format.
don't return any other format or text.

"""

# List of accepted image formats
ACCEPTED_IMAGE_FORMATS = ["jpeg", "jpg", "png", "bmp", "gif", "tiff", "webp"]

async def validate_image(file: UploadFile = File(...)) -> UploadFile:
    """
    Dependency to validate that the uploaded file is a valid image format.
    """
    # Check the file extension
    ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if ext not in ACCEPTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Accepted formats: {', '.join(ACCEPTED_IMAGE_FORMATS)}"
        )
    
    # Read the file content
    contents = await file.read()
    
    # Detect the file type from its content
    file_type = imghdr.what(None, h=contents)
    
    # Reset the file pointer for later use
    await file.seek(0)
    
    if not file_type or file_type not in ACCEPTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail="The uploaded file is not a valid image."
        )
    
    # Verify image can be opened by PIL
    try:
        Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file cannot be processed as an image."
        )
    
    # Reset the file pointer again
    await file.seek(0)
    return file

def encode_image_to_base64(image_data):
    """Convert image data to base64 encoding"""
    return base64.b64encode(image_data).decode('utf-8')

def process_aadhaar_image(image_data, custom_prompt=None):
    """Process the uploaded Aadhaar image"""
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_data)
        
        # Prepare the prompt
        prompt = AADHAAR_PROMPT
        
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
        
        # For simplicity, we're returning the raw result
        # In a production app, you might want to parse this into structured data
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

@app.post("/aadhaar/upload/", response_class=JSONResponse)
async def upload_aadhaar(
    file: UploadFile = Depends(validate_image), 
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Aadhaar OCR API"}

if __name__ == "__main__":
    uvicorn.run("aadhaar_image_ocr:app", host="0.0.0.0", port=8000, reload=True)