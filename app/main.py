import threading
import time
from fastapi import HTTPException , FastAPI , UploadFile , File
from FaceModel.utils import save_upload_file, verify_faces, recognize_face, get_embedding, UPLOAD_FOLDER , MODEL_NAME
from Chatbot.voice_auth import authenticate_user
from Chatbot.audio_processing import extract_embedding , convert_to_wav
from Chatbot.text_to_speech import generate_tts
from pydantic import BaseModel
import os

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
async def verify_route(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_path = save_upload_file(file1, UPLOAD_FOLDER)
    img2_path = save_upload_file(file2, UPLOAD_FOLDER)
    return verify_faces(img1_path, img2_path)

@app.post("/recognize/")
async def recognize_route(file: UploadFile = File(...), db_path: str = "database"):
    img_path = save_upload_file(file, UPLOAD_FOLDER)
    return recognize_face(img_path, db_path)

@app.post("/embed/")
async def embed_route(file: UploadFile = File(...)):
    img_path = save_upload_file(file, UPLOAD_FOLDER)
    return get_embedding(img_path)



def delete_file_after_delay(file_path, delay=300):
    """Deletes the file after a specified delay (in seconds)."""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

@app.post("/extract-embedding/")
async def extract_embedding_route(file: UploadFile = File(...)):
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

@app.post("/authenticate/")
async def verify_user(saved_embedding: list, file: UploadFile = File(...)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

# -----------------------------------------------------------------------
# @app.post("/extract-embedding/")
# async def extract_embedding_route(file: UploadFile = File(...)):
#     """
#     Extracts and returns the speaker embedding from an uploaded audio file.
#     """
#     file_path = f"temp_{file.filename}.wav"
#     with open(file_path, "wb") as buffer:
#         buffer.write(file.file.read())

#     try:
#         embedding = extract_embedding(file_path)
#         os.remove(file_path)  # Clean up temp file
#         return {"embedding": embedding.tolist()}
#     except Exception as e:
#         os.remove(file_path)  # Clean up temp file
#         raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")

# @app.post("/authenticate/")
# async def verify_user(saved_embedding: list, file: UploadFile = File(...)):
#     """
#     Compares uploaded audio embedding with a previously saved embedding.
#     """
#     file_path = f"temp_{file.filename}.wav"
#     with open(file_path, "wb") as buffer:
#         buffer.write(file.file.read())

#     try:
#         result = authenticate_user(saved_embedding, file_path)
#         os.remove(file_path)  # Clean up temp file
#         return result
#     except Exception as e:
#         os.remove(file_path)  # Clean up temp file
#         raise HTTPException(status_code=500, detail=f"Error during authentication: {str(e)}")

# -----------------------------------------------------------------------





# from fastapi import FastAPI, UploadFile, File, HTTPException
# import uvicorn
# from deepface import DeepFace
# import shutil
# import os
# import uuid
# import tensorflow as tf

# # Initialize FastAPI
# app = FastAPI()

# # Set up upload folder
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Disable GPU access to avoid CUDA errors
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Limit TensorFlow memory usage
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpu,
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
#             )
#     except RuntimeError as e:
#         print("TensorFlow GPU error:", e)


# @app.get("/")
# async def root():
#     return {"message": "DeepFace API is running!"}


# @app.post("/analyze/")
# async def analyze_image(file: UploadFile = File(...)):
#     """Analyze a single image using DeepFace"""
#     unique_filename = f"{uuid.uuid4()}_{file.filename}"
#     file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

#     # Save uploaded file
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

#     # Analyze image
#     try:
#         result = DeepFace.analyze(img_path=file_path, actions=["age", "gender", "emotion", "race"])
#         return {"filename": file.filename, "analysis": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"DeepFace analysis failed: {str(e)}")
#     finally:
#         os.remove(file_path)  # Cleanup uploaded file after processing


# @app.post("/verify/")
# async def verify_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     """Verify if two uploaded images belong to the same person"""
#     unique_filename1 = f"{uuid.uuid4()}_{file1.filename}"
#     unique_filename2 = f"{uuid.uuid4()}_{file2.filename}"
    
#     path1 = os.path.join(UPLOAD_FOLDER, unique_filename1)
#     path2 = os.path.join(UPLOAD_FOLDER, unique_filename2)

#     # Save uploaded files
#     try:
#         with open(path1, "wb") as buffer1, open(path2, "wb") as buffer2:
#             shutil.copyfileobj(file1.file, buffer1)
#             shutil.copyfileobj(file2.file, buffer2)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

#     # Perform face verification
#     try:
#         result = DeepFace.verify(img1_path=path1, img2_path=path2)
#         return {"verification_result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"DeepFace verification failed: {str(e)}")
#     finally:
#         os.remove(path1)  # Cleanup uploaded files
#         os.remove(path2)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
