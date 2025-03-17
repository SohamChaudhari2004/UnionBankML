import os
from fastapi import UploadFile, HTTPException
from deepface import DeepFace
import shutil

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_NAME = "Facenet512"

# def save_upload_file(upload_file: UploadFile, folder: str) -> str:
#     file_path = os.path.join(folder, upload_file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(upload_file.file, buffer)
#     return file_path

def save_upload_file(upload_file: UploadFile, destination_folder: str) -> str:
    file_location = os.path.join(destination_folder, upload_file.filename)
    with open(file_location, "wb") as file:
        shutil.copyfileobj(upload_file.file, file)
    return file_location

def verify_faces(file1_path: str, file2_path: str) -> dict:
    try:
        result = DeepFace.verify(file1_path, file2_path, model_name=MODEL_NAME)
        return {"verified": result["verified"], "distance": result["distance"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def recognize_face(img_path: str, db_path: str = "database") -> dict:
    try:
        dfs = DeepFace.find(img_path=img_path, db_path=db_path, model_name=MODEL_NAME)
        return {"matches": dfs[0].to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
def get_embedding(img_path: str) -> dict:
    try:
        embeddings = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME)
        return {"embedding": embeddings[0]["embedding"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
