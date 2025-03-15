from Chatbot.audio_processing import extract_embedding , convert_to_wav
from config import THRESHOLD
import numpy as np


# compare the embeddings using cosine similarity
def compare_embeddings(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# authenticate the user by comparing the embeddings
def authenticate_user(saved_embedding: list, audio_file_path: str):
    converted_audio = convert_to_wav(audio_file_path)
    new_embedding = extract_embedding(converted_audio)

    similarity_score = compare_embeddings(saved_embedding, new_embedding)

    return {
        "authenticated": similarity_score >= THRESHOLD,
        "similarity_score": similarity_score
    } 
