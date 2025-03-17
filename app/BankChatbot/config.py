"""Configuration settings for the Loan Chatbot application."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API settings
GROQ_API_KEY = "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" 
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable is not set")

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"

# Vector DB settings
DB_DIRECTORY = "loan_chroma_db"

# Data settings
LOAN_DATA_PATH = os.getenv("LOAN_DATA_PATH", "loan_data.txt")
if not os.path.exists(LOAN_DATA_PATH):
    print(f"Warning: Loan data file not found at {LOAN_DATA_PATH}")

# Chat settings
TEMPERATURE = 0.2
MAX_TOKENS = 2048
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5