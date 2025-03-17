import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # LLM settings
    LLAMA_MODEL_NAME: str = os.getenv("LLAMA_MODEL_NAME", "llama2-70b-4096")
    WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "whisper-large-v3")
    
    # Database settings
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./data/vector_db")
    
    # Web search settings
    DDG_SEARCH_MAX_RESULTS: int = int(os.getenv("DDG_SEARCH_MAX_RESULTS", "3"))
    
    # Server settings
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_for_jwt_here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File paths
    BANKING_DOCS_PATH: str = "./data/banking_docs.txt"
    
    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Application settings
    APP_NAME: str = "Banking Assistant API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "FastAPI server with chatbot for banking query assistance"

settings = Settings()