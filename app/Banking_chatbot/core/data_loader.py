import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings

class DataLoader:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Ensure vector_db directory exists
        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="banking_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def load_documents(self, file_path=None):
        """Load documents from the specified file and index them in ChromaDB"""
        if file_path is None:
            file_path = settings.BANKING_DOCS_PATH
        
        if not os.path.exists(file_path):
            print(f"Warning: Document file not found at {file_path}")
            return False
        
        # Load and split the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Check if collection is empty before loading
        if self.collection.count() > 0:
            print("Collection already populated. Skipping document loading.")
            return True
        
        # Prepare data for chromadb
        ids = [f"doc_{i}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add embeddings to collection
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add documents to collection
        for i in range(len(chunks)):
            self.collection.add(
                ids=[ids[i]],
                embeddings=[embeddings[i]],
                metadatas=[metadatas[i]],
                documents=[texts[i]]
            )
        
        print(f"Added {len(chunks)} document chunks to vector database")
        return True
    
    def get_collection(self):
        """Return the ChromaDB collection"""
        return self.collection