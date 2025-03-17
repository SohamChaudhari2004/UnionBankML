from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional
from models.retrieval import RetrievalResult, SourceType, DocumentChunk
from core.config import settings
import os


class RetrievalService:
    def __init__(self, collection=None):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        if collection:
            self.collection = collection
        else:
            # Initialize a new connection to ChromaDB
            client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
            
            # Get collection if it exists
            if "banking_documents" in [col.name for col in client.list_collections()]:
                self.collection = client.get_collection("banking_documents")
            else:
                # Create collection if it doesn't exist
                self.collection = client.create_collection(
                    name="banking_documents",
                    metadata={"hnsw:space": "cosine"}
                )
        
    async def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Get embedding for query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieval_results = []
        
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                # Convert distance to a similarity score (1 - distance)
                # Chromadb uses cosine distance, so higher is better
                score = float(results["distances"][0][i]) if results["distances"] else None
                
                retrieval_results.append(
                    RetrievalResult(
                        source_type=SourceType.DOCUMENT,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        relevance_score=score
                    )
                )
        
        return retrieval_results
        
    def get_document_chunks(self) -> List[DocumentChunk]:
        """Get all document chunks in the collection"""
        # Get all documents in the collection
        documents = self.collection.get(include=["documents", "metadatas"])
        
        chunks = []
        for i in range(len(documents["ids"])):
            chunks.append(
                DocumentChunk(
                    id=documents["ids"][i],
                    content=documents["documents"][i],
                    metadata=documents["metadatas"][i] if documents["metadatas"] else {}
                )
            )
        
        return chunks