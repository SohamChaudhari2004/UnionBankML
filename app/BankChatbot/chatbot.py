"""Loan Chatbot implementation."""

import os
from typing import Dict, Any, List

# Text processing and embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Vector storage
import chromadb
from langchain.vectorstores import Chroma

# LLM integration
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Groq client
from groq import Groq

# Import app config
from .config import (
    EMBEDDING_MODEL, LLM_MODEL, DB_DIRECTORY, 
    TEMPERATURE, MAX_TOKENS, CHUNK_SIZE, 
    CHUNK_OVERLAP, RETRIEVAL_K
)

class LoanChatbot:
    """RAG-based loan chatbot with conversation memory."""
    
    def __init__(self, api_key: str, data_path: str, session_id: str = "default"):
        """
        Initialize the RAG-based loan chatbot with conversation memory.
        
        Args:
            api_key: Groq API key
            data_path: Path to the loan data text file
            session_id: Unique session identifier for conversation memory
        """
        self.api_key = api_key
        self.data_path = data_path
        self.session_id = session_id
        
        # Set up embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.api_key)
        
        # Set up LLM using langchain-groq
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize the vector database and QA chain
        self.initialize_system()
    
    def initialize_system(self):
        """Load data, create embeddings, and set up the conversational chain."""
        # Check if DB already exists to avoid rebuilding
        if os.path.exists(DB_DIRECTORY) and os.listdir(DB_DIRECTORY):
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=self.embedding_model
            )
        else:
            print("Creating new vector database from loan data...")
            self._create_vector_database()
        
        # Set up the conversational QA chain with customized prompt
        condense_prompt_template = """
        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question 
        that includes relevant context from the conversation history.
        
        Chat History:
        {chat_history}
        
        Follow-up Question: {question}
        
        Standalone Question:
        """
        
        CONDENSE_QUESTION_PROMPT = PromptTemplate(
            template=condense_prompt_template, 
            input_variables=["chat_history", "question"]
        )
        
        qa_prompt_template = """
        You are a helpful loan assistant with access to loan-related information. 
        Answer the user's question based on the provided context about loans and the conversation history.
        
        Context: {context}
        
        Question: {question}
        
        If the information is not in the context, please say "I don't have enough information about that in my database" instead of making up an answer.
        
        Answer:
        """
        
        QA_PROMPT = PromptTemplate(
            template=qa_prompt_template, 
            input_variables=["context", "question"]
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        print("RAG system with conversation memory initialized successfully!")
    
    def _create_vector_database(self):
        """Load and process the loan data to create embeddings database."""
        # Load loan data from text file
        with open(self.data_path, 'r', encoding='utf-8') as file:
            loan_data = file.read()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        texts = text_splitter.split_text(loan_data)
        
        # Create Document objects
        documents = [Document(page_content=text) for text in texts]
        
        # Create vector database
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=DB_DIRECTORY
        )
        
        # Persist the database
        self.vectordb.persist()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a text query through the conversational RAG system.
        
        Args:
            query: The user's question
            
        Returns:
            Dict containing the answer and source documents
        """
        try:
            result = self.conversation_chain({"question": query})
            return {
                "answer": result["answer"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            print(f"Error processing query: {e}")
            return {"answer": f"Sorry, I encountered an error processing your question: {str(e)}", "sources": []}
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        return {"status": "success", "message": "Conversation history has been cleared."}