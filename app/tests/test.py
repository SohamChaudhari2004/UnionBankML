import os
import tempfile
from typing import List, Dict, Any

# Text processing and embedding
from sentence_transformers import SentenceTransformer
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
from groq import Groq

# Voice processing (simplified)
import speech_recognition as sr

class LoanChatbot:
    def __init__(self, api_key: str, data_path: str):
        """
        Initialize the RAG-based loan chatbot with conversation memory.
        
        Args:
            api_key: Groq API key
            data_path: Path to the loan data text file
        """
        self.api_key = api_key
        self.data_path = data_path
        
        # Set up embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Set up vector store
        self.db_directory = "loan_chroma_db"
        
        # Initialize Groq client for both chat and transcription
        self.groq_client = Groq(api_key=self.api_key)
        
        # Set up LLM using langchain-groq
        self.llm = ChatGroq(
            api_key=self.api_key,

    
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=2048,
        )
        
        # Initialize speech recognition components (only for microphone capture)
        self.recognizer = sr.Recognizer()
        
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
        if os.path.exists(self.db_directory) and os.listdir(self.db_directory):
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=self.db_directory,
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
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}),
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
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        texts = text_splitter.split_text(loan_data)
        
        # Create Document objects
        documents = [Document(page_content=text) for text in texts]
        
        # Create vector database
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.db_directory
        )
        
        # Persist the database
        self.vectordb.persist()
        
    # def process_voice_input_with_groq(self, audio_file_path=None):

    #     """
    #     Process voice input using Groq's transcription API.
        
    #     Args:
    #         audio_file_path: Path to audio file, if None, captures from microphone
            
    #     Returns:
    #         Transcribed text from the voice input
    #     """
    #     try:
    #         if audio_file_path:
    #             # Process existing audio file with Groq
    #             print(f"Processing audio file {audio_file_path}...")
    #             with open(audio_file_path, "rb") as file:
    #                 transcription = self.groq_client.audio.transcriptions.create(
                        
                        
    #                     file=(audio_file_path, file.read()),
    #                     model="whisper-large-v3-turbo",
    #                     response_format="verbose_json",
    #                 )
    #             transcribed_text = transcription.text
    #         else:
    #             # Use microphone and save to temp file
    #             print("Listening... Speak now.")
    #             temp_audio_path = os.path.join(tempfile.gettempdir(), "audio_input.wav")
                
    #             with sr.Microphone() as source:
    #                 self.recognizer.adjust_for_ambient_noise(source)
    #                 print("Listening...")
    #                 audio = self.recognizer.listen(source)
                
    #             # Save audio to temporary file
    #             with open(temp_audio_path, "wb") as f:
    #                 f.write(audio.get_wav_data())
                
    #             # Process with Groq
    #             with open(temp_audio_path, "rb") as file:
    #                 transcription = self.groq_client.audio.transcriptions.create(
    #                     file=(temp_audio_path, file.read()),
    #                     model="whisper-large-v3-turbo",
    #                     response_format="verbose_json",
    #                 )
                
    #             # Clean up temp file
    #             if os.path.exists(temp_audio_path):
    #                 os.remove(temp_audio_path)
                
    #             transcribed_text = transcription.text
                
    #         print(f"Transcribed: {transcribed_text}")
    #         return transcribed_text
        
    #     except Exception as e:
    #         print(f"Error processing voice input: {e}")
    #         return None


    def process_voice_input_with_groq(self, audio_file_path=None):
        """
        Process voice input using Groq's transcription API.

    Args:
        audio_file_path: Path to audio file, if None, captures from microphone

    Returns:
        Transcribed text from the voice input
    """
        try:
            if audio_file_path:
                # Process existing audio file with Groq
                print(f"Processing audio file {audio_file_path}...")
                with open(audio_file_path, "rb") as file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=(audio_file_path, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )
                transcribed_text = transcription.text
            else:
                # Use microphone and save to temp file
                print("Listening... Speak now.")
                temp_audio_path = os.path.join(tempfile.gettempdir(), "audio_input.wav")

                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = self.recognizer.listen(source)

                # Save audio to temporary file
                with open(temp_audio_path, "wb") as f:
                    f.write(audio.get_wav_data())

                # Process with Groq
                with open(temp_audio_path, "rb") as file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=(temp_audio_path, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )

                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

                transcribed_text = transcription.text

            print(f"Transcribed: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            print(f"Error processing voice input: {e}")
            return None
    
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
    
    def chat_loop(self):
        """Main chat loop for interaction with conversation memory."""
        print("Loan Chatbot initialized. Type 'voice' to use voice input, 'exit' to quit, 'clear' to reset conversation history.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                self.memory.clear()
                print("Chatbot: Conversation history has been cleared.")
                continue
            
            elif user_input.lower() == 'voice':
                print("Chatbot: Listening for voice input...")
                transcribed_text = self.process_voice_input_with_groq()
                if transcribed_text:
                    print(f"You (voice): {transcribed_text}")
                    user_input = transcribed_text
                else:
                    print("Chatbot: Sorry, I couldn't understand the audio.")
                    continue
            
            # Process the query with conversation memory
            result = self.process_query(user_input)
            print(f"\nChatbot: {result['answer']}")
            
            # Optional: Display sources
            # if result['sources']:
            #     print("\nSources:")
            #     for i, source in enumerate(result['sources'], 1):
            #         print(f"{i}. {source[:100]}...")

# Example usage
# Example usage
if __name__ == "__main__":
    # Set your API key and data path
    GROQ_API_KEY = "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"  # Directly set the API key
    LOAN_DATA_PATH = "loan_data.txt"
    
    # Initialize and run the chatbot
    chatbot = LoanChatbot(api_key=GROQ_API_KEY, data_path=LOAN_DATA_PATH)
    chatbot.chat_loop()