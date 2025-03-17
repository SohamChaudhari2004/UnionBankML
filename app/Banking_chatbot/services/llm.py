from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from core.config import settings


class LLMService:
    def __init__(self):
        self.chat_model = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.LLAMA_MODEL_NAME,
            temperature=0.7,
            max_tokens=2048
        )
        
    def get_system_prompt(self, with_rag=True):
        """Get the system prompt for the chat model"""
        base_prompt = """You are a helpful and knowledgeable banking assistant. 
        You help customers with their banking-related queries in a clear, accurate, and concise manner.
        Always be polite and professional, and prioritize giving accurate information.
        
        When you don't know the answer, openly admit it rather than making up information.
        
        Your responses should be:
        - Accurate: Based on factual banking information
        - Clear: Easy to understand, even for customers unfamiliar with banking terminology
        - Concise: Direct and to the point, while being comprehensive
        - Helpful: Providing actionable advice when appropriate
        
        Banking areas you can assist with include accounts, loans, mortgages, investments, 
        credit cards, online banking, transfers, and general financial advice.
        """
        
        if with_rag:
            base_prompt += """
            
            For each user query, I will provide you with relevant retrieved context. 
            Use this context to inform your responses.
            If the retrieved information doesn't answer the question, say so clearly
            rather than trying to guess.
            
            Retrieved context format:
            [Retrieved passages will appear here]
            
            Answer based on the retrieved context and your knowledge of banking.
            """
            
        return base_prompt
        
    async def generate_response(self, query, retrieved_context=None, chat_history=None):
        """
        Generate a response using the LLama model
        
        Args:
            query: User query text
            retrieved_context: Optional context from RAG system
            chat_history: Optional list of previous chat messages
            
        Returns:
            Generated response text
        """
        # Prepare the messages list
        messages = []
        
        # Add system prompt
        system_prompt = self.get_system_prompt(with_rag=retrieved_context is not None)
        messages.append(SystemMessage(content=system_prompt))
        
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
        
        # Prepare user query with context if available
        if retrieved_context:
            context_str = "\n\n".join([f"Source {i+1}:\n{item.content}" 
                                     for i, item in enumerate(retrieved_context)])
            user_message = f"""
            Retrieved context:
            {context_str}
            
            User query: {query}
            
            Please answer the user query based on the retrieved context and your knowledge of banking.
            """
        else:
            user_message = query
            
        messages.append(HumanMessage(content=user_message))
        
        # Generate response
        response = await self.chat_model.ainvoke(messages)
        
        return response.content