# Banking Chatbot with RAG and Active Retrieval
```
banking_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py              # Chat API endpoints
│   │   └── audio.py             # Audio processing endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration settings
│   │   ├── chat_manager.py      # Chat session management
│   │   └── data_loader.py       # Load documents to vector DB
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio.py             # Audio conversion models
│   │   ├── chat.py              # Chat models
│   │   └── retrieval.py         # Retrieval models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py     # Whisper integration
│   │   ├── llm.py               # LLaMA integration
│   │   ├── retrieval.py         # RAG system
│   │   ├── web_search.py        # Active retrieval
│   │   └── chat_service.py      # Main chat service
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Helper functions
├── data/
│   ├── banking_docs.txt         # Banking knowledge base
│   └── vector_db/              # Directory for ChromaDB
├── .env                        # Environment variables
└── requirements.txt            # Project dependencies
```