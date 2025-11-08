"""
Configuration for Free API Keys and Models
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Free API Keys Configuration
class APIKeys:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Model Configuration
class ModelConfig:
    # Primary LLM Models (Groq - Free)
    PRIMARY_LLM = "llama-3.1-8b-instant"  # Fast for routing, basic tasks
    ADVANCED_LLM = "llama-3.1-70b-versatile"  # Complex reasoning
    CONTEXT_LLM = "mixtral-8x7b-32768"  # Long context for RAG
    
    # Embedding Models (Free)
    PRIMARY_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace
    BACKUP_EMBEDDING = "cohere/embed-english-v3.0"  # Cohere
    
    # Search APIs
    PRIMARY_SEARCH = "serper"  # 2,500 free searches/month
    BACKUP_SEARCH = "tavily"   # 1,000 free searches/month

# Database Configuration  
class DatabaseConfig:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/realestate")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Vector Store Configuration
class VectorConfig:
    FAISS_INDEX_PATH = "data/faiss_index"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5

# Rate Limiting Configuration
class RateLimits:
    GROQ_DAILY_LIMIT = 25000  # Conservative estimate
    SERPER_MONTHLY_LIMIT = 2400  # Leave buffer
    TAVILY_MONTHLY_LIMIT = 950   # Leave buffer
    HF_MONTHLY_LIMIT = 950       # Leave buffer