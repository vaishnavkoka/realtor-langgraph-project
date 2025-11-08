"""
Free LLM and Embedding Model Configuration for LangChain
"""

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# Updated import for modern LangChain
try:
    from langchain_cohere import CohereEmbeddings
except ImportError:
    # Fallback for older versions
    from langchain_community.embeddings import CohereEmbeddings
from config.settings import APIKeys, ModelConfig
import logging

logger = logging.getLogger(__name__)

class FreeLLMProvider:
    """Manages free LLM models with fallbacks"""
    
    def __init__(self):
        self._primary_llm = None
        self._advanced_llm = None
        self._context_llm = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Groq models"""
        try:
            if APIKeys.GROQ_API_KEY:
                self._primary_llm = ChatGroq(
                    groq_api_key=APIKeys.GROQ_API_KEY,
                    model_name=ModelConfig.PRIMARY_LLM,
                    temperature=0.1,
                    max_tokens=1024
                )
                
                self._advanced_llm = ChatGroq(
                    groq_api_key=APIKeys.GROQ_API_KEY,
                    model_name=ModelConfig.ADVANCED_LLM,
                    temperature=0.3,
                    max_tokens=2048
                )
                
                self._context_llm = ChatGroq(
                    groq_api_key=APIKeys.GROQ_API_KEY,
                    model_name=ModelConfig.CONTEXT_LLM,
                    temperature=0.2,
                    max_tokens=4096
                )
                
                logger.info("✅ Groq models initialized successfully")
            else:
                logger.warning("⚠️ GROQ_API_KEY not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq models: {e}")
    
    def get_primary_llm(self):
        """Get fast LLM for routing and basic tasks"""
        return self._primary_llm
    
    def get_advanced_llm(self):
        """Get advanced LLM for complex reasoning"""
        return self._advanced_llm
    
    def get_context_llm(self):
        """Get LLM with long context for RAG"""
        return self._context_llm

class FreeEmbeddingProvider:
    """Manages free embedding models with fallbacks"""
    
    def __init__(self):
        self._primary_embeddings = None
        self._backup_embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace and Cohere embeddings"""
        try:
            # Primary: HuggingFace (completely free)
            self._primary_embeddings = HuggingFaceEmbeddings(
                model_name=ModelConfig.PRIMARY_EMBEDDING,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ HuggingFace embeddings initialized")
            
            # Backup: Cohere (free tier)
            if APIKeys.COHERE_API_KEY:
                self._backup_embeddings = CohereEmbeddings(
                    cohere_api_key=APIKeys.COHERE_API_KEY,
                    model="embed-english-v3.0"
                )
                logger.info("✅ Cohere embeddings initialized as backup")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
    
    def get_primary_embeddings(self):
        """Get primary embedding model"""
        return self._primary_embeddings
    
    def get_backup_embeddings(self):
        """Get backup embedding model"""
        return self._backup_embeddings

# Global instances
llm_provider = FreeLLMProvider()
embedding_provider = FreeEmbeddingProvider()

# Export functions for easy access
def get_primary_llm():
    return llm_provider.get_primary_llm()

def get_advanced_llm():
    return llm_provider.get_advanced_llm()

def get_context_llm():
    return llm_provider.get_context_llm()

def get_primary_embeddings():
    return embedding_provider.get_primary_embeddings()

def get_backup_embeddings():
    return embedding_provider.get_backup_embeddings()