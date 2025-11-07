"""
RAG Agent - Retrieval-Augmented Generation for Property Search
Uses FAISS vector store and HuggingFace embeddings for semantic search
"""

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import os
import pickle
import numpy as np
from sqlalchemy.orm import sessionmaker
from models.free_models import get_primary_llm
from models.rate_limiter import rate_limiter
import sys

# Add src directory to path to import database schema
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from database_schema import Property, Certificate, get_session, get_database_url

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    LangChain-based RAG Agent for semantic property search
    Uses FAISS vector store for efficient similarity search
    """
    
    def __init__(self, vector_store_path: str = "data/vector_store"):
        self.llm = get_primary_llm()
        self.vector_store_path = vector_store_path
        self.embeddings_model = None
        self.vector_store = None
        
        # Initialize database connection
        db_url = get_database_url('sqlite')
        self.session = get_session(db_url)
        
        # Initialize embeddings and vector store
        self._initialize_embeddings()
        self._load_or_create_vector_store()
        
        # Create LangChain tools
        self.semantic_search_tool = Tool(
            name="semantic_search",
            description="Perform semantic search on property descriptions and documents",
            func=self.semantic_search_tool_func
        )
        
        self.similarity_search_tool = Tool(
            name="similarity_search",
            description="Find similar properties based on description similarity",
            func=self.similarity_search_tool_func
        )
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        try:
            # Use a lightweight, fast model for embeddings
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one from database"""
        vector_store_file = f"{self.vector_store_path}/faiss_index"
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        if os.path.exists(f"{vector_store_file}.faiss") and os.path.exists(f"{vector_store_file}.pkl"):
            try:
                # Load existing vector store with safe deserialization
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings_model,
                    index_name="faiss_index",
                    allow_dangerous_deserialization=True  # Safe since we created this file
                )
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}. Creating new one...")
                self._create_vector_store()
        else:
            logger.info("No existing vector store found. Creating new one...")
            self._create_vector_store()
    
    def _create_vector_store(self):
        """Create vector store from property data in database"""
        try:
            # Get all properties from database
            properties = self.session.query(Property).all()
            
            if not properties:
                logger.warning("No properties found in database")
                return
            
            logger.info(f"Creating vector store from {len(properties)} properties...")
            
            # Convert properties to documents
            documents = []
            for prop in properties:
                # Combine all text fields for better semantic search
                content = f"""
                Title: {prop.title}
                Location: {prop.location}
                City: {prop.city}
                Description: {prop.long_description}
                Rooms: {prop.num_rooms}
                Size: {prop.property_size_sqft} sqft
                Price: ₹{prop.price:,}
                Seller: {prop.seller_type}
                Tags: {prop.metadata_tags or ''}
                Certificates: {prop.certificates or ''}
                """.strip()
                
                # Create metadata with searchable fields
                metadata = {
                    "property_id": prop.property_id,
                    "id": prop.id,
                    "title": prop.title,
                    "location": prop.location,
                    "city": prop.city,
                    "price": prop.price,
                    "num_rooms": prop.num_rooms,
                    "property_size_sqft": prop.property_size_sqft,
                    "seller_type": prop.seller_type,
                    "listing_date": prop.listing_date.isoformat() if prop.listing_date else None,
                    "metadata_tags": prop.metadata_tags or '',
                    "certificates": prop.certificates or ''
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings_model
            )
            
            # Save vector store
            self.vector_store.save_local(self.vector_store_path, index_name="faiss_index")
            logger.info(f"Created and saved vector store with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def semantic_search_tool_func(self, query: str) -> str:
        """LangChain tool wrapper for semantic search"""
        result = self.semantic_search(json.loads(query) if query.startswith('{') else {"query": query})
        return json.dumps(result)
    
    def similarity_search_tool_func(self, query: str) -> str:
        """LangChain tool wrapper for similarity search"""
        result = self.find_similar_properties(query)
        return json.dumps(result)
    
    def semantic_search(self, search_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform semantic search on property descriptions
        """
        try:
            query_text = search_criteria.get("query", "")
            top_k = search_criteria.get("limit", 10)
            
            if not self.vector_store:
                return {
                    "success": False,
                    "error": "Vector store not initialized",
                    "properties": [],
                    "count": 0,
                    "agent": "RAGAgent"
                }
            
            # Perform similarity search
            similar_docs = self.vector_store.similarity_search_with_score(
                query_text, 
                k=top_k
            )
            
            # Format results
            properties = []
            for doc, score in similar_docs:
                prop_data = {
                    "similarity_score": float(score),
                    "relevance": self._calculate_relevance(score),
                    **doc.metadata
                }
                properties.append(prop_data)
            
            # Generate AI-powered summary if requested
            summary = None
            if search_criteria.get("include_summary", False):
                summary = self._generate_search_summary(query_text, properties)
            
            return {
                "success": True,
                "properties": properties,
                "count": len(properties),
                "query": query_text,
                "search_type": "semantic",
                "summary": summary,
                "agent": "RAGAgent"
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "properties": [],
                "count": 0,
                "agent": "RAGAgent"
            }
    
    def find_similar_properties(self, property_description: str, limit: int = 5) -> Dict[str, Any]:
        """
        Find properties similar to a given description
        """
        try:
            if not self.vector_store:
                return {
                    "success": False,
                    "error": "Vector store not initialized",
                    "similar_properties": [],
                    "count": 0
                }
            
            # Search for similar properties
            similar_docs = self.vector_store.similarity_search_with_score(
                property_description, 
                k=limit
            )
            
            # Format results
            similar_properties = []
            for doc, score in similar_docs:
                prop_data = {
                    "similarity_score": float(score),
                    "relevance": self._calculate_relevance(score),
                    "title": doc.metadata.get("title", ""),
                    "location": doc.metadata.get("location", ""),
                    "price": doc.metadata.get("price", 0),
                    "num_rooms": doc.metadata.get("num_rooms", 0),
                    "property_size_sqft": doc.metadata.get("property_size_sqft", 0),
                    "property_id": doc.metadata.get("property_id", ""),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                similar_properties.append(prop_data)
            
            return {
                "success": True,
                "similar_properties": similar_properties,
                "count": len(similar_properties),
                "query": property_description
            }
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "similar_properties": [],
                "count": 0
            }
    
    def _calculate_relevance(self, score: float) -> str:
        """Calculate relevance level based on similarity score"""
        # FAISS uses L2 distance, lower is better
        if score < 0.5:
            return "Very High"
        elif score < 1.0:
            return "High"
        elif score < 1.5:
            return "Medium"
        elif score < 2.0:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_search_summary(self, query: str, properties: List[Dict]) -> str:
        """Generate AI-powered search summary using LLM"""
        try:
            if not rate_limiter.can_make_request("groq"):
                return "Summary unavailable due to rate limiting"
            
            # Prepare property summaries
            prop_summaries = []
            for prop in properties[:5]:  # Limit to top 5 for summary
                summary = f"- {prop['title']} in {prop.get('location', 'Unknown')} - ₹{prop.get('price', 0):,} ({prop.get('property_size_sqft', 0)} sqft)"
                prop_summaries.append(summary)
            
            prompt = f"""
            Analyze this real estate search and provide a helpful summary.
            
            User Query: "{query}"
            
            Found Properties:
            {chr(10).join(prop_summaries)}
            
            Provide a brief, helpful summary (2-3 sentences) that:
            1. Summarizes what was found
            2. Highlights key patterns or insights
            3. Suggests next steps if appropriate
            
            Keep it concise and user-friendly.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rate_limiter.record_request("groq")
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Unable to generate summary at this time"
    
    def get_property_recommendations(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get property recommendations based on user preferences
        """
        try:
            # Build preference-based query
            preference_parts = []
            
            if user_preferences.get("property_type"):
                preference_parts.append(f"{user_preferences['property_type']} property")
            
            if user_preferences.get("location"):
                preference_parts.append(f"in {user_preferences['location']}")
            
            if user_preferences.get("budget"):
                budget = user_preferences['budget']
                if budget > 10000000:  # 1 crore+
                    preference_parts.append("luxury high-end premium")
                elif budget > 5000000:  # 50 lakh+
                    preference_parts.append("premium mid-range")
                else:
                    preference_parts.append("affordable budget-friendly")
            
            if user_preferences.get("amenities"):
                amenities = " ".join(user_preferences["amenities"])
                preference_parts.append(f"with {amenities}")
            
            # Create semantic query
            query = " ".join(preference_parts)
            
            # Perform semantic search
            results = self.semantic_search({
                "query": query,
                "limit": 8,
                "include_summary": True
            })
            
            if results["success"]:
                results["recommendation_type"] = "preference_based"
                results["user_preferences"] = user_preferences
            
            return results
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "properties": [],
                "count": 0
            }
    
    def update_vector_store(self):
        """Update vector store with latest property data"""
        try:
            logger.info("Updating vector store with latest data...")
            self._create_vector_store()
            return {"success": True, "message": "Vector store updated successfully"}
        except Exception as e:
            logger.error(f"Failed to update vector store: {e}")
            return {"success": False, "error": str(e)}
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            # Get total number of vectors
            total_vectors = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            
            # Get dimension of vectors
            dimension = self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else 0
            
            return {
                "total_documents": total_vectors,
                "vector_dimension": dimension,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_type": "FAISS",
                "storage_path": self.vector_store_path
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Test the RAG agent
    rag_agent = RAGAgent()
    
    test_queries = [
        "luxury villa with swimming pool",
        "affordable apartment in Hyderabad",
        "spacious house with garden",
        "modern studio apartment",
        "property with good connectivity"
    ]
    
    print("🔍 Testing RAG Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        result = rag_agent.semantic_search({
            "query": query,
            "limit": 3,
            "include_summary": True
        })
        
        if result["success"]:
            print(f"✅ Found {result['count']} properties")
            for i, prop in enumerate(result["properties"][:2], 1):
                print(f"   {i}. {prop['title']} - Relevance: {prop['relevance']}")
            
            if result.get("summary"):
                print(f"📝 Summary: {result['summary']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 30)