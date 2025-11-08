"""
Missing Architecture Components Analysis
=======================================

Based on the architecture diagram, here are the missing components we need to implement:

1. API GATEWAY - Entry point for all requests
2. SESSION MANAGER - Manage user sessions across requests
3. USER INTERFACE - Frontend/API interface layer
4. RESPONSE FORMATTER - Standardized response formatting

Let's implement these missing pieces:
"""

# ========================================
# 1. API GATEWAY IMPLEMENTATION
# ========================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
import logging
from datetime import datetime

# Import our existing agents
import sys
import os
sys.path.append(os.path.dirname(__file__))
from agents.langgraph_orchestrator import LangGraphRealEstateOrchestrator
from agents.memory_enhanced_planner import MemoryEnhancedPlannerAgent
from components.memory_component import MemoryComponent

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    success: bool
    response: str
    session_id: str
    agent_execution_details: Dict[str, Any]
    suggestions: List[str] = []
    timestamp: datetime

class APIGateway:
    """
    Main API Gateway - Entry point for all Real Estate Search requests
    Handles authentication, session management, and request routing
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Real Estate Search Engine API",
            description="Intelligent Multi-Agent Real Estate Search System",
            version="1.0.0"
        )
        self.session_manager = SessionManager()
        self.orchestrator = LangGraphRealEstateOrchestrator()
        self.memory_planner = MemoryEnhancedPlannerAgent()
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on your needs
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v1/search", response_model=QueryResponse)
        async def search_properties(request: QueryRequest, background_tasks: BackgroundTasks):
            """Main property search endpoint"""
            try:
                # Step 1: Session Management
                session_id = await self.session_manager.get_or_create_session(
                    request.user_id, request.session_id
                )
                
                # Step 2: Query Processing with Memory
                result = await self.memory_planner.plan_and_execute(
                    query=request.query,
                    user_id=request.user_id or "anonymous",
                    preferences=request.preferences
                )
                
                # Step 3: Response Formatting
                formatted_response = ResponseFormatter.format_search_response(result)
                
                # Step 4: Background tasks (analytics, cleanup)
                background_tasks.add_task(
                    self._log_request, request, result, session_id
                )
                
                return QueryResponse(
                    success=True,
                    response=formatted_response["response"],
                    session_id=session_id,
                    agent_execution_details=formatted_response["execution_details"],
                    suggestions=formatted_response.get("suggestions", []),
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Search request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/session/{session_id}/history")
        async def get_session_history(session_id: str):
            """Get conversation history for a session"""
            history = await self.session_manager.get_session_history(session_id)
            return {"session_id": session_id, "history": history}
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "agents": await self._check_agent_health(),
                "timestamp": datetime.now()
            }


# ========================================
# 2. SESSION MANAGER IMPLEMENTATION
# ========================================

import redis
import json
from typing import Optional

class SessionManager:
    """
    Manages user sessions, conversation context, and state persistence
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.use_redis = True
        except:
            # Fallback to in-memory storage
            self.memory_store = {}
            self.use_redis = False
            logger.warning("Redis unavailable, using in-memory session storage")
    
    async def get_or_create_session(
        self, 
        user_id: Optional[str], 
        session_id: Optional[str]
    ) -> str:
        """Get existing session or create new one"""
        
        if session_id:
            # Validate existing session
            if await self.session_exists(session_id):
                return session_id
        
        # Create new session
        new_session_id = self._generate_session_id(user_id)
        await self.create_session(new_session_id, user_id)
        
        return new_session_id
    
    async def create_session(self, session_id: str, user_id: Optional[str]):
        """Create new session with initial context"""
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "conversation_history": [],
            "user_preferences": {},
            "search_context": {}
        }
        
        await self._store_session(session_id, session_data)
    
    async def update_session(
        self, 
        session_id: str, 
        query: str, 
        response: str,
        context: Dict[str, Any] = None
    ):
        """Update session with new interaction"""
        session_data = await self._get_session(session_id)
        
        if session_data:
            session_data["last_activity"] = datetime.now().isoformat()
            session_data["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "context": context or {}
            })
            
            # Keep last 50 interactions
            session_data["conversation_history"] = session_data["conversation_history"][-50:]
            
            await self._store_session(session_id, session_data)
    
    def _generate_session_id(self, user_id: Optional[str]) -> str:
        """Generate unique session ID"""
        prefix = user_id[:8] if user_id else "anon"
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
    
    async def _store_session(self, session_id: str, data: Dict):
        """Store session data"""
        if self.use_redis:
            await self.redis_client.setex(
                f"session:{session_id}", 
                86400,  # 24 hours
                json.dumps(data)
            )
        else:
            self.memory_store[session_id] = data
    
    async def _get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data"""
        if self.use_redis:
            data = await self.redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else None
        else:
            return self.memory_store.get(session_id)


# ========================================
# 3. RESPONSE FORMATTER IMPLEMENTATION
# ========================================

class ResponseFormatter:
    """
    Standardizes response formatting across all agents
    """
    
    @staticmethod
    def format_search_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results into standardized response"""
        
        if not result.get("success"):
            return ResponseFormatter._format_error_response(result)
        
        # Extract properties from different agents
        properties = ResponseFormatter._extract_properties(result)
        
        # Generate user-friendly response
        response_text = ResponseFormatter._generate_response_text(
            properties, result.get("query", "")
        )
        
        return {
            "response": response_text,
            "execution_details": {
                "agents_used": result.get("active_agents", []),
                "execution_time": result.get("execution_time", 0),
                "confidence_score": result.get("confidence_score", 0.5),
                "properties_found": len(properties),
                "fallback_used": result.get("fallback_info", {}).get("is_fallback", False)
            },
            "properties": properties,
            "suggestions": ResponseFormatter._generate_suggestions(result)
        }
    
    @staticmethod
    def _extract_properties(result: Dict) -> List[Dict]:
        """Extract properties from agent results"""
        properties = []
        
        # From structured data agent
        if "structured_data" in result.get("agent_results", {}):
            structured_props = result["agent_results"]["structured_data"].get("properties", [])
            properties.extend(structured_props)
        
        # From RAG agent
        if "rag" in result.get("agent_results", {}):
            rag_props = result["agent_results"]["rag"].get("properties", [])
            properties.extend(rag_props)
        
        # Remove duplicates based on property_id
        seen_ids = set()
        unique_properties = []
        for prop in properties:
            prop_id = prop.get("property_id") or prop.get("id")
            if prop_id not in seen_ids:
                seen_ids.add(prop_id)
                unique_properties.append(prop)
        
        return unique_properties[:20]  # Limit to 20 results
    
    @staticmethod
    def _generate_response_text(properties: List[Dict], query: str) -> str:
        """Generate natural language response"""
        if not properties:
            return f"I couldn't find properties matching '{query}'. Try broadening your search criteria or check out our suggestions below."
        
        count = len(properties)
        response = f"I found {count} propert{'y' if count == 1 else 'ies'} matching your criteria:\n\n"
        
        for i, prop in enumerate(properties[:5], 1):
            title = prop.get("title", "Property")
            price = prop.get("price", 0)
            location = prop.get("location", "Location not specified")
            
            response += f"{i}. **{title}**\n"
            response += f"   📍 Location: {location}\n"
            response += f"   💰 Price: ₹{price:,}\n"
            
            if "property_size_sqft" in prop:
                response += f"   📐 Size: {prop['property_size_sqft']} sqft\n"
            
            response += "\n"
        
        if count > 5:
            response += f"... and {count - 5} more properties available.\n\n"
        
        return response
    
    @staticmethod
    def _generate_suggestions(result: Dict) -> List[str]:
        """Generate helpful suggestions based on result"""
        suggestions = []
        
        properties_count = len(ResponseFormatter._extract_properties(result))
        
        if properties_count == 0:
            suggestions = [
                "Try expanding your budget range",
                "Consider nearby locations",
                "Look for different property types",
                "Check properties with flexible configurations"
            ]
        elif properties_count < 5:
            suggestions = [
                "Try nearby areas for more options",
                "Consider adjusting your budget",
                "Look for similar property types"
            ]
        
        return suggestions


# ========================================
# 4. MAIN APPLICATION ENTRY POINT
# ========================================

def create_app():
    """Create and configure the FastAPI application"""
    gateway = APIGateway()
    return gateway.app

# For running the server
if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    print("🏠🚀 Real Estate Search Engine API Gateway Starting...")
    print("=" * 60)
    print("📋 Available Endpoints:")
    print("  POST /api/v1/search - Main property search")
    print("  GET  /api/v1/session/{id}/history - Session history")
    print("  GET  /api/v1/health - Health check")
    print("=" * 60)
    print("🔗 Documentation: http://localhost:8000/docs")
    print("🔗 API: http://localhost:8000/api/v1/")
    
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )