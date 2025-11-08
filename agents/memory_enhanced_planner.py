"""
Memory-Enhanced Planner Agent
Integrates the Memory Component with the existing Planner Agent for personalized, context-aware interactions

Features:
1. Conversation memory integration
2. Personalized property recommendations
3. Context-aware query processing
4. User preference learning
5. Adaptive response generation
"""

import json
import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
import time
import asyncio
import sys
import os

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# LangGraph imports
from langgraph.graph import StateGraph, END
try:
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
except ImportError:
    # Fallback for older versions
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    def add_messages(existing_messages, new_messages):
        if existing_messages is None:
            existing_messages = []
        if isinstance(new_messages, list):
            return existing_messages + new_messages
        else:
            return existing_messages + [new_messages]

# Agent imports
from agents.query_router import QueryRouterAgent
from agents.structured_data_agent import StructuredDataAgent 
from agents.rag_agent import RAGAgent
from agents.web_research_agent import WebResearchAgent

# Memory component import
from components.memory_component import MemoryComponent

# Rate limiter import
from utils.rate_limiter import groq_rate_limiter, rate_limited, log_rate_limit_stats

# LLM imports - with fallback
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(current_dir), 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    from models.llm_models import get_llm
    from models.rate_limiter import RateLimiter
except ImportError:
    # Fallback implementations
    from langchain_groq import ChatGroq
    import os
    
    def get_llm():
        """Fallback LLM implementation"""
        return ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY', 'your-groq-api-key-here')
        )
    
    class RateLimiter:
        """Fallback rate limiter"""
        def can_make_request(self, service):
            return True
        def record_request(self, service):
            pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryEnhancedPlannerState(TypedDict):
    """Enhanced state with memory integration"""
    messages: List[BaseMessage]
    query: str
    user_id: str
    session_id: str
    user_intent: str
    execution_plan: List[Dict[str, Any]]
    agent_results: Dict[str, Any]
    memory_context: Dict[str, Any]
    personalized_results: List[Dict[str, Any]]
    final_response: str
    processing_metadata: Dict[str, Any]


class MemoryEnhancedPlannerAgent:
    """
    Enhanced Planner Agent with Memory Component integration for personalized experiences
    """
    
    def __init__(self):
        logger.info("🧠💫 Initializing Memory-Enhanced Planner Agent...")
        
        # Initialize all sub-agents
        self.query_router = QueryRouterAgent()
        self.structured_agent = StructuredDataAgent() 
        self.rag_agent = RAGAgent()
        self.web_agent = WebResearchAgent()
        
        # Initialize Memory Component
        self.memory = MemoryComponent()
        
        # Initialize LLM and rate limiter
        self.llm = get_llm()
        self.rate_limiter = RateLimiter()
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        logger.info("✅ Memory-Enhanced Planner Agent initialized successfully!")

    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with memory integration"""
        
        workflow = StateGraph(MemoryEnhancedPlannerState)
        
        # Define workflow nodes
        workflow.add_node("load_memory_context", self._load_memory_context)
        workflow.add_node("analyze_query_with_context", self._analyze_query_with_context)
        workflow.add_node("create_personalized_plan", self._create_personalized_plan)
        workflow.add_node("execute_database_search", self._execute_database_search)
        workflow.add_node("execute_semantic_search", self._execute_semantic_search)
        workflow.add_node("execute_web_research", self._execute_web_research)
        workflow.add_node("personalize_results", self._personalize_results)
        workflow.add_node("synthesize_with_memory", self._synthesize_with_memory)
        workflow.add_node("generate_contextual_response", self._generate_contextual_response)
        workflow.add_node("update_memory", self._update_memory)
        
        # Define workflow edges
        workflow.set_entry_point("load_memory_context")
        
        workflow.add_edge("load_memory_context", "analyze_query_with_context")
        workflow.add_edge("analyze_query_with_context", "create_personalized_plan")
        workflow.add_edge("create_personalized_plan", "execute_database_search")
        workflow.add_edge("execute_database_search", "execute_semantic_search")
        workflow.add_edge("execute_semantic_search", "execute_web_research")
        workflow.add_edge("execute_web_research", "personalize_results")
        workflow.add_edge("personalize_results", "synthesize_with_memory")
        workflow.add_edge("synthesize_with_memory", "generate_contextual_response")
        workflow.add_edge("generate_contextual_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow

    async def plan_and_execute_with_memory(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Main entry point for memory-enhanced planning and execution
        """
        logger.info(f"🧠🎯 Planning execution with memory for user {user_id}: '{query}'")
        
        start_time = time.time()
        
        # Create or retrieve session
        session_id = self.memory.create_user_session(user_id)
        
        # Initialize enhanced state
        initial_state = MemoryEnhancedPlannerState(
            messages=[HumanMessage(content=query)],
            query=query,
            user_id=user_id,
            session_id=session_id,
            user_intent="",
            execution_plan=[],
            agent_results={},
            memory_context={},
            personalized_results=[],
            final_response="",
            processing_metadata={
                "start_time": start_time,
                "agents_used": [],
                "execution_steps": [],
                "user_id": user_id,
                "session_id": session_id
            }
        )
        
        try:
            # Execute enhanced workflow
            result = await self.app.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result["processing_metadata"]["processing_time"] = processing_time
            result["processing_metadata"]["end_time"] = time.time()
            
            logger.info(f"⚡ Memory-enhanced query execution completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in memory-enhanced plan execution: {e}")
            return {
                "error": str(e),
                "query": query,
                "user_id": user_id,
                "processing_metadata": {
                    "start_time": start_time,
                    "processing_time": time.time() - start_time,
                    "error": True
                }
            }

    def _load_memory_context(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 1: Load user's memory context"""
        logger.info("🧠📚 Step 1: Loading user memory context...")
        
        user_id = state["user_id"]
        memory_context = self.memory.get_memory_context(user_id)
        
        state["memory_context"] = memory_context
        state["processing_metadata"]["execution_steps"].append("load_memory_context")
        
        # Log memory insights
        recent_queries = len(memory_context.get("recent_queries", []))
        preferences = len(memory_context.get("preferred_locations", [])) + len(memory_context.get("budget_preferences", []))
        
        logger.info(f"   📝 Loaded {recent_queries} recent queries")
        logger.info(f"   🎯 Found {preferences} learned preferences")
        
        return state

    def _analyze_query_with_context(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 2: Analyze query with memory context"""
        logger.info("🔍🧠 Step 2: Analyzing query with memory context...")
        
        query = state["query"]
        memory_context = state["memory_context"]
        
        # Get enhanced routing with memory context
        routing_result = self.query_router.route_query(query)
        
        # Enhance with memory insights
        if memory_context.get("preferred_locations"):
            if not routing_result["extracted_entities"].get("location"):
                # Suggest location based on history
                preferred_loc = memory_context["preferred_locations"][0]
                routing_result["extracted_entities"]["suggested_location"] = preferred_loc
                logger.info(f"   🏠 Suggested location from memory: {preferred_loc}")
        
        if memory_context.get("budget_preferences"):
            if not routing_result["extracted_entities"].get("max_price"):
                # Suggest budget based on history
                budget = memory_context["budget_preferences"][0]
                routing_result["extracted_entities"]["suggested_budget"] = budget
                logger.info(f"   💰 Suggested budget from memory: ₹{budget:,}")
        
        state["user_intent"] = routing_result["intent"]
        state["processing_metadata"]["query_analysis"] = routing_result
        state["processing_metadata"]["execution_steps"].append("analyze_query_with_context")
        
        logger.info(f"   📋 Intent: {routing_result['intent']} (context-enhanced)")
        
        return state

    def _create_personalized_plan(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 3: Create personalized execution plan"""
        logger.info("📋🎯 Step 3: Creating personalized execution plan...")
        
        query = state["query"]
        routing_result = state["processing_metadata"]["query_analysis"]
        memory_context = state["memory_context"]
        user_profile = memory_context.get("user_profile", {})
        
        # Create enhanced plan based on user profile
        plan = []
        
        # Database search with preference hints
        plan.append({
            "step": 1,
            "agent": "structured_data",
            "action": "database_search", 
            "parameters": routing_result["extracted_entities"],
            "personalization": {
                "preferred_locations": memory_context.get("preferred_locations", []),
                "budget_hints": memory_context.get("budget_preferences", [])
            },
            "reasoning": "Search database with personalized criteria"
        })
        
        # Semantic search with user preferences
        semantic_limit = 12 if user_profile.get("total_searches", 0) > 10 else 8
        plan.append({
            "step": 2,
            "agent": "rag",
            "action": "semantic_search",
            "parameters": {"query": query, "limit": semantic_limit},
            "personalization": {
                "boost_preferences": True,
                "user_history": memory_context.get("recent_queries", [])
            },
            "reasoning": f"Enhanced semantic search with {semantic_limit} results based on experience"
        })
        
        # Web research based on investment profile
        location = routing_result["extracted_entities"].get("location", "")
        investment_profile = user_profile.get("investment_profile", "moderate")
        
        if location or memory_context.get("preferred_locations"):
            search_location = location or memory_context["preferred_locations"][0]
            
            plan.append({
                "step": 3,
                "agent": "web_research", 
                "action": "market_research",
                "parameters": {
                    "location": search_location,
                    "investment_focus": investment_profile
                },
                "reasoning": f"Market research for {search_location} (profile: {investment_profile})"
            })
            
            plan.append({
                "step": 4,
                "agent": "web_research",
                "action": "area_insights", 
                "parameters": {"location": search_location},
                "reasoning": f"Area insights for {search_location}"
            })
        
        state["execution_plan"] = plan
        state["processing_metadata"]["execution_steps"].append("create_personalized_plan")
        
        logger.info(f"   📝 Created {len(plan)}-step personalized plan")
        for step in plan:
            logger.info(f"      {step['step']}. {step['agent']} - {step['reasoning']}")
        
        return state

    def _execute_database_search(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 4a: Execute database search with memory hints"""
        logger.info("🗃️🧠 Step 4a: Executing memory-enhanced database search...")
        
        try:
            query_info = state["processing_metadata"]["query_analysis"]
            memory_context = state["memory_context"]
            
            # Enhance search parameters with memory
            search_params = {
                "query": state["query"],
                "extracted_entities": query_info["extracted_entities"],
                "memory_hints": {
                    "preferred_locations": memory_context.get("preferred_locations", []),
                    "budget_preferences": memory_context.get("budget_preferences", [])
                }
            }
            
            result = self.structured_agent.search_properties(search_params)
            state["agent_results"]["database"] = result
            state["processing_metadata"]["agents_used"].append("StructuredDataAgent")
            state["processing_metadata"]["execution_steps"].append("execute_database_search")
            
            count = result["count"] if result["success"] else 0
            logger.info(f"   ✅ Memory-enhanced database search: {count} matches found")
            
        except Exception as e:
            logger.error(f"   ❌ Database search failed: {e}")
            state["agent_results"]["database"] = {"success": False, "error": str(e)}
        
        return state

    def _execute_semantic_search(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 4b: Execute semantic search with context"""
        logger.info("🧠🔍 Step 4b: Executing context-aware semantic search...")
        
        try:
            plan_step = next((step for step in state["execution_plan"] if step["agent"] == "rag"), {})
            search_params = plan_step.get("parameters", {"query": state["query"], "limit": 8})
            search_params["include_summary"] = False
            
            result = self.rag_agent.semantic_search(search_params)
            state["agent_results"]["semantic"] = result
            state["processing_metadata"]["agents_used"].append("RAGAgent")
            state["processing_metadata"]["execution_steps"].append("execute_semantic_search")
            
            count = result["count"] if result["success"] else 0
            logger.info(f"   ✅ Context-aware semantic search: {count} matches found")
            
        except Exception as e:
            logger.error(f"   ❌ Semantic search failed: {e}")
            state["agent_results"]["semantic"] = {"success": False, "error": str(e)}
        
        return state

    def _execute_web_research(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 4c: Execute web research with user profile"""
        logger.info("🌐👤 Step 4c: Executing profile-aware web research...")
        
        try:
            query_info = state["processing_metadata"]["query_analysis"]
            memory_context = state["memory_context"]
            user_profile = memory_context.get("user_profile", {})
            
            location = query_info["extracted_entities"].get("location")
            if not location and memory_context.get("preferred_locations"):
                location = memory_context["preferred_locations"][0]
            
            if location:
                # Market research with investment profile
                investment_focus = user_profile.get("investment_profile", "moderate")
                market_params = {
                    "location": location,
                    "property_type": query_info["extracted_entities"].get("property_type", ""),
                    "investment_focus": investment_focus,
                    "time_period": "2024"
                }
                market_result = self.web_agent.research_market_trends(market_params)
                
                # Area insights
                area_params = {"location": location}
                area_result = self.web_agent.get_area_insights(area_params)
                
                state["agent_results"]["web_research"] = {
                    "market_trends": market_result,
                    "area_insights": area_result
                }
                state["processing_metadata"]["agents_used"].append("WebResearchAgent")
                
                logger.info(f"   ✅ Profile-aware research for {location} ({investment_focus} profile)")
            else:
                logger.info("   ⚠️ No location for web research")
                state["agent_results"]["web_research"] = {"message": "No location specified"}
            
            state["processing_metadata"]["execution_steps"].append("execute_web_research")
            
        except Exception as e:
            logger.error(f"   ❌ Web research failed: {e}")
            state["agent_results"]["web_research"] = {"error": str(e)}
        
        return state

    def _personalize_results(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 5: Personalize results using memory"""
        logger.info("🎯🧠 Step 5: Personalizing results with memory insights...")
        
        # Collect all properties from agents
        all_properties = []
        
        # From database
        database_results = state["agent_results"].get("database", {})
        if database_results.get("success") and database_results.get("properties"):
            all_properties.extend(database_results["properties"])
        
        # From semantic search
        semantic_results = state["agent_results"].get("semantic", {})
        if semantic_results.get("success") and semantic_results.get("properties"):
            all_properties.extend(semantic_results["properties"])
        
        # Remove duplicates
        unique_properties = []
        seen_ids = set()
        for prop in all_properties:
            prop_id = prop.get("property_id") or prop.get("id")
            if prop_id and prop_id not in seen_ids:
                seen_ids.add(prop_id)
                unique_properties.append(prop)
        
        # Apply personalization
        user_id = state["user_id"]
        personalized_properties = self.memory.get_personalized_recommendations(user_id, unique_properties)
        
        state["personalized_results"] = personalized_properties
        state["processing_metadata"]["execution_steps"].append("personalize_results")
        
        logger.info(f"   ✨ Personalized {len(personalized_properties)} properties")
        
        # Log top personalized property
        if personalized_properties:
            top_prop = personalized_properties[0]
            score = top_prop.get("personalization_score", 0)
            reasons = top_prop.get("recommendation_reasons", [])
            logger.info(f"   🏆 Top match: {top_prop.get('title', 'Property')} (score: {score:.2f})")
            if reasons:
                logger.info(f"       Reasons: {', '.join(reasons[:2])}")
        
        return state

    def _synthesize_with_memory(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 6: Synthesize results with memory context"""
        logger.info("🔗🧠 Step 6: Synthesizing results with memory context...")
        
        # Enhanced synthesis with memory insights
        database_results = state["agent_results"].get("database", {})
        semantic_results = state["agent_results"].get("semantic", {})
        web_results = state["agent_results"].get("web_research", {})
        personalized_properties = state["personalized_results"]
        memory_context = state["memory_context"]
        
        synthesis = {
            "total_exact_matches": database_results.get("count", 0) if database_results.get("success") else 0,
            "total_semantic_matches": semantic_results.get("count", 0) if semantic_results.get("success") else 0,
            "personalized_matches": len(personalized_properties),
            "has_market_data": web_results.get("market_trends", {}).get("success", False),
            "has_area_insights": web_results.get("area_insights", {}).get("success", False),
            "recommended_properties": personalized_properties[:5],  # Top 5 personalized
            "market_summary": "",
            "memory_insights": {
                "returning_user": len(memory_context.get("recent_queries", [])) > 0,
                "learned_preferences": len(memory_context.get("preferred_locations", [])) + 
                                     len(memory_context.get("budget_preferences", [])),
                "user_experience": memory_context.get("user_profile", {}).get("total_searches", 0),
                "personalization_applied": len(personalized_properties) > 0
            }
        }
        
        # Extract market summary
        if web_results.get("market_trends", {}).get("success"):
            market_data = web_results["market_trends"]
            if market_data.get("market_analysis"):
                synthesis["market_summary"] = market_data["market_analysis"][:500] + "..."
        
        state["agent_results"]["synthesis"] = synthesis
        state["processing_metadata"]["execution_steps"].append("synthesize_with_memory")
        
        memory_insights = synthesis["memory_insights"]
        logger.info(f"   📊 Synthesized with memory: {synthesis['personalized_matches']} personalized matches")
        logger.info(f"   🧠 User experience: {memory_insights['user_experience']} searches, {memory_insights['learned_preferences']} preferences")
        
        return state

    def _generate_contextual_response(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 7: Generate memory-aware response"""
        logger.info("✍️🧠 Step 7: Generating memory-contextual response...")
        
        try:
            query = state["query"]
            synthesis = state["agent_results"].get("synthesis", {})
            memory_context = state["memory_context"]
            
            # Build enhanced prompt with memory context
            prompt = self._build_memory_aware_prompt(query, synthesis, memory_context, state)
            
            # Generate response using rate-limited LLM call
            final_response = self._generate_memory_contextual_response(prompt)
            
            state["final_response"] = final_response
            state["processing_metadata"]["execution_steps"].append("generate_contextual_response")
            
            logger.info("   ✅ Memory-contextual response generated")
            
        except Exception as e:
            logger.error(f"   ❌ Response generation failed: {e}")
            state["final_response"] = self._generate_memory_fallback_response(
                state["query"], 
                state["agent_results"].get("synthesis", {}),
                state["memory_context"]
            )
        
        return state

    def _update_memory(self, state: MemoryEnhancedPlannerState) -> MemoryEnhancedPlannerState:
        """Step 8: Update memory with interaction"""
        logger.info("💾🧠 Step 8: Updating memory with interaction...")
        
        try:
            # Extract interaction details
            user_id = state["user_id"]
            query = state["query"]
            routing_result = state["processing_metadata"]["query_analysis"]
            agents_used = state["processing_metadata"]["agents_used"]
            response = state["final_response"]
            properties_shown = [prop.get("property_id", "") for prop in state["personalized_results"][:5]]
            
            # Update memory
            memory_context = self.memory.process_user_interaction(
                user_id=user_id,
                query=query,
                entities=routing_result["extracted_entities"],
                intent=routing_result["intent"],
                agents_used=agents_used,
                response=response,
                properties_shown=properties_shown
            )
            
            state["processing_metadata"]["updated_memory_context"] = memory_context
            state["processing_metadata"]["execution_steps"].append("update_memory")
            
            logger.info("   ✅ Memory updated with interaction")
            
        except Exception as e:
            logger.error(f"   ❌ Memory update failed: {e}")
        
        return state

    def _build_memory_aware_prompt(self, query: str, synthesis: Dict, memory_context: Dict, state: MemoryEnhancedPlannerState) -> str:
        """Build memory-aware prompt for response generation"""
        
        user_profile = memory_context.get("user_profile", {})
        recent_queries = memory_context.get("recent_queries", [])
        memory_insights = synthesis.get("memory_insights", {})
        
        prompt = f"""
You are an expert real estate advisor with access to user's conversation history and preferences.

USER QUERY: "{query}"

USER CONTEXT:
- Experience Level: {user_profile.get('total_searches', 0)} previous searches
- Investment Profile: {user_profile.get('investment_profile', 'moderate')}
- Recent Interest: {recent_queries[-1] if recent_queries else 'First interaction'}
- Returning User: {'Yes' if memory_insights.get('returning_user') else 'New user'}

SEARCH RESULTS SUMMARY:
- Exact Database Matches: {synthesis.get('total_exact_matches', 0)}
- Semantic Matches: {synthesis.get('total_semantic_matches', 0)}  
- Personalized Recommendations: {synthesis.get('personalized_matches', 0)}
- Market Data Available: {synthesis.get('has_market_data', False)}

PERSONALIZED RECOMMENDATIONS:
{self._format_personalized_properties_for_prompt(synthesis.get('recommended_properties', []))}

MEMORY-BASED INSIGHTS:
- Learned Preferences: {memory_insights.get('learned_preferences', 0)} preferences identified
- Personalization Applied: {'Yes' if memory_insights.get('personalization_applied') else 'No'}

MARKET INSIGHTS:
{synthesis.get('market_summary', 'Limited market data available')}

Please provide a personalized response that:
1. Acknowledges the user's experience level and history (if returning user)
2. Highlights personalized recommendations and why they match their preferences
3. Provides market context relevant to their investment profile
4. Uses appropriate detail level for their experience
5. Suggests next steps based on their search pattern

Format your response professionally with clear sections and personalized insights.
"""
        return prompt

    def _format_personalized_properties_for_prompt(self, properties: List[Dict]) -> str:
        """Format personalized property recommendations for prompt"""
        if not properties:
            return "No personalized recommendations available."
        
        formatted = []
        for i, prop in enumerate(properties[:3], 1):
            title = prop.get('title', 'Property')
            price = prop.get('price', 0)
            location = prop.get('location', 'Location not specified')
            score = prop.get('personalization_score', 0)
            reasons = prop.get('recommendation_reasons', [])
            
            reasons_text = f" (Reasons: {', '.join(reasons[:2])})" if reasons else ""
            formatted.append(f"{i}. {title} - ₹{price:,} in {location} (Match Score: {score:.2f}){reasons_text}")
        
        return "\n".join(formatted)

    def _generate_memory_fallback_response(self, query: str, synthesis: Dict, memory_context: Dict) -> str:
        """Generate fallback response when LLM is unavailable"""
        
        user_profile = memory_context.get("user_profile", {})
        total_searches = user_profile.get('total_searches', 0)
        is_returning = len(memory_context.get("recent_queries", [])) > 0
        
        response = f"# Personalized Real Estate Search Results\n\n"
        
        # Personalized greeting
        if is_returning and total_searches > 1:
            response += f"Welcome back! Based on your {total_searches} previous searches, I've personalized these results for you.\n\n"
        elif is_returning:
            response += f"Great to see you again! I've applied insights from our previous conversation.\n\n"
        else:
            response += f"Welcome! Let me help you find the perfect property.\n\n"
        
        response += f"**Query:** {query}\n\n"
        
        # Results summary with personalization
        exact_count = synthesis.get('total_exact_matches', 0)
        personalized_count = synthesis.get('personalized_matches', 0)
        
        response += f"## 🎯 Personalized Search Results\n"
        response += f"- **Exact Matches:** {exact_count} properties\n"
        response += f"- **Personalized Recommendations:** {personalized_count} properties\n"
        response += f"- **Market Data:** {'Available' if synthesis.get('has_market_data') else 'Limited'}\n\n"
        
        # Property recommendations
        properties = synthesis.get('recommended_properties', [])
        if properties:
            response += f"## 🏠 Top Personalized Recommendations\n\n"
            for i, prop in enumerate(properties[:3], 1):
                title = prop.get('title', 'Property')
                price = prop.get('price', 0)
                location = prop.get('location', 'Location not specified')
                score = prop.get('personalization_score', 0)
                reasons = prop.get('recommendation_reasons', [])
                
                response += f"{i}. **{title}** - ₹{price:,:}\n"
                response += f"   📍 {location}\n"
                response += f"   ⭐ Match Score: {score:.2f}\n"
                if reasons:
                    response += f"   💡 Why this matches: {', '.join(reasons[:2])}\n"
                response += f"\n"
        
        # Memory insights
        learned_prefs = synthesis.get('memory_insights', {}).get('learned_preferences', 0)
        if learned_prefs > 0:
            response += f"## 🧠 Personalization Insights\n"
            response += f"I've learned {learned_prefs} of your preferences to provide better recommendations.\n\n"
        
        # Next steps
        response += f"## 🎯 Next Steps\n"
        if is_returning:
            response += f"- Review personalized recommendations above\n"
            response += f"- Let me know if you'd like to refine your search criteria\n"
        else:
            response += f"- Explore the recommended properties\n"
            response += f"- Share more preferences to get better recommendations\n"
        
        return response

    @rate_limited(groq_rate_limiter)
    def _generate_memory_contextual_response(self, prompt: str) -> str:
        """Generate response with rate limiting"""
        try:
            # Log rate limit stats before request
            log_rate_limit_stats()
            
            response_msg = self.llm.invoke([HumanMessage(content=prompt)])
            return response_msg.content
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return "I apologize, but I'm currently experiencing high demand. Please try again in a moment."

    # Synchronous wrapper for the main async method
    def search_with_memory(self, query: str, user_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for plan_and_execute_with_memory"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.plan_and_execute_with_memory(query, user_id))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.plan_and_execute_with_memory(query, user_id))
            finally:
                loop.close()


# Example usage and testing
if __name__ == "__main__":
    async def test_memory_enhanced_planner():
        planner = MemoryEnhancedPlannerAgent()
        
        # Simulate returning user with multiple queries
        user_id = "test_user_456"
        
        test_queries = [
            "Find 3 BHK apartments in Mumbai under 2 crores",
            "Show me investment properties in Mumbai with good rental yield", 
            "Any luxury apartments in Mumbai area?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"🧠🔍 Memory Test {i}: {query}")
            print('='*80)
            
            result = await planner.plan_and_execute_with_memory(query, user_id)
            
            if "error" not in result:
                print(f"\n📋 Final Response:\n{result['final_response']}")
                print(f"\n⚡ Processing Time: {result['processing_metadata']['processing_time']:.2f}s")
                print(f"🤖 Agents Used: {', '.join(result['processing_metadata']['agents_used'])}")
                
                # Show memory insights
                memory_insights = result['agent_results']['synthesis']['memory_insights']
                print(f"🧠 Memory Insights: {memory_insights['learned_preferences']} preferences, "
                      f"{memory_insights['user_experience']} total searches")
            else:
                print(f"❌ Error: {result['error']}")
    
    # Run test
    asyncio.run(test_memory_enhanced_planner())