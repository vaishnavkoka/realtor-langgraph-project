"""
Planner Agent - Multi-Agent Real Estate Search Orchestrator
Uses LangGraph to coordinate complex workflows between all agents

This agent acts as the main coordinator that:
1. Analyzes complex user queries
2. Plans multi-step execution workflows  
3. Coordinates between Router, Structured Data, RAG, and Web Research agents
4. Aggregates and synthesizes results from multiple agents
5. Provides comprehensive responses with reasoning
"""

import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import time
import asyncio

# LangGraph imports
from langgraph.graph import StateGraph, END
try:
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
except ImportError:
    # Fallback for older versions
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    def add_messages(existing_messages, new_messages):
        """Simple fallback for add_messages functionality"""
        if existing_messages is None:
            existing_messages = []
        if isinstance(new_messages, list):
            return existing_messages + new_messages
        else:
            return existing_messages + [new_messages]

# Agent imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.query_router import QueryRouterAgent
from agents.structured_data_agent import StructuredDataAgent 
from agents.rag_agent import RAGAgent
from agents.web_research_agent import WebResearchAgent

# LLM imports - with fallback
try:
    import sys
    import os
    # Add src directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(current_dir), 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    from models.llm_models import get_llm
    from models.rate_limiter import RateLimiter
except ImportError:
    # Fallback implementations
    from groq import Groq
    import os
    
    def get_llm():
        """Fallback LLM implementation"""
        from langchain_groq import ChatGroq
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


class PlannerState(TypedDict):
    """State for the planner workflow"""
    messages: List[BaseMessage]
    query: str
    user_intent: str
    execution_plan: List[Dict[str, Any]]
    agent_results: Dict[str, Any]
    final_response: str
    processing_metadata: Dict[str, Any]


class PlannerAgent:
    """
    Master orchestrator agent using LangGraph for complex multi-agent workflows
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Planner Agent...")
        
        # Initialize all sub-agents
        self.query_router = QueryRouterAgent()
        self.structured_agent = StructuredDataAgent() 
        self.rag_agent = RAGAgent()
        self.web_agent = WebResearchAgent()
        
        # Initialize LLM and rate limiter
        self.llm = get_llm()
        self.rate_limiter = RateLimiter()
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        logger.info("✅ Planner Agent initialized successfully!")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for multi-agent coordination"""
        
        workflow = StateGraph(PlannerState)
        
        # Define workflow nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("create_execution_plan", self._create_execution_plan)
        workflow.add_node("execute_database_search", self._execute_database_search)
        workflow.add_node("execute_semantic_search", self._execute_semantic_search)
        workflow.add_node("execute_web_research", self._execute_web_research)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_query")
        
        # Conditional routing based on analysis
        workflow.add_conditional_edges(
            "analyze_query",
            self._should_create_plan,
            {
                "create_plan": "create_execution_plan",
                "simple_search": "execute_database_search"
            }
        )
        
        workflow.add_edge("create_execution_plan", "execute_database_search")
        workflow.add_edge("execute_database_search", "execute_semantic_search")
        workflow.add_edge("execute_semantic_search", "execute_web_research") 
        workflow.add_edge("execute_web_research", "synthesize_results")
        workflow.add_edge("synthesize_results", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow

    async def plan_and_execute(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for planning and executing complex queries
        """
        logger.info(f"🎯 Planning execution for query: '{query}'")
        
        start_time = time.time()
        
        # Initialize state
        initial_state = PlannerState(
            messages=[HumanMessage(content=query)],
            query=query,
            user_intent="",
            execution_plan=[],
            agent_results={},
            final_response="",
            processing_metadata={
                "start_time": start_time,
                "agents_used": [],
                "execution_steps": []
            }
        )
        
        try:
            # Execute workflow
            result = await self.app.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result["processing_metadata"]["processing_time"] = processing_time
            result["processing_metadata"]["end_time"] = time.time()
            
            logger.info(f"⚡ Query execution completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in plan execution: {e}")
            return {
                "error": str(e),
                "query": query,
                "processing_metadata": {
                    "start_time": start_time,
                    "processing_time": time.time() - start_time,
                    "error": True
                }
            }

    def _analyze_query(self, state: PlannerState) -> PlannerState:
        """Step 1: Analyze the user query to understand intent and complexity"""
        logger.info("🔍 Step 1: Analyzing query complexity and intent...")
        
        query = state["query"]
        
        # Use router to get initial intent analysis
        routing_result = self.query_router.route_query(query)
        
        state["user_intent"] = routing_result["intent"]
        state["processing_metadata"]["query_analysis"] = routing_result
        state["processing_metadata"]["execution_steps"].append("analyze_query")
        
        logger.info(f"   📋 Detected intent: {routing_result['intent']}")
        logger.info(f"   🎯 Complexity: {routing_result['method']}")
        
        return state

    def _should_create_plan(self, state: PlannerState) -> str:
        """Decision node: Determine if complex planning is needed"""
        
        query = state["query"].lower()
        routing_result = state["processing_metadata"]["query_analysis"]
        
        # Complex queries need full planning
        complex_indicators = [
            "compare", "best", "investment", "analysis", "market trends",
            "versus", "vs", "recommend", "suggest", "evaluate", "portfolio"
        ]
        
        is_complex = (
            any(indicator in query for indicator in complex_indicators) or
            routing_result["method"] == "llm" or
            len(routing_result["extracted_entities"]) > 2
        )
        
        if is_complex:
            logger.info("🧠 Complex query detected - creating execution plan")
            return "create_plan"
        else:
            logger.info("⚡ Simple query detected - executing direct search")
            return "simple_search"

    def _create_execution_plan(self, state: PlannerState) -> PlannerState:
        """Step 2: Create detailed execution plan for complex queries"""
        logger.info("📋 Step 2: Creating execution plan...")
        
        query = state["query"]
        routing_result = state["processing_metadata"]["query_analysis"]
        
        # Create execution plan based on query analysis
        plan = []
        
        # Always start with database search
        plan.append({
            "step": 1,
            "agent": "structured_data",
            "action": "database_search", 
            "parameters": routing_result["extracted_entities"],
            "reasoning": "Search structured database for exact matches"
        })
        
        # Add semantic search for broader results
        plan.append({
            "step": 2,
            "agent": "rag",
            "action": "semantic_search",
            "parameters": {"query": query, "limit": 10},
            "reasoning": "Find semantically similar properties using embeddings"
        })
        
        # Add web research for market context
        location = routing_result["extracted_entities"].get("location", "")
        if location:
            plan.append({
                "step": 3,
                "agent": "web_research", 
                "action": "market_research",
                "parameters": {"location": location},
                "reasoning": f"Get market insights and trends for {location}"
            })
            
            plan.append({
                "step": 4,
                "agent": "web_research",
                "action": "area_insights", 
                "parameters": {"location": location},
                "reasoning": f"Get area-specific insights for {location}"
            })
        
        state["execution_plan"] = plan
        state["processing_metadata"]["execution_steps"].append("create_execution_plan")
        
        logger.info(f"   📝 Created {len(plan)}-step execution plan")
        for step in plan:
            logger.info(f"      {step['step']}. {step['agent']} - {step['reasoning']}")
        
        return state

    def _execute_database_search(self, state: PlannerState) -> PlannerState:
        """Step 3a: Execute structured database search"""
        logger.info("🗃️  Step 3a: Executing database search...")
        
        try:
            query_info = state["processing_metadata"]["query_analysis"]
            search_params = {
                "query": state["query"],
                "extracted_entities": query_info["extracted_entities"]
            }
            
            result = self.structured_agent.search_properties(search_params)
            state["agent_results"]["database"] = result
            state["processing_metadata"]["agents_used"].append("StructuredDataAgent")
            state["processing_metadata"]["execution_steps"].append("execute_database_search")
            
            count = result["count"] if result["success"] else 0
            logger.info(f"   ✅ Database search: {count} exact matches found")
            
        except Exception as e:
            logger.error(f"   ❌ Database search failed: {e}")
            state["agent_results"]["database"] = {"success": False, "error": str(e)}
        
        return state

    def _execute_semantic_search(self, state: PlannerState) -> PlannerState:
        """Step 3b: Execute semantic RAG search"""
        logger.info("🧠 Step 3b: Executing semantic search...")
        
        try:
            search_params = {
                "query": state["query"],
                "limit": 8,
                "include_summary": False
            }
            
            result = self.rag_agent.semantic_search(search_params)
            state["agent_results"]["semantic"] = result
            state["processing_metadata"]["agents_used"].append("RAGAgent")
            state["processing_metadata"]["execution_steps"].append("execute_semantic_search")
            
            count = result["count"] if result["success"] else 0
            logger.info(f"   ✅ Semantic search: {count} relevant matches found")
            
        except Exception as e:
            logger.error(f"   ❌ Semantic search failed: {e}")
            state["agent_results"]["semantic"] = {"success": False, "error": str(e)}
        
        return state

    def _execute_web_research(self, state: PlannerState) -> PlannerState:
        """Step 3c: Execute web research for market insights"""
        logger.info("🌐 Step 3c: Executing web research...")
        
        try:
            query_info = state["processing_metadata"]["query_analysis"]
            location = query_info["extracted_entities"].get("location", "India")
            property_type = query_info["extracted_entities"].get("property_type", "")
            
            # Market trends research
            market_params = {
                "location": location,
                "property_type": property_type,
                "time_period": "2024"
            }
            market_result = self.web_agent.research_market_trends(market_params)
            
            # Area insights research
            area_params = {"location": location}
            area_result = self.web_agent.get_area_insights(area_params)
            
            state["agent_results"]["web_research"] = {
                "market_trends": market_result,
                "area_insights": area_result
            }
            state["processing_metadata"]["agents_used"].append("WebResearchAgent")
            state["processing_metadata"]["execution_steps"].append("execute_web_research")
            
            logger.info(f"   ✅ Web research completed for {location}")
            
        except Exception as e:
            logger.error(f"   ❌ Web research failed: {e}")
            state["agent_results"]["web_research"] = {"error": str(e)}
        
        return state

    def _synthesize_results(self, state: PlannerState) -> PlannerState:
        """Step 4: Synthesize results from all agents"""
        logger.info("🔗 Step 4: Synthesizing results from all agents...")
        
        # Collect all results
        database_results = state["agent_results"].get("database", {})
        semantic_results = state["agent_results"].get("semantic", {})
        web_results = state["agent_results"].get("web_research", {})
        
        # Create synthesis
        synthesis = {
            "total_exact_matches": database_results.get("count", 0) if database_results.get("success") else 0,
            "total_semantic_matches": semantic_results.get("count", 0) if semantic_results.get("success") else 0,
            "has_market_data": web_results.get("market_trends", {}).get("success", False),
            "has_area_insights": web_results.get("area_insights", {}).get("success", False),
            "recommended_properties": [],
            "market_summary": "",
            "investment_insights": []
        }
        
        # Combine property recommendations
        if database_results.get("success") and database_results.get("properties"):
            synthesis["recommended_properties"].extend(database_results["properties"][:3])
        
        if semantic_results.get("success") and semantic_results.get("properties"):
            # Add semantic results that aren't already in database results
            semantic_props = semantic_results["properties"][:5]
            existing_ids = {prop.get("property_id") for prop in synthesis["recommended_properties"]}
            for prop in semantic_props:
                if prop.get("property_id") not in existing_ids:
                    synthesis["recommended_properties"].append(prop)
        
        # Extract market summary
        if web_results.get("market_trends", {}).get("success"):
            market_data = web_results["market_trends"]
            if market_data.get("market_analysis"):
                synthesis["market_summary"] = market_data["market_analysis"][:500] + "..."
        
        state["agent_results"]["synthesis"] = synthesis
        state["processing_metadata"]["execution_steps"].append("synthesize_results")
        
        logger.info(f"   📊 Synthesized {len(synthesis['recommended_properties'])} property recommendations")
        
        return state

    def _generate_response(self, state: PlannerState) -> PlannerState:
        """Step 5: Generate comprehensive final response using LLM"""
        logger.info("✍️  Step 5: Generating comprehensive response...")
        
        try:
            # Prepare context for LLM
            query = state["query"]
            synthesis = state["agent_results"].get("synthesis", {})
            
            # Build comprehensive prompt
            prompt = self._build_response_prompt(query, synthesis, state)
            
            # Generate response using LLM
            if self.rate_limiter.can_make_request("groq"):
                response_msg = self.llm.invoke([HumanMessage(content=prompt)])
                final_response = response_msg.content
                self.rate_limiter.record_request("groq")
            else:
                final_response = self._generate_fallback_response(query, synthesis)
            
            state["final_response"] = final_response
            state["processing_metadata"]["execution_steps"].append("generate_response")
            
            logger.info("   ✅ Comprehensive response generated")
            
        except Exception as e:
            logger.error(f"   ❌ Response generation failed: {e}")
            state["final_response"] = self._generate_fallback_response(
                state["query"], 
                state["agent_results"].get("synthesis", {})
            )
        
        return state

    def _build_response_prompt(self, query: str, synthesis: Dict, state: PlannerState) -> str:
        """Build comprehensive prompt for LLM response generation"""
        
        prompt = f"""
You are an expert real estate advisor providing comprehensive property search results. 

USER QUERY: "{query}"

SEARCH RESULTS SUMMARY:
- Exact Database Matches: {synthesis.get('total_exact_matches', 0)}
- Semantic Matches: {synthesis.get('total_semantic_matches', 0)}  
- Market Data Available: {synthesis.get('has_market_data', False)}
- Area Insights Available: {synthesis.get('has_area_insights', False)}

RECOMMENDED PROPERTIES:
{self._format_properties_for_prompt(synthesis.get('recommended_properties', []))}

MARKET INSIGHTS:
{synthesis.get('market_summary', 'No market data available')}

PROCESSING METADATA:
- Agents Used: {', '.join(state['processing_metadata']['agents_used'])}
- Processing Steps: {len(state['processing_metadata']['execution_steps'])} steps

Please provide a comprehensive response that:
1. Directly answers the user's query
2. Summarizes the search results with specific property recommendations
3. Provides market context and investment insights when available
4. Includes actionable next steps for the user
5. Maintains a professional yet conversational tone

Format your response with clear sections and bullet points for easy reading.
"""
        return prompt

    def _format_properties_for_prompt(self, properties: List[Dict]) -> str:
        """Format property list for LLM prompt"""
        if not properties:
            return "No properties found matching the criteria."
        
        formatted = []
        for i, prop in enumerate(properties[:5], 1):
            title = prop.get('title', 'Property')
            price = prop.get('price', 0)
            location = prop.get('location', 'Location not specified')
            rooms = prop.get('num_rooms', 'N/A')
            
            formatted.append(f"{i}. {title} - ₹{price:,} ({rooms} rooms) in {location}")
        
        return "\n".join(formatted)

    def _generate_fallback_response(self, query: str, synthesis: Dict) -> str:
        """Generate fallback response when LLM is unavailable"""
        
        exact_count = synthesis.get('total_exact_matches', 0)
        semantic_count = synthesis.get('total_semantic_matches', 0)
        properties = synthesis.get('recommended_properties', [])
        
        response = f"# Real Estate Search Results\n\n"
        response += f"**Query:** {query}\n\n"
        
        # Results summary
        response += f"## 📊 Search Results Summary\n"
        response += f"- **Exact Matches:** {exact_count} properties\n"
        response += f"- **Related Properties:** {semantic_count} properties\n"
        response += f"- **Market Data:** {'Available' if synthesis.get('has_market_data') else 'Limited'}\n\n"
        
        # Property recommendations
        if properties:
            response += f"## 🏠 Top Property Recommendations\n\n"
            for i, prop in enumerate(properties[:3], 1):
                title = prop.get('title', 'Property')
                price = prop.get('price', 0)
                location = prop.get('location', 'Location not specified')
                response += f"{i}. **{title}** - ₹{price:,:}\n   📍 {location}\n\n"
        else:
            response += f"## 🔍 No Exact Matches Found\n\n"
            response += f"Consider expanding your search criteria or exploring similar properties in nearby areas.\n\n"
        
        # Market insights
        if synthesis.get('market_summary'):
            response += f"## 📈 Market Insights\n\n{synthesis['market_summary'][:300]}...\n\n"
        
        response += f"## 🎯 Next Steps\n"
        response += f"- Review the recommended properties above\n"
        response += f"- Contact our agents for detailed property information\n"
        response += f"- Schedule property viewings for shortlisted options\n"
        
        return response

    # Synchronous wrapper for the main async method
    def search(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for plan_and_execute"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.plan_and_execute(query))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.plan_and_execute(query))
            finally:
                loop.close()


# Example usage and testing
if __name__ == "__main__":
    async def test_planner():
        planner = PlannerAgent()
        
        test_queries = [
            "Find luxury villas in Mumbai under 5 crores with good investment potential",
            "Compare apartments in Bangalore vs Hyderabad for first-time buyers",
            "Best studio apartments in Jamshedpur for rental income"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"🔍 Testing Query: {query}")
            print('='*80)
            
            result = await planner.plan_and_execute(query)
            
            if "error" not in result:
                print(f"\n📋 Final Response:\n{result['final_response']}")
                print(f"\n⚡ Processing Time: {result['processing_metadata']['processing_time']:.2f}s")
                print(f"🤖 Agents Used: {', '.join(result['processing_metadata']['agents_used'])}")
            else:
                print(f"❌ Error: {result['error']}")
    
    # Run test
    asyncio.run(test_planner())