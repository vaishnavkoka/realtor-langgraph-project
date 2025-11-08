"""
LangGraph-based Real Estate Agent Orchestrator

This module provides sophisticated agent orchestration using LangGraph to manage
multiple specialized agents based on user queries. Each agent is activated based
on intelligent query analysis and workflow state management.

Agents Available:
1. ReportGenerationAgent - User preference analysis, market reports
2. RenovationEstimationAgent - BHK-wise renovation cost calculations  
3. StructuredDataAgent - Database search and property filtering
4. RAGAgent - Semantic search and knowledge retrieval
5. WebResearchAgent - Real-time market research
6. QueryRouterAgent - Intelligent query routing and intent detection

Features:
- Parallel agent execution for complex queries
- Smart agent selection based on query intent
- State management across workflow steps
- Result synthesis and aggregation
- Error handling and fallback mechanisms
"""

import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Literal
from datetime import datetime
import time
import asyncio

# LangGraph core imports
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Agent imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.query_router import QueryRouterAgent
from agents.structured_data_agent import StructuredDataAgent 
from agents.rag_agent import RAGAgent
from agents.web_research_agent import WebResearchAgent
from agents.report_generation_agent import ReportGenerationAgent
from agents.renovation_estimation_agent import RenovationEstimationAgent

# LLM and utilities
try:
    from models.llm_models import get_llm
    from utils.rate_limiter import RateLimiter
except ImportError:
    # Fallback implementations
    from langchain_groq import ChatGroq
    import os
    
    def get_llm():
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
    
    class RateLimiter:
        def can_make_request(self, service): return True
        def record_request(self, service): pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentWorkflowState(TypedDict):
    """State definition for the LangGraph agent orchestration workflow"""
    
    # Input and context
    messages: List[BaseMessage]
    original_query: str
    user_id: str
    search_results: Optional[Dict[str, Any]]
    
    # Query analysis
    query_intent: str
    query_complexity: str  # simple, moderate, complex
    required_agents: List[str]
    agent_priorities: Dict[str, float]
    
    # Agent execution state
    active_agents: List[str]
    agent_results: Dict[str, Any]
    agent_errors: Dict[str, str]
    execution_metadata: Dict[str, Any]
    
    # Final output
    synthesized_result: Optional[Dict[str, Any]]
    final_response: str
    confidence_score: float


class LangGraphRealEstateOrchestrator:
    """
    Advanced LangGraph-based orchestrator for Real Estate Agent System
    
    This orchestrator intelligently activates and coordinates multiple agents
    based on user query analysis, managing complex workflows with state persistence.
    """
    
    def __init__(self):
        logger.info(" Initializing LangGraph Real Estate Orchestrator...")
        
        # Initialize all available agents
        self.agents = {
            'query_router': QueryRouterAgent(),
            'structured_data': StructuredDataAgent(),
            'rag': RAGAgent(), 
            'web_research': WebResearchAgent(),
            'report_generation': ReportGenerationAgent(),
            'renovation_estimation': RenovationEstimationAgent()
        }
        
        # Initialize LLM and utilities
        self.llm = get_llm()
        self.rate_limiter = RateLimiter()
        
        # Build and compile LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        logger.info(" LangGraph Orchestrator initialized with 6 specialized agents!")

    def _build_workflow(self) -> StateGraph:
        """Build the comprehensive LangGraph workflow for agent orchestration"""
        
        # Initialize workflow with state
        workflow = StateGraph(AgentWorkflowState)
        
        # === WORKFLOW NODES ===
        
        # 1. Query Analysis Phase
        workflow.add_node("analyze_query", self._analyze_query_intent)
        workflow.add_node("determine_agents", self._determine_required_agents)
        
        # 2. Agent Execution Phase  
        workflow.add_node("execute_structured_search", self._execute_structured_data_agent)
        workflow.add_node("execute_semantic_search", self._execute_rag_agent)
        workflow.add_node("execute_web_research", self._execute_web_research_agent)
        workflow.add_node("execute_report_generation", self._execute_report_generation_agent)
        workflow.add_node("execute_renovation_estimation", self._execute_renovation_estimation_agent)
        
        # 3. Synthesis Phase
        workflow.add_node("synthesize_results", self._synthesize_agent_results)
        workflow.add_node("generate_final_response", self._generate_final_response)
        
        # === WORKFLOW ROUTING ===
        
        # Entry point
        workflow.set_entry_point("analyze_query")
        
        # Query analysis flow
        workflow.add_edge("analyze_query", "determine_agents")
        
        # Conditional agent routing based on requirements
        workflow.add_conditional_edges(
            "determine_agents",
            self._route_to_agents,
            {
                "structured_only": "execute_structured_search",
                "report_generation": "execute_report_generation", 
                "renovation_estimation": "execute_renovation_estimation",
                "comprehensive": "execute_structured_search",
                "web_research": "execute_web_research"
            }
        )
        
        # Structured search routing
        workflow.add_conditional_edges(
            "execute_structured_search",
            self._continue_after_structured,
            {
                "to_semantic": "execute_semantic_search",
                "to_synthesis": "synthesize_results",
                "to_reports": "execute_report_generation"
            }
        )
        
        # Semantic search routing  
        workflow.add_conditional_edges(
            "execute_semantic_search",
            self._continue_after_semantic,
            {
                "to_web": "execute_web_research",
                "to_synthesis": "synthesize_results",
                "to_reports": "execute_report_generation"
            }
        )
        
        # Agent completion routing
        workflow.add_edge("execute_web_research", "synthesize_results")
        workflow.add_edge("execute_report_generation", "synthesize_results")
        workflow.add_edge("execute_renovation_estimation", "synthesize_results")
        
        # Final synthesis
        workflow.add_edge("synthesize_results", "generate_final_response")
        workflow.add_edge("generate_final_response", END)
        
        return workflow

    # === WORKFLOW NODES IMPLEMENTATION ===

    async def _analyze_query_intent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Analyze user query to understand intent and complexity"""
        logger.info(" Analyzing query intent and complexity...")
        
        try:
            query = state["original_query"]
            
            analysis_prompt = f"""
            Analyze this real estate query to understand user intent and complexity:
            
            Query: "{query}"
            
            Provide analysis in JSON format:
            {{
                "primary_intent": "search|analysis|estimation|comparison|information",
                "secondary_intents": ["intent1", "intent2"],
                "complexity": "simple|moderate|complex",
                "domain_focus": "properties|market|renovation|investment|location",
                "requires_data_search": true|false,
                "requires_calculations": true|false,
                "requires_external_research": true|false,
                "time_sensitivity": "low|medium|high"
            }}
            
            Intent Definitions:
            - search: Finding specific properties or data
            - analysis: Market analysis, trends, reports  
            - estimation: Cost calculations, valuations
            - comparison: Comparing properties or options
            - information: General information requests
            
            Complexity Factors:
            - simple: Single intent, basic query
            - moderate: Multiple intents, some analysis needed
            - complex: Multi-step process, multiple agents needed
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Try to parse JSON, with fallback
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("JSON parsing failed, using fallback analysis")
                analysis = {
                    "primary_intent": "search",
                    "secondary_intents": [],
                    "complexity": "simple",
                    "domain_focus": "properties",
                    "requires_data_search": True,
                    "requires_calculations": "renovat" in query.lower(),
                    "requires_external_research": False,
                    "time_sensitivity": "low"
                }
            
            # Update state with analysis
            state["query_intent"] = analysis["primary_intent"]
            state["query_complexity"] = analysis["complexity"]
            state["execution_metadata"] = {
                "analysis": analysis,
                "analyzed_at": datetime.now().isoformat()
            }
            
            logger.info(f" Query Analysis: Intent={analysis['primary_intent']}, Complexity={analysis['complexity']}")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback
            state["query_intent"] = "search"
            state["query_complexity"] = "simple"
            state["execution_metadata"] = {"analysis_error": str(e)}
        
        return state

    async def _determine_required_agents(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Determine which agents are required based on query analysis"""
        logger.info(" Determining required agents for execution...")
        
        query = state["original_query"].lower()
        intent = state["query_intent"]
        complexity = state["query_complexity"]
        analysis = state["execution_metadata"].get("analysis", {})
        
        required_agents = []
        priorities = {}
        
        # === AGENT SELECTION LOGIC ===
        
        # 1. Renovation Estimation Agent - ONLY for explicit renovation queries
        if any(keyword in query for keyword in ["renovat", "renovation", "repair", "upgrade", "remodel", "refurbish", "interior", "cost estimate", "estimate cost"]) and \
           not any(search_keyword in query for search_keyword in ["search", "find", "looking for", "available", "properties", "buy", "purchase"]):
            required_agents.append("renovation_estimation")
            priorities["renovation_estimation"] = 0.9
            logger.info("🔨 Renovation Estimation Agent selected")
        
        # 2. Report Generation Agent
        if any(keyword in query for keyword in ["report", "analysis", "summary", "preference", "insight"]):
            required_agents.append("report_generation") 
            priorities["report_generation"] = 0.8
            logger.info("📊 Report Generation Agent selected")
        
        # 3. Structured Data Agent (primary for property searches)
        if intent in ["search", "comparison"] or analysis.get("requires_data_search", False) or \
           any(keyword in query for keyword in ["bhk", "bedroom", "property", "house", "apartment", "flat", "villa", "location", "city", "price", "buy", "purchase", "find", "available"]):
            required_agents.append("structured_data")
            priorities["structured_data"] = 0.8  # Primary for property searches
            logger.info(" Structured Data Agent selected")
            
            # ALWAYS add RAG agent for property searches to provide fallback and enhancement
            required_agents.append("rag")
            priorities["rag"] = 0.7  # Complementary to structured search
            logger.info(" RAG Agent selected (automatic for property searches)")
        
        # 4. RAG Agent for complex queries (standalone)
        elif complexity in ["moderate", "complex"] or analysis.get("domain_focus") in ["market", "investment"]:
            required_agents.append("rag")
            priorities["rag"] = 0.8  # Higher priority when standalone
            logger.info(" RAG Agent selected (standalone for complex queries)")
        
        # 5. Web Research Agent for current market info
        if any(keyword in query for keyword in ["current", "latest", "recent", "trend", "market"]) or analysis.get("requires_external_research", False):
            required_agents.append("web_research")
            priorities["web_research"] = 0.5
            logger.info("Web Research Agent selected")
        
        # Default to comprehensive search if no agents selected
        if not required_agents:
            required_agents = ["structured_data", "rag"]  # Use both for better coverage
            priorities["structured_data"] = 0.8
            priorities["rag"] = 0.7
            logger.info(" Default: Structured Data + RAG Agents selected")
        
        # Update state
        state["required_agents"] = required_agents
        state["agent_priorities"] = priorities
        state["active_agents"] = []
        state["agent_results"] = {}
        state["agent_errors"] = {}
        
        logger.info(f" Selected {len(required_agents)} agents: {required_agents}")
        
        return state

    def _route_to_agents(self, state: AgentWorkflowState) -> Literal["structured_only", "report_generation", "renovation_estimation", "comprehensive", "web_research"]:
        """Route to appropriate agent execution path"""
        
        required = state["required_agents"]
        priorities = state["agent_priorities"]
        
        # Single agent scenarios
        if len(required) == 1:
            if "renovation_estimation" in required:
                return "renovation_estimation"
            elif "report_generation" in required:
                return "report_generation"
            elif "web_research" in required:
                return "web_research" 
            else:
                return "structured_only"
        
        # Multi-agent scenarios - prioritize comprehensive execution
        if "renovation_estimation" in required and priorities.get("renovation_estimation", 0) > 0.8:
            return "renovation_estimation"
        elif "report_generation" in required and priorities.get("report_generation", 0) > 0.7:
            return "report_generation"
        else:
            # Use comprehensive for multi-agent execution (structured + rag + web as needed)
            return "comprehensive"

    # === AGENT EXECUTION METHODS ===

    async def _execute_structured_data_agent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Execute structured data search agent"""
        logger.info(" Executing Structured Data Agent...")
        
        try:
            agent = self.agents["structured_data"]
            query = state["original_query"]
            
            # Format the query as expected by the structured data agent
            search_criteria = {
                "query": query,
                "extracted_entities": self._extract_entities_from_query(query)
            }
            
            # Execute structured search
            result = await asyncio.to_thread(agent.search_properties, search_criteria)
            
            state["agent_results"]["structured_data"] = result
            state["active_agents"].append("structured_data")
            
            properties_found = len(result.get('properties', []))
            logger.info(f" Structured Data Agent completed: Found {properties_found} properties")
            
            # If no properties found, automatically activate RAG agent for additional context
            if properties_found == 0:
                logger.info(" No properties found, activating RAG agent for additional context...")
                
                # Add RAG agent to required agents if not already present
                if "rag" not in state.get("required_agents", []):
                    state.setdefault("required_agents", []).append("rag")
                
                # Set flag to indicate RAG should provide fallback suggestions
                state["needs_rag_fallback"] = True
            
        except Exception as e:
            logger.error(f"Property search failed: {e}")
            state["agent_results"]["structured_data"] = {
                "success": False,
                "error": str(e),
                "properties": [],
                "count": 0,
                "agent": "StructuredDataAgent"
            }
            state["agent_errors"]["structured_data"] = str(e)
            
            # Also trigger RAG agent on error for fallback suggestions
            if "rag" not in state.get("required_agents", []):
                state.setdefault("required_agents", []).append("rag")
            state["needs_rag_fallback"] = True
            logger.info(" Structured search failed, activating RAG agent for fallback...")
        
        return state

    def _extract_entities_from_query(self, query: str) -> Dict[str, Any]:
        """Extract entities from natural language query for structured search"""
        entities = {}
        query_lower = query.lower()
        
        # Extract location
        import re
        location_match = re.search(r'\bin\s+(\w+)', query_lower)
        if location_match:
            entities["location"] = location_match.group(1)
        
        # Extract rooms/bedrooms
        room_patterns = [
            r'(\d+)\s*(?:bhk|bedroom|room|bed)',
            r'(\d+)\s*room',
            r'(\d+)\s*bed'
        ]
        for pattern in room_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities["rooms"] = int(match.group(1))
                entities["bedrooms"] = int(match.group(1))
                break
        
        # Extract property type
        property_types = ['villa', 'apartment', 'house', 'flat', 'studio']
        for prop_type in property_types:
            if prop_type in query_lower:
                entities["property_type"] = prop_type
                break
        
        # Extract budget keywords
        if any(word in query_lower for word in ['under', 'below', 'less than', 'budget']):
            # Could extract specific amounts, but for now just mark that budget is mentioned
            entities["has_budget"] = True
        
        return entities

    async def _execute_rag_agent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Execute RAG semantic search agent"""
        logger.info(" Executing RAG Agent...")
        
        try:
            agent = self.agents["rag"]
            query = state["original_query"]
            
            # Check if this is a fallback activation due to no structured results
            structured_result = state.get("agent_results", {}).get("structured_data", {})
            properties_found = len(structured_result.get("properties", []))
            is_fallback = state.get("needs_rag_fallback", False) or (properties_found == 0 and "structured_data" in state.get("active_agents", []))
            
            if is_fallback:
                logger.info("🔄 RAG Agent activated as fallback - providing alternative suggestions...")
                
                # Modify query to get broader suggestions
                entities = self._extract_entities_from_query(query)
                location = entities.get("location", "")
                rooms = entities.get("rooms", "")
                prop_type = entities.get("property_type", "property")
                
                # Create broader search queries for better suggestions
                fallback_queries = [
                    f"properties in {location}" if location else "",
                    f"{rooms} room properties" if rooms else "",
                    f"{prop_type} properties" if prop_type else "",
                    "property investment tips",
                    "real estate market insights"
                ]
                
                # Execute multiple searches for comprehensive fallback
                fallback_results = []
                for fallback_query in fallback_queries:
                    if fallback_query:  # Skip empty queries
                        try:
                            search_criteria = {"query": fallback_query, "extracted_entities": {}}
                            result = await asyncio.to_thread(agent.semantic_search, search_criteria)
                            fallback_results.append({
                                "query": fallback_query,
                                "result": result
                            })
                        except Exception as e:
                            logger.warning(f"RAG fallback query failed: {fallback_query}, {e}")
                
                # Combine results
                combined_result = {
                    "success": True,
                    "is_fallback": True,
                    "original_query": query,
                    "fallback_suggestions": fallback_results,
                    "message": f"No exact matches found for '{query}'. Here are some alternatives and suggestions:",
                    "agent": "RAGAgent"
                }
                
                state["agent_results"]["rag"] = combined_result
                
            else:
                # Execute normal semantic search
                search_criteria = {"query": query, "extracted_entities": self._extract_entities_from_query(query)}
                result = await asyncio.to_thread(agent.semantic_search, search_criteria)
                state["agent_results"]["rag"] = result
            
            state["active_agents"].append("rag")
            
            logger.info(" RAG Agent completed semantic search")
            
        except Exception as e:
            logger.error(f"RAG Agent failed: {e}")
            state["agent_results"]["rag"] = {
                "success": False,
                "error": str(e),
                "is_fallback": state.get("needs_rag_fallback", False),
                "agent": "RAGAgent"
            }
            state["agent_errors"]["rag"] = str(e)
        
        return state

    async def _execute_web_research_agent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Execute web research agent"""
        logger.info(" Executing Web Research Agent...")
        
        try:
            agent = self.agents["web_research"]
            query = state["original_query"]
            
            # Execute web research
            result = await asyncio.to_thread(agent.research_market_trends, query)
            
            state["agent_results"]["web_research"] = result
            state["active_agents"].append("web_research")
            
            logger.info(" Web Research Agent completed")
            
        except Exception as e:
            logger.error(f"Web Research Agent failed: {e}")
            state["agent_errors"]["web_research"] = str(e)
        
        return state

    async def _execute_report_generation_agent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Execute report generation agent"""
        logger.info("📊 Executing Report Generation Agent...")
        
        try:
            agent = self.agents["report_generation"]
            query = state["original_query"]
            user_id = state["user_id"]
            search_results = state.get("search_results", {})
            
            # Execute report generation
            result = await asyncio.to_thread(agent.process_report_request, query, user_id, search_results)
            
            state["agent_results"]["report_generation"] = result
            state["active_agents"].append("report_generation")
            
            logger.info(f" Report Generation Agent completed: {result.get('report_type')}")
            
        except Exception as e:
            logger.error(f"Report Generation Agent failed: {e}")
            state["agent_errors"]["report_generation"] = str(e)
        
        return state

    async def _execute_renovation_estimation_agent(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Execute renovation estimation agent"""
        logger.info("🔨 Executing Renovation Estimation Agent...")
        
        try:
            agent = self.agents["renovation_estimation"]
            
            # Extract property details from query or search results
            bhk_config = "2BHK"  # Default
            property_type = "apartment"  # Default
            area = 1200  # Default
            renovation_level = "premium"  # Default
            
            # Try to extract from query
            query_lower = state["original_query"].lower()
            if "studio" in query_lower:
                bhk_config = "studio"
                property_type = "studio"
            elif "1bhk" in query_lower:
                bhk_config = "1BHK"
            elif "3bhk" in query_lower:
                bhk_config = "3BHK"
            elif "4bhk" in query_lower:
                bhk_config = "4BHK"
            elif "villa" in query_lower:
                property_type = "villa"
            
            # Extract renovation level
            if "basic" in query_lower:
                renovation_level = "basic"
            elif "luxury" in query_lower or "premium" in query_lower:
                renovation_level = "luxury"
            elif "complete" in query_lower:
                renovation_level = "complete"
            
            # Execute renovation estimation
            result = await asyncio.to_thread(
                agent.estimate_renovation_cost,
                property_type=property_type,
                bhk_config=bhk_config,
                total_area=area,
                renovation_level=renovation_level
            )
            
            # Convert RenovationEstimate object to dictionary for consistency
            result_dict = {
                "success": True,
                "agent": "RenovationEstimationAgent",
                "property_type": result.property_type,
                "bhk_config": result.bhk_config,
                "total_area": result.total_area,
                "renovation_level": result.renovation_level,
                "total_cost": result.total_cost,
                "cost_per_sqft": result.cost_per_sqft,
                "room_breakdown": result.room_breakdown,
                "category_breakdown": result.category_breakdown,
                "timeline_weeks": result.timeline_weeks,
                "estimated_at": result.estimated_at.isoformat() if result.estimated_at else None,
                "recommendations": result.recommendations,
                "cost_factors": result.cost_factors
            }
            
            state["agent_results"]["renovation_estimation"] = result_dict
            state["active_agents"].append("renovation_estimation")
            
            logger.info(f" Renovation Estimation Agent completed: ₹{result.total_cost:,.0f}")
            
        except Exception as e:
            logger.error(f"Renovation Estimation Agent failed: {e}")
            state["agent_errors"]["renovation_estimation"] = str(e)
        
        return state

    # === ROUTING CONDITIONS ===

    def _continue_after_structured(self, state: AgentWorkflowState) -> Literal["to_semantic", "to_synthesis", "to_reports"]:
        """Determine next step after structured search"""
        required = state["required_agents"]
        
        if "rag" in required:
            return "to_semantic"
        elif "report_generation" in required:
            return "to_reports"
        else:
            return "to_synthesis"

    def _continue_after_semantic(self, state: AgentWorkflowState) -> Literal["to_web", "to_synthesis", "to_reports"]:
        """Determine next step after semantic search"""
        required = state["required_agents"]
        
        if "web_research" in required:
            return "to_web"
        elif "report_generation" in required:
            return "to_reports"
        else:
            return "to_synthesis"

    # === SYNTHESIS METHODS ===

    async def _synthesize_agent_results(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Synthesize results from all executed agents"""
        logger.info(" Synthesizing results from all agents...")
    
        try:
            agent_results = state["agent_results"]
            agent_errors = state["agent_errors"]
            
            synthesis = {
                "executed_agents": state["active_agents"],
                "successful_agents": list(agent_results.keys()),
                "failed_agents": list(agent_errors.keys()),
                "primary_result": None,
                "supporting_data": {},
                "confidence_score": 0.0
            }
            
            # Determine primary result based on priorities
            priorities = state["agent_priorities"]
            best_agent = None
            best_score = 0
            
            for agent_name, result in agent_results.items():
                score = priorities.get(agent_name, 0)
                if score > best_score:
                    best_score = score
                    best_agent = agent_name
            
            if best_agent:
                synthesis["primary_result"] = agent_results[best_agent]
                synthesis["confidence_score"] = best_score
                
                # Add supporting data from other agents
                for agent_name, result in agent_results.items():
                    if agent_name != best_agent:
                        synthesis["supporting_data"][agent_name] = result
            
            state["synthesized_result"] = synthesis
            state["confidence_score"] = synthesis["confidence_score"]
            
            logger.info(f" Synthesis completed. Primary agent: {best_agent}, Confidence: {synthesis['confidence_score']}")
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            state["synthesized_result"] = {"error": str(e)}
            state["confidence_score"] = 0.0
        
        return state

    async def _generate_final_response(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Generate final comprehensive response"""
        logger.info(" Generating final response...")
        
        try:
            synthesis = state["synthesized_result"]
            query = state["original_query"]
            agent_results = state.get("agent_results", {})
            
            if synthesis and "primary_result" in synthesis:
                primary = synthesis["primary_result"]
                
                # Check if we have a RAG fallback scenario
                structured_result = agent_results.get("structured_data", {})
                rag_result = agent_results.get("rag", {})
                
                if (structured_result.get("properties", []) == [] and 
                    rag_result.get("is_fallback", False)):
                    
                    # Generate response for no properties found with RAG suggestions
                    response = f"🔍 No exact matches found for '{query}'\n\n"
                    
                    if structured_result.get("success", True):
                        response += " Database searched successfully - 0 properties match your criteria\n"
                    else:
                        response += f" Database search encountered issues: {structured_result.get('error', 'Unknown error')}\n"
                    
                    response += "\n Alternative suggestions from market analysis:\n"
                    
                    if rag_result.get("fallback_suggestions"):
                        for i, suggestion in enumerate(rag_result["fallback_suggestions"][:3], 1):
                            response += f"{i}. {suggestion.get('query', 'Alternative search')}\n"
                    
                    response += "\n Consider:\n"
                    response += "• Expanding your search area\n"
                    response += "• Adjusting room requirements\n" 
                    response += "• Checking similar property types\n"
                    response += "• Reviewing your budget range"
                
                # Format response based on primary result type
                elif state["active_agents"] and "renovation_estimation" in state["active_agents"]:
                    renovation_result = state["agent_results"]["renovation_estimation"]
                    response = f"🔨 Renovation Estimation:\n"
                    response += f"Total Cost: ₹{renovation_result['total_cost']:,.0f}\n"
                    response += f"Timeline: {renovation_result['timeline_weeks']} weeks\n"
                    response += f"Property: {renovation_result['bhk_config']} {renovation_result['property_type']}\n"
                    response += f"Area: {renovation_result['total_area']:,.0f} sq ft"
                    
                elif state["active_agents"] and "report_generation" in state["active_agents"]:
                    report_result = state["agent_results"]["report_generation"]
                    response = f"📊 {report_result.get('report_type', 'Report').title()} Generated:\n"
                    response += f"Summary: {report_result.get('summary', 'Report completed successfully')}"
                    
                elif structured_result and len(structured_result.get("properties", [])) > 0:
                    # Properties found - show count and basic info
                    properties = structured_result["properties"]
                    response = f"🏠 Found {len(properties)} properties matching your criteria:\n"
                    for i, prop in enumerate(properties[:3], 1):  # Show first 3
                        if isinstance(prop, dict):
                            title = prop.get('title', 'Property')
                            location = prop.get('location', 'Unknown location')
                            price = prop.get('price', 0)
                            response += f"{i}. {title} - {location} - ₹{price:,}\n"
                    
                    if len(properties) > 3:
                        response += f"... and {len(properties) - 3} more properties"
                        
                else:
                    response = f" Query processed successfully using {len(state['active_agents'])} agents"
            else:
                response = "❌ Unable to process query. Please try again."
            
            state["final_response"] = response
            
            logger.info(" Final response generated")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["final_response"] = f"❌ Error generating response: {e}"
        
        return state

    # === PUBLIC INTERFACE ===

    async def process_query(self, query: str, user_id: str = "default", search_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries through LangGraph orchestration
        
        Args:
            query: User's natural language query
            user_id: Unique user identifier
            search_results: Optional existing search results
            
        Returns:
            Comprehensive response with agent results and metadata
        """
        logger.info(f"Processing query through LangGraph orchestration: '{query[:50]}...'")
        
        # Initialize state
        initial_state: AgentWorkflowState = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "user_id": user_id,
            "search_results": search_results,
            "query_intent": "",
            "query_complexity": "",
            "required_agents": [],
            "agent_priorities": {},
            "active_agents": [],
            "agent_results": {},
            "agent_errors": {},
            "execution_metadata": {},
            "synthesized_result": None,
            "final_response": "",
            "confidence_score": 0.0
        }
        
        try:
            # Execute LangGraph workflow
            start_time = time.time()
            
            result_state = await self.app.ainvoke(initial_state)
            
            execution_time = time.time() - start_time
            
            # Prepare final response
            final_result = {
                "success": True,
                "query": query,
                "final_response": result_state["final_response"],
                "confidence_score": result_state["confidence_score"],
                "execution_metadata": {
                    **result_state["execution_metadata"],
                    "execution_time_seconds": execution_time,
                    "active_agents": result_state["active_agents"],
                    "agent_results_count": len(result_state["agent_results"]),
                    "agent_errors_count": len(result_state["agent_errors"])
                },
                "agent_results": result_state["agent_results"],
                "synthesized_result": result_state["synthesized_result"]
            }
            
            logger.info(f" LangGraph orchestration completed in {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"LangGraph orchestration failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "final_response": f"❌ Processing failed: {e}"
            }

    def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive overview of all available agents and their capabilities"""
        return {
            "query_router": {
                "purpose": "Intelligent query routing and intent detection",
                "capabilities": ["intent_classification", "complexity_analysis", "routing_decisions"],
                "typical_queries": ["route queries to appropriate agents"]
            },
            "structured_data": {
                "purpose": "Database search and property filtering",
                "capabilities": ["property_search", "filtering", "sorting", "data_retrieval"],
                "typical_queries": ["find 3BHK apartments", "properties under 1 crore"]
            },
            "rag": {
                "purpose": "Semantic search and knowledge retrieval",
                "capabilities": ["semantic_search", "knowledge_retrieval", "context_understanding"],
                "typical_queries": ["explain market trends", "what affects property prices"]
            },
            "web_research": {
                "purpose": "Real-time market research and current information",
                "capabilities": ["market_trends", "current_prices", "news_analysis", "external_data"],
                "typical_queries": ["current market trends", "latest property news"]
            },
            "report_generation": {
                "purpose": "User preference analysis and comprehensive reports",
                "capabilities": ["preference_analysis", "market_reports", "investment_analysis", "pdf_export"],
                "typical_queries": ["generate market report", "analyze my preferences"]
            },
            "renovation_estimation": {
                "purpose": "BHK-wise renovation cost calculations",
                "capabilities": ["cost_estimation", "bhk_pricing", "timeline_calculation", "breakdown_analysis"],
                "typical_queries": ["renovation cost for 2BHK", "estimate renovation budget"]
            }
        }


# === CONVENIENCE FUNCTIONS ===

def create_orchestrator() -> LangGraphRealEstateOrchestrator:
    """Factory function to create and return a configured orchestrator"""
    return LangGraphRealEstateOrchestrator()


async def process_real_estate_query(query: str, user_id: str = "default", search_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to process a real estate query using LangGraph orchestration
    
    Args:
        query: User's natural language query
        user_id: Unique user identifier  
        search_results: Optional existing search results
        
    Returns:
        Comprehensive response with agent results
    """
    orchestrator = create_orchestrator()
    return await orchestrator.process_query(query, user_id, search_results)


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = create_orchestrator()
        
        # Test different types of queries
        test_queries = [
            "What would it cost to renovate a 3BHK apartment?",
            "Generate a market analysis report for Mumbai properties",
            "Find 2BHK apartments under 80 lakhs in Pune",
            "Analyze my property preferences and generate insights"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing Query: {query}")
            result = await orchestrator.process_query(query)
            print(f"Response: {result['final_response']}")
            print(f"Agents Used: {result['execution_metadata']['active_agents']}")
    
    # Run the example
    asyncio.run(main())