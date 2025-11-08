from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import asyncio
import logging
import sys
import os
from datetime import datetime
import traceback

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all agents
try:
    from agents.langgraph_orchestrator import LangGraphRealEstateOrchestrator
    from agents.query_router import QueryRouterAgent
    from agents.structured_data_agent import StructuredDataAgent
    from agents.rag_agent import RAGAgent
    from agents.web_research_agent import WebResearchAgent
    from agents.report_generation_agent import ReportGenerationAgent
    from agents.renovation_estimation_agent import RenovationEstimationAgent
    from agents.planner_agent import PlannerAgent
    from agents.memory_enhanced_planner import MemoryEnhancedPlannerAgent
    from components.memory_component import MemoryComponent
except ImportError as e:
    logger.error(f"Failed to import agents: {e}")
    LangGraphRealEstateOrchestrator = None

# Models
class QueryRequest(BaseModel):
    query: str

class PropertyResponse(BaseModel):
    success: bool
    properties: List[Dict[str, Any]]
    response_text: str
    agents_used: List[str]
    execution_time: float
    agent_details: Dict[str, Any] = {}

# Global variables for orchestrator and agents
orchestrator = None
all_agents = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all agents and orchestrator on startup"""
    global orchestrator, all_agents
    
    try:
        logger.info("🚀 Initializing All Agents and Orchestrator...")
        
        # Initialize individual agents first
        logger.info("🤖 Initializing Individual Agents...")
        
        # 1. Memory Component (required by other agents)
        logger.info("  🧠 Initializing Memory Component...")
        memory_component = MemoryComponent()
        all_agents["memory_component"] = memory_component
        
        # 2. Query Router Agent
        logger.info("  🔍 Initializing Query Router Agent...")
        query_router = QueryRouterAgent()
        all_agents["query_router"] = query_router
        
        # 3. Structured Data Agent  
        logger.info("  🗃️ Initializing Structured Data Agent...")
        structured_agent = StructuredDataAgent()
        all_agents["structured_data"] = structured_agent
        
        # 4. RAG Agent
        logger.info("  🧠 Initializing RAG Agent...")
        rag_agent = RAGAgent()
        all_agents["rag"] = rag_agent
        
        # 5. Web Research Agent
        logger.info("  🌐 Initializing Web Research Agent...")
        web_research_agent = WebResearchAgent()
        all_agents["web_research"] = web_research_agent
        
        # 6. Report Generation Agent
        logger.info("  📊 Initializing Report Generation Agent...")
        report_agent = ReportGenerationAgent()
        all_agents["report_generation"] = report_agent
        
        # 7. Renovation Estimation Agent
        logger.info("  🔨 Initializing Renovation Estimation Agent...")
        renovation_agent = RenovationEstimationAgent()
        all_agents["renovation_estimation"] = renovation_agent
        
        # 8. Planner Agent
        logger.info("  📋 Initializing Planner Agent...")
        planner_agent = PlannerAgent()
        all_agents["planner"] = planner_agent
        
        # 9. Memory Enhanced Planner Agent
        logger.info("  🧠📋 Initializing Memory Enhanced Planner Agent...")
        memory_planner = MemoryEnhancedPlannerAgent()
        all_agents["memory_enhanced_planner"] = memory_planner
        
        logger.info(f"✅ Successfully initialized {len(all_agents)} individual agents!")
        
        # Initialize LangGraph Orchestrator (coordinates all agents)
        logger.info("🔀 Initializing LangGraph Orchestrator...")
        orchestrator = LangGraphRealEstateOrchestrator()
        
        logger.info("✅ LangGraph Orchestrator initialized successfully")
        logger.info(f"🎯 Complete System Ready: {len(all_agents)} Agents + Orchestrator")
        
        # Log all available agents
        agent_names = list(all_agents.keys())
        logger.info(f"🤖 Available Agents: {', '.join(agent_names)}")
        
        # Test orchestrator
        test_result = await orchestrator.process_query("test query", user_id="test")
        logger.info(f"✅ Orchestrator test successful: {test_result.get('final_response', 'OK')[:50]}...")
        
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")
        traceback.print_exc()
        # Don't fail startup - we can still serve basic responses
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down all agents and orchestrator...")
    all_agents.clear()

app = FastAPI(
    title="Real Estate AI Search Engine",
    description="Multi-agent property search system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_model=PropertyResponse)
async def search_properties(request: QueryRequest):
    """Main property search using LangGraph Orchestrator"""
    start_time = datetime.now()
    
    try:
        query = request.query.strip()
        logger.info(f"🔍 Processing query: {query}")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Use LangGraph Orchestrator - it handles all agent routing automatically
        if not orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not available")
        
        logger.info("🤖 Using LangGraph Orchestrator (handles all agents)")
        
        # Call orchestrator - it will route to appropriate agents automatically
        result = await orchestrator.process_query(query, user_id="api_user")
        
        # Extract results from orchestrator with detailed logging
        properties = extract_properties_from_orchestrator_result(result)
        
        # Better agent tracking
        agents_used = []
        agent_details = result.get("agent_results", {})
        
        # Track which agents actually executed
        for agent_name, agent_result in agent_details.items():
            if agent_result and agent_result.get("success", False):
                agents_used.append(agent_name)
        
        # Also check active_agents if available
        if result.get("active_agents"):
            agents_used.extend(result.get("active_agents", []))
        
        # Remove duplicates
        agents_used = list(set(agents_used))
        
        logger.info(f"📊 Agents executed: {agents_used}")
        logger.info(f"📊 Agent results: {list(agent_details.keys())}")
        
        # Generate response
        response_text = result.get("final_response", generate_response_text(properties, query, "orchestrator"))
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Orchestrator completed: {len(properties)} results, agents: {agents_used}")
        
        return PropertyResponse(
            success=True,
            properties=properties,
            response_text=response_text,
            agents_used=agents_used,
            execution_time=execution_time,
            agent_details=agent_details
        )
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        traceback.print_exc()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PropertyResponse(
            success=False,
            properties=[],
            response_text=f"Search failed: {str(e)}. Please try a different query.",
            agents_used=[],
            execution_time=execution_time
        )

@app.post("/renovation-estimate")
async def get_renovation_estimate(request: dict):
    """Dedicated renovation estimation endpoint using orchestrator"""
    try:
        # Format as renovation query for orchestrator
        query = f"Estimate renovation cost for {request.get('bedrooms', 2)}BHK {request.get('property_type', 'apartment')} of {request.get('size_sqft', 1200)} sqft with {request.get('level', 'standard')} quality"
        
        result = await orchestrator.process_query(query, user_id="api_user")
        
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/agents/status")
async def agents_status():
    """Check orchestrator and individual agent status"""
    global all_agents, orchestrator
    
    if orchestrator and all_agents:
        # Check each individual agent status
        agent_statuses = {}
        
        for agent_name, agent_instance in all_agents.items():
            try:
                # Check if agent is properly initialized
                if agent_instance:
                    agent_statuses[agent_name] = "✅ Available"
                else:
                    agent_statuses[agent_name] = "❌ Not Available"
            except:
                agent_statuses[agent_name] = "❌ Error"
        
        return {
            "orchestrator": "✅ Available",
            "agents": agent_statuses,
            "total_agents": len(agent_statuses),
            "active_agents": len([s for s in agent_statuses.values() if "✅" in s]),
            "agent_details": {
                "query_router": "Intent detection and routing",
                "structured_data": "Database search and filtering", 
                "rag": "Semantic search and knowledge retrieval",
                "web_research": "Real-time market research",
                "report_generation": "Reports and analysis generation",
                "renovation_estimation": "Cost estimation and calculations",
                "planner": "Task planning and coordination",
                "memory_enhanced_planner": "Advanced planning with memory",
                "memory_component": "Session and user memory management"
            }
        }
    else:
        return {
            "orchestrator": "❌ Not Available",
            "agents": {},
            "total_agents": 0,
            "active_agents": 0,
            "error": "Orchestrator or agents not initialized"
        }
@app.get("/agents/list")
async def list_agents():
    """List all available agents with details"""
    global all_agents
    
    agent_list = []
    for agent_name, agent_instance in all_agents.items():
        agent_info = {
            "name": agent_name,
            "status": "✅ Available" if agent_instance else "❌ Not Available",
            "type": type(agent_instance).__name__ if agent_instance else "Unknown",
            "description": get_agent_description(agent_name)
        }
        agent_list.append(agent_info)
    
    return {
        "total_agents": len(agent_list),
        "agents": agent_list,
        "orchestrator_available": bool(orchestrator)
    }

@app.post("/agents/{agent_name}/query")
async def query_individual_agent(agent_name: str, request: QueryRequest):
    """Query a specific agent directly"""
    global all_agents
    
    if agent_name not in all_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    agent = all_agents[agent_name]
    if not agent:
        raise HTTPException(status_code=500, detail=f"Agent '{agent_name}' not available")
    
    try:
        query = request.query
        logger.info(f"🎯 Direct query to {agent_name}: {query}")
        
        # Handle different agent types
        if agent_name == "query_router":
            result = agent.route_query(query)
        elif agent_name == "structured_data":
            result = agent.search_properties({"query": query})
        elif agent_name == "rag":
            result = agent.semantic_search({"query": query})
        elif agent_name == "web_research":
            result = await agent.research_query(query)
        elif agent_name == "report_generation":
            result = agent.generate_report({"query": query, "report_type": "custom"})
        elif agent_name == "renovation_estimation":
            result = agent.estimate_renovation_cost()
        elif agent_name == "planner":
            result = await agent.create_plan(query)
        elif agent_name == "memory_enhanced_planner":
            result = await agent.plan_and_execute(query, "direct_user")
        else:
            result = {"error": f"Direct query not supported for {agent_name}"}
        
        return {
            "success": True,
            "agent_name": agent_name,
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Direct agent query failed for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent query failed: {str(e)}")

def get_agent_description(agent_name: str) -> str:
    """Get description for each agent"""
    descriptions = {
        "query_router": "Analyzes user queries and routes them to appropriate agents",
        "structured_data": "Searches property database using SQL queries and filters",
        "rag": "Performs semantic search using vector embeddings and knowledge retrieval",
        "web_research": "Conducts real-time web research for market data and trends",
        "report_generation": "Creates comprehensive reports, charts, and PDF documents",
        "renovation_estimation": "Calculates renovation costs and provides estimates",
        "planner": "Creates execution plans and coordinates multi-step tasks",
        "memory_enhanced_planner": "Advanced planning with user memory and preferences",
        "memory_component": "Manages user sessions, preferences, and conversation history"
    }
    return descriptions.get(agent_name, "Specialized AI agent for real estate tasks")

@app.post("/generate-report")
async def generate_report(request: dict):
    """Generate detailed reports with charts and PDFs"""
    try:
        report_type = request.get("report_type", "market_analysis")
        location = request.get("location", "")
        include_charts = request.get("include_charts", True)
        include_pdf = request.get("include_pdf", True)
        
        # Create enhanced query for report generation
        query = f"generate comprehensive {report_type} report for {location}"
        if include_charts:
            query += " with charts and visualizations"
        if include_pdf:
            query += " in PDF format"
        
        logger.info(f"📊 Generating report: {report_type} for {location}")
        
        # Use orchestrator for report generation
        if not orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not available")
        
        result = await orchestrator.process_query(query, user_id="report_user")
        
        # Extract and format results
        properties = extract_properties_from_orchestrator_result(result)
        agents_used = []
        agent_details = result.get("agent_results", {})
        
        for agent_name, agent_result in agent_details.items():
            if agent_result and agent_result.get("success", False):
                agents_used.append(agent_name)
        
        # Generate report response
        response_text = result.get("final_response", "Report generated successfully")
        
        # Add chart data (sample data for now)
        chart_data = generate_sample_chart_data(location)
        
        # Generate PDF data (base64 encoded)
        pdf_data = None
        if include_pdf:
            pdf_data = generate_pdf_base64(report_type, location, response_text)
        
        return {
            "success": True,
            "report_type": report_type,
            "location": location,
            "response_text": response_text,
            "agents_used": agents_used,
            "properties": properties,
            "chart_data": chart_data if include_charts else None,
            "pdf_data": pdf_data if include_pdf else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/search-history-analysis")
async def search_history_analysis():
    """Analyze search history and generate insights"""
    try:
        # In a real implementation, this would fetch from database
        # For now, return sample analysis
        
        analysis_data = {
            "total_searches": 127,
            "unique_queries": 89,
            "success_rate": 0.85,
            "popular_locations": [
                {"location": "Mumbai", "count": 45},
                {"location": "Bangalore", "count": 32},
                {"location": "Delhi", "count": 28},
                {"location": "Pune", "count": 22}
            ],
            "property_types": [
                {"type": "2BHK", "count": 56},
                {"type": "3BHK", "count": 41},
                {"type": "1BHK", "count": 23},
                {"type": "4BHK", "count": 7}
            ],
            "budget_distribution": [
                {"range": "<50L", "count": 25},
                {"range": "50-100L", "count": 48},
                {"range": "100-150L", "count": 35},
                {"range": ">150L", "count": 19}
            ],
            "search_trends": generate_search_trends_data(),
            "insights": [
                "Most searches happen between 10 AM - 2 PM",
                "2BHK apartments are the most searched property type",
                "Mumbai and Bangalore account for 60% of all searches",
                "Average budget has increased by 12% this quarter"
            ]
        }
        
        return {
            "success": True,
            "analysis": analysis_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Search history analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_sample_chart_data(location=""):
    """Generate sample chart data"""
    return {
        "price_trends": {
            "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "prices": [85, 87, 89, 88, 90, 92]
        },
        "property_distribution": {
            "types": ["2BHK", "3BHK", "1BHK", "4BHK"],
            "counts": [45, 35, 15, 5]
        },
        "location_comparison": {
            "locations": ["Mumbai", "Bangalore", "Delhi", "Pune"],
            "avg_prices": [95, 75, 85, 65],
            "growth_rates": [8, 12, 6, 15]
        }
    }

def generate_search_trends_data():
    """Generate search trends data"""
    from datetime import datetime, timedelta
    import random
    
    # Generate last 30 days of data
    dates = []
    searches = []
    
    for i in range(30):
        date = datetime.now() - timedelta(days=i)
        dates.insert(0, date.strftime('%Y-%m-%d'))
        searches.insert(0, random.randint(15, 45))
    
    return {
        "dates": dates,
        "searches": searches
    }

def generate_pdf_base64(report_type, location, content):
    """Generate PDF report as base64 string"""
    try:
        # Simple PDF content (in real implementation, use reportlab)
        pdf_content = f"""
PDF Report: {report_type}
Location: {location}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{content}

This is a sample PDF report. In a full implementation, this would include:
- Professional formatting
- Charts and graphs
- Property listings with images
- Market analysis
- Investment recommendations
        """
        
        import base64
        return base64.b64encode(pdf_content.encode()).decode()
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "orchestrator_available": orchestrator is not None,
        "managed_agents": 6 if orchestrator else 0
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🏠 Real Estate AI Search Engine",
        "status": "running",
        "orchestrator_available": orchestrator is not None,
        "managed_agents": 6 if orchestrator else 0,
        "docs": "/docs"
    }

# Helper Functions
def extract_properties_from_orchestrator_result(result: Dict) -> List[Dict]:
    """Extract properties from orchestrator result"""
    properties = []
    
    agent_results = result.get("agent_results", {})
    
    # From structured data agent
    if "structured_data" in agent_results:
        structured_props = agent_results["structured_data"].get("properties", [])
        for prop in structured_props:
            if isinstance(prop, dict):
                properties.append(prop)
    
    # From RAG agent
    if "rag" in agent_results:
        rag_result = agent_results["rag"]
        if isinstance(rag_result, dict):
            rag_props = rag_result.get("properties", [])
            for prop in rag_props:
                if isinstance(prop, dict):
                    properties.append(prop)
    
    # From renovation agent
    if "renovation_estimation" in agent_results:
        renovation_result = agent_results["renovation_estimation"]
        if isinstance(renovation_result, dict):
            properties.append({
                "id": "renovation_estimate",
                "title": "Renovation Cost Estimate",
                "price": renovation_result.get("total_cost", 0),
                "source": "Renovation_Agent",
                "renovation_details": renovation_result
            })
    
    return properties[:20]  # Limit to 20 results

def generate_response_text(properties: List[Dict], query: str, method: str) -> str:
    """Generate user-friendly response text"""
    if not properties:
        return f"🔍 No properties found for '{query}'. The AI agents searched our database but couldn't find matching properties. Try different keywords or broader search terms."
    
    count = len(properties)
    
    # Check if this is a renovation estimate
    if properties and properties[0].get("source") == "Renovation_Agent":
        estimate = properties[0]
        cost = estimate.get("price", 0)
        return f"🔨 Renovation Cost Estimate for '{query}': ₹{cost:,}\n\nThis estimate was calculated by our AI Renovation Agent based on your requirements."
    
    # Regular property search results
    response = f"🏠 Found {count} propert{'y' if count == 1 else 'ies'} matching '{query}'\n\n"
    response += f"🤖 Search performed by: LangGraph Multi-Agent System\n\n"
    
    # Show top 3 properties
    for i, prop in enumerate(properties[:3], 1):
        title = prop.get("title", "Property")
        price = prop.get("price", 0)
        location = prop.get("location", "Location not specified")
        
        response += f"**{i}. {title}**\n"
        response += f"📍 {location}\n"
        if price > 0:
            response += f"💰 ₹{price:,}\n"
        
        if prop.get("property_size_sqft"):
            response += f"📐 {prop['property_size_sqft']} sqft\n"
        if prop.get("num_rooms"):
            response += f"🏠 {prop['num_rooms']} rooms\n"
        
        response += "\n"
    
    if count > 3:
        response += f"... and {count - 3} more properties available.\n"
    
    return response