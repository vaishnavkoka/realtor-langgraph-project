"""
Query Router Agent - Intent Detection and Routing
Uses Groq Llama-3.1 for fast intent classification and entity extraction
"""

from langchain_core.messages import HumanMessage
from langchain.tools import Tool
from typing import Dict, List, Optional, Any
import json
import logging
import re
from models.free_models import get_primary_llm
from models.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

class QueryRouterAgent:
    """
    LangChain-based Query Router Agent
    Routes user queries to appropriate agents based on intent detection
    """
    
    def __init__(self):
        self.llm = get_primary_llm()
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Create LangChain tools
        self.routing_tool = Tool(
            name="route_query",
            description="Route user query to appropriate agents",
            func=self.route_query_tool
        )
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize pattern matching for quick intent detection"""
        return {
            "search": [
                "find", "search", "looking for", "show me", "properties",
                "apartment", "house", "flat", "bhk", "bedroom", "budget"
            ],
            "analysis": [
                "compare", "analysis", "market", "trends", "investment",
                "roi", "returns", "growth", "appreciation"
            ],
            "estimation": [
                "cost", "price", "estimate", "renovation", "repair",
                "upgrade", "modernize", "budget for"
            ],
            "report": [
                "report", "summary", "download", "pdf", "document",
                "generate", "create report", "detailed analysis"
            ],
            "research": [
                "market rates", "neighborhood", "amenities", "schools",
                "transportation", "connectivity", "infrastructure"
            ]
        }
    
    def _quick_intent_detection(self, query: str) -> Optional[str]:
        """Fast pattern-based intent detection (fallback)"""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return "search"  # Default to search
    
    def _extract_entities_pattern_based(self, query: str) -> Dict[str, Any]:
        """Pattern-based entity extraction (fallback)"""
        entities = {}
        
        # Extract budget
        budget_patterns = [
            r'(\d+)\s*lakh[s]?',
            r'(\d+)\s*crore[s]?',
            r'under\s*(\d+)',
            r'below\s*(\d+)',
            r'budget\s*(\d+)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount = int(match.group(1))
                if 'lakh' in query.lower():
                    entities['budget'] = amount * 100000
                elif 'crore' in query.lower():
                    entities['budget'] = amount * 10000000
                else:
                    entities['budget'] = amount
                break
        
        # Extract bedrooms
        bedroom_pattern = r'(\d+)\s*bhk|(\d+)\s*bedroom'
        match = re.search(bedroom_pattern, query, re.IGNORECASE)
        if match:
            entities['bedrooms'] = int(match.group(1) or match.group(2))
        
        # Extract locations (based on actual data)
        common_locations = [
            'hyderabad', 'jamshedpur', 'nagpur', 'mumbai', 'delhi', 'bangalore', 
            'chennai', 'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 
            'lucknow', 'kanpur', 'patna', 'bhopal', 'ludhiana', 'agra',
            'jharkhand', 'assam', 'mizoram', 'maharashtra', 'karnataka', 'telangana'
        ]
        
        query_lower = query.lower()
        for location in common_locations:
            if location in query_lower:
                entities['location'] = location.title()
                break
        
        # Extract property type (based on actual data)
        property_types = ['house', 'studio', 'villa', 'apartment', 'flat', 'plot', 'commercial']
        for prop_type in property_types:
            if prop_type in query_lower:
                entities['property_type'] = prop_type
                break
        
        return entities
    
    def route_query_tool(self, query: str) -> str:
        """LangChain tool function for query routing"""
        result = self.route_query(query)
        return json.dumps(result)
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main routing function using LLM with pattern-based fallback
        """
        # Check rate limits
        if not rate_limiter.can_make_request("groq"):
            logger.warning("Groq rate limit reached, using pattern-based routing")
            return self._fallback_routing(query)
        
        # Try LLM-based routing first
        try:
            result = self._llm_based_routing(query)
            rate_limiter.record_request("groq")
            return result
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return self._fallback_routing(query)
    
    def _llm_based_routing(self, query: str) -> Dict[str, Any]:
        """Use LLM for sophisticated intent detection and entity extraction"""
        
        routing_prompt = f"""
        You are a real estate query router. Analyze the user query and return ONLY a JSON response with routing information.

        User Query: "{query}"

        Return ONLY valid JSON in this exact format (no additional text):
        {{
            "intent": "search",
            "confidence": 0.9,
            "complexity": "simple",
            "required_agents": ["StructuredDataAgent"],
            "priority": "medium",
            "extracted_entities": {{
                "location": "extracted_location_or_null",
                "budget": null,
                "property_type": "extracted_type_or_null", 
                "bedrooms": null,
                "bathrooms": null,
                "amenities": null,
                "timeline": null
            }},
            "user_requirements": "brief description",
            "reasoning": "why this routing was chosen"
        }}

        Intent types: search, analysis, estimation, report, research
        Agent types: StructuredDataAgent, RAGAgent, WebResearchAgent
        
        IMPORTANT: Return ONLY the JSON object, no markdown formatting or additional text.
        """
        
        response = self.llm.invoke([HumanMessage(content=routing_prompt)])
        
        try:
            # Clean and parse JSON response
            response_content = response.content.strip()
            
            # Handle empty responses
            if not response_content:
                logger.error("Empty LLM response received")
                return self._fallback_routing(query)
            
            # Try to extract JSON if it's wrapped in other text
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(0)
            
            # Parse JSON response
            routing_info = json.loads(response_content)
            
            # Validate and set defaults
            routing_info.setdefault("intent", "search")
            routing_info.setdefault("confidence", 0.8)
            routing_info.setdefault("complexity", "moderate")
            routing_info.setdefault("priority", "medium")
            routing_info.setdefault("required_agents", ["StructuredDataAgent"])
            routing_info.setdefault("extracted_entities", {})
            
            # Add metadata
            routing_info["success"] = True
            routing_info["method"] = "llm"
            routing_info["original_query"] = query
            
            logger.info(f"LLM routing successful: {routing_info['intent']} with confidence {routing_info['confidence']}")
            return routing_info
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw LLM response: {repr(response.content)}")
            logger.error(f"Cleaned response: {repr(response_content)}")
            return self._fallback_routing(query)
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """Pattern-based fallback routing when LLM is unavailable"""
        
        intent = self._quick_intent_detection(query)
        entities = self._extract_entities_pattern_based(query)
        
        # Map intents to agents
        agent_mapping = {
            "search": ["StructuredDataAgent", "RAGAgent"],
            "analysis": ["StructuredDataAgent", "RAGAgent", "WebResearchAgent"],
            "estimation": ["RenovationEstimationAgent"],
            "report": ["ReportGeneratorAgent"],
            "research": ["WebResearchAgent"]
        }
        
        routing_info = {
            "intent": intent,
            "confidence": 0.7,  # Lower confidence for pattern-based
            "complexity": "simple",
            "required_agents": agent_mapping.get(intent, ["StructuredDataAgent"]),
            "priority": "medium",
            "extracted_entities": entities,
            "user_requirements": f"Pattern-based analysis: {intent} intent detected",
            "reasoning": "Fallback pattern-based routing used",
            "success": True,
            "method": "pattern",
            "original_query": query
        }
        
        logger.info(f"Pattern-based routing: {intent}")
        return routing_info
    
    def get_routing_explanation(self, routing_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of routing decision"""
        
        if not routing_result.get("success"):
            return "❌ Failed to analyze your query. Please try rephrasing."
        
        intent = routing_result.get("intent", "unknown")
        confidence = routing_result.get("confidence", 0)
        agents = routing_result.get("required_agents", [])
        entities = routing_result.get("extracted_entities", {})
        
        explanation = f"🎯 **Intent Detected**: {intent.title()} (confidence: {confidence:.1%})\n\n"
        
        if entities:
            explanation += "📋 **Extracted Information**:\n"
            for key, value in entities.items():
                if value:
                    explanation += f"   • {key.title()}: {value}\n"
            explanation += "\n"
        
        explanation += f"🤖 **Agents Assigned**: {', '.join(agents)}\n\n"
        explanation += f"💭 **Reasoning**: {routing_result.get('reasoning', 'Intent-based routing')}"
        
        return explanation

# Example usage and testing
if __name__ == "__main__":
    # Test the router
    router = QueryRouterAgent()
    
    test_queries = [
        "Find me a 2BHK apartment in Bangalore under 50 lakhs",
        "Compare market rates for properties in Whitefield vs Koramangala",
        "Estimate renovation cost for a 1200 sqft apartment",
        "Generate a property investment report for HSR Layout",
        "What are the amenities in Electronic City?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = router.route_query(query)
        print(f"📊 Result: {result['intent']} -> {result['required_agents']}")
        print(f"📝 Entities: {result['extracted_entities']}")
        print("-" * 50)