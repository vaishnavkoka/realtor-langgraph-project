"""
Structured Data Agent - PostgreSQL Database Queries
Converts natural language to SQL and executes property searches using actual database schema
"""

from langchain_core.messages import HumanMessage
from langchain.tools import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Optional, Any
import json
import logging
import time
from datetime import datetime
from models.free_models import get_primary_llm
from models.rate_limiter import rate_limiter
from config.settings import DatabaseConfig
import sys
import os

# Add src directory to path to import database schema
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from database_schema import Property, Certificate, SearchLog, get_session, get_database_url

logger = logging.getLogger(__name__)

class StructuredDataAgent:
    """
    LangChain-based agent for structured property database queries
    Converts natural language filters to SQL and executes searches
    """
    
    def __init__(self):
        self.llm = get_primary_llm()
        
        # Initialize database connection using your actual database schema
        db_url = get_database_url('sqlite')  # Use sqlite for development
        self.session = get_session(db_url)
        
        # Create LangChain tools
        self.search_tool = Tool(
            name="search_properties",
            description="Search properties in database based on criteria",
            func=self.search_properties_tool
        )
        
        self.filter_tool = Tool(
            name="build_filters",
            description="Build SQL filters from natural language",
            func=self.build_filters_tool
        )
    
    def search_properties_tool(self, criteria: str) -> str:
        """LangChain tool wrapper for property search"""
        result = self.search_properties(json.loads(criteria) if criteria.startswith('{') else {"query": criteria})
        return json.dumps(result)
    
    def build_filters_tool(self, query: str) -> str:
        """LangChain tool wrapper for filter building"""
        result = self.build_sql_filters(query)
        return json.dumps(result)
    
    def search_properties(self, search_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main property search function
        """
        try:
            # Extract criteria
            entities = search_criteria.get("extracted_entities", {})
            query_text = search_criteria.get("query", "")
            
            # Build SQL query
            sql_query = self._build_property_query(entities, query_text)
            
            # Execute query
            start_time = time.time()
            results = self._execute_query(sql_query)
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log search for analytics
            self._log_search(search_criteria.get("query", ""), len(results), response_time_ms)
            
            # Format results
            formatted_results = self._format_property_results(results)
            
            return {
                "success": True,
                "properties": formatted_results,
                "count": len(formatted_results),
                "sql_query": str(sql_query),
                "search_criteria": entities,
                "agent": "StructuredDataAgent"
            }
            
        except Exception as e:
            logger.error(f"Property search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "properties": [],
                "count": 0,
                "agent": "StructuredDataAgent"
            }
    
    def _build_property_query(self, entities: Dict[str, Any], query_text: str = ""):
        """Build SQLAlchemy query from extracted entities using actual database schema"""
        
        # Start with base query
        query = self.session.query(Property)
        
        # Apply filters based on entities using actual field names
        if entities.get("location"):
            location = entities["location"]
            query = query.filter(
                Property.location.ilike(f"%{location}%") |
                Property.city.ilike(f"%{location}%") |
                Property.title.ilike(f"%{location}%")
            )
        
        if entities.get("budget"):
            budget = entities["budget"]
            # Assuming budget is maximum price
            query = query.filter(Property.price <= budget)
        
        if entities.get("bedrooms") or entities.get("rooms"):
            # Map bedrooms to num_rooms in your schema
            rooms = entities.get("bedrooms", entities.get("rooms"))
            query = query.filter(Property.num_rooms == rooms)
        
        if entities.get("property_type"):
            prop_type = entities["property_type"]
            # Search in title and long_description for property type
            query = query.filter(
                Property.title.ilike(f"%{prop_type}%") |
                Property.long_description.ilike(f"%{prop_type}%")
            )
        
        if entities.get("min_area") or entities.get("max_area"):
            if entities.get("min_area"):
                query = query.filter(Property.property_size_sqft >= entities["min_area"])
            if entities.get("max_area"):
                query = query.filter(Property.property_size_sqft <= entities["max_area"])
        
        # Filter by seller type if specified
        if entities.get("seller_type"):
            query = query.filter(Property.seller_type.ilike(f"%{entities['seller_type']}%"))
        
        # Order by relevance (price, then latest listing)
        query = query.order_by(Property.price.asc(), Property.listing_date.desc())
        
        # Limit results to avoid overwhelming response
        query = query.limit(20)
        
        return query
    
    def _execute_query(self, query) -> List:
        """Execute the SQLAlchemy query safely"""
        try:
            results = query.all()
            logger.info(f"Found {len(results)} properties matching criteria")
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def _format_property_results(self, properties: List) -> List[Dict[str, Any]]:
        """Format property objects into JSON-serializable format using actual schema fields"""
        formatted = []
        
        for prop in properties:
            formatted.append({
                "id": prop.id,
                "property_id": prop.property_id,
                "title": prop.title,
                "description": prop.long_description,
                "price": prop.price,
                "location": prop.location,
                "city": prop.city,
                "num_rooms": prop.num_rooms,
                "property_size_sqft": prop.property_size_sqft,
                "seller_type": prop.seller_type,
                "listing_date": prop.listing_date.isoformat() if prop.listing_date else None,
                "certificates": prop.certificates.split(',') if prop.certificates else [],
                "seller_contact": prop.seller_contact,
                "metadata_tags": prop.metadata_tags.split(',') if prop.metadata_tags else [],
                "price_per_sqft": round(prop.price / prop.property_size_sqft, 2) if prop.property_size_sqft else None,
                "created_at": prop.created_at.isoformat() if prop.created_at else None,
                "updated_at": prop.updated_at.isoformat() if prop.updated_at else None,
                "is_indexed": prop.is_indexed
            })
        
        return formatted
    
    def _log_search(self, query: str, results_count: int, response_time_ms: float):
        """Log search query for analytics using SearchLog table"""
        try:
            search_log = SearchLog(
                query=query,
                search_type="structured",
                results_count=results_count,
                user_session="anonymous",  # TODO: Implement session tracking
                response_time_ms=response_time_ms
            )
            self.session.add(search_log)
            self.session.commit()
            logger.debug(f"Logged search: {query} -> {results_count} results")
        except Exception as e:
            logger.error(f"Failed to log search: {e}")
            # Don't fail the search because of logging issues
            self.session.rollback()
    
    def build_sql_filters(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Use LLM to convert natural language to SQL filters
        """
        # Check rate limits
        if not rate_limiter.can_make_request("groq"):
            return self._fallback_filter_building(natural_language_query)
        
        try:
            filters = self._llm_based_filter_building(natural_language_query)
            rate_limiter.record_request("groq")
            return filters
        except Exception as e:
            logger.error(f"LLM filter building failed: {e}")
            return self._fallback_filter_building(natural_language_query)
    
    def _llm_based_filter_building(self, query: str) -> Dict[str, Any]:
        """Use LLM to intelligently build SQL filters based on actual database schema"""
        
        filter_prompt = f"""
        Convert this natural language property search query into structured filters for our real estate database.

        User Query: "{query}"

        Database Schema - Available property fields:
        - property_id (string): unique property identifier
        - num_rooms (integer): number of rooms (bedrooms)
        - property_size_sqft (integer): property size in square feet
        - title (string): property title/headline
        - long_description (text): detailed property description
        - city (string): city name
        - location (string): specific area/neighborhood within city
        - price (integer): property price in rupees
        - seller_type (string): owner, agent, builder
        - listing_date (datetime): when property was listed
        - certificates (string): comma-separated certificate files
        - seller_contact (string): contact phone number
        - metadata_tags (string): comma-separated tags

        Return JSON with extracted filters:
        {{
            "location_filters": {{
                "city": "city_name or null",
                "location_contains": ["area1", "area2"] or null,
                "location_exact": "specific_location or null"
            }},
            "price_filters": {{
                "min_price": number or null,
                "max_price": number or null,
                "budget_range": [min, max] or null
            }},
            "property_filters": {{
                "num_rooms": number or null,
                "min_size_sqft": number or null,
                "max_size_sqft": number or null,
                "seller_type": "owner|agent|builder or null"
            }},
            "text_search": {{
                "title_contains": ["keyword1", "keyword2"] or null,
                "description_contains": ["keyword1", "keyword2"] or null
            }},
            "date_filters": {{
                "listed_after": "YYYY-MM-DD or null",
                "listed_before": "YYYY-MM-DD or null"
            }},
            "sorting_preference": {{
                "sort_by": "price|size|date|relevance",
                "order": "asc|desc"
            }},
            "limit": number or 20
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=filter_prompt)])
        
        try:
            filters = json.loads(response.content)
            filters["success"] = True
            filters["method"] = "llm"
            return filters
        except json.JSONDecodeError:
            return self._fallback_filter_building(query)
    
    def _fallback_filter_building(self, query: str) -> Dict[str, Any]:
        """Simple pattern-based filter building"""
        
        filters = {
            "location_filters": {},
            "price_filters": {},
            "property_filters": {},
            "amenity_filters": [],
            "other_filters": {},
            "success": True,
            "method": "pattern"
        }
        
        # Extract basic patterns (reuse from query router)
        query_lower = query.lower()
        
        # Price extraction
        import re
        price_patterns = [
            (r'under (\d+) lakh', lambda x: {"max_price": int(x) * 100000}),
            (r'under (\d+) crore', lambda x: {"max_price": int(x) * 10000000}),
            (r'(\d+) to (\d+) lakh', lambda x, y: {"min_price": int(x) * 100000, "max_price": int(y) * 100000}),
        ]
        
        for pattern, extractor in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters["price_filters"] = extractor(*match.groups())
                break
        
        return filters
    
    def get_advanced_search_suggestions(self, current_criteria: Dict[str, Any]) -> List[str]:
        """Suggest refinements based on current search criteria and actual database schema"""
        
        suggestions = []
        entities = current_criteria.get("extracted_entities", {})
        
        if not entities.get("location") and not entities.get("city"):
            suggestions.append(" Try specifying a location (e.g., 'in Bangalore', 'Whitefield area', 'HSR Layout')")
        
        if not entities.get("budget"):
            suggestions.append(" Add a budget range (e.g., 'under 50 lakhs', 'between 30-40 lakhs')")
        
        if not entities.get("num_rooms") and not entities.get("bedrooms"):
            suggestions.append(" Specify number of rooms (e.g., '2 rooms', '3BHK', '4 bedroom')")
        
        if not entities.get("property_size_sqft"):
            suggestions.append(" Add size requirement (e.g., 'more than 1000 sqft', 'under 1500 square feet')")
        
        if not entities.get("seller_type"):
            suggestions.append(" Filter by seller type: 'by owner', 'from agent', 'builder properties'")
        
        # Additional suggestions based on available data
        suggestions.extend([
            " Search in descriptions: 'with parking', 'furnished apartment', 'ready to move'",
            " Filter by listing date: 'recently listed', 'posted this month'",
            " Properties with certificates: 'verified documents', 'with certificates'",
            " Direct owner contact: 'owner properties', 'no broker'"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def search_properties_with_certificates(self, search_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Search properties that have certificate documents"""
        try:
            # Get base search results
            base_result = self.search_properties(search_criteria)
            
            if not base_result["success"]:
                return base_result
            
            # Filter properties that have certificates
            properties_with_certs = []
            
            for prop in base_result["properties"]:
                if prop["certificates"]:  # Has certificates
                    # Get certificate details
                    cert_details = self._get_certificate_details(prop["property_id"])
                    prop["certificate_details"] = cert_details
                    properties_with_certs.append(prop)
            
            return {
                "success": True,
                "properties": properties_with_certs,
                "count": len(properties_with_certs),
                "total_properties_searched": base_result["count"],
                "certificate_availability": f"{len(properties_with_certs)}/{base_result['count']}",
                "agent": "StructuredDataAgent"
            }
            
        except Exception as e:
            logger.error(f"Certificate search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "properties": [],
                "count": 0,
                "agent": "StructuredDataAgent"
            }
    
    def _get_certificate_details(self, property_id: str) -> List[Dict[str, Any]]:
        """Get certificate details for a property"""
        try:
            certificates = self.session.query(Certificate).filter(
                Certificate.property_id == property_id
            ).all()
            
            cert_details = []
            for cert in certificates:
                cert_details.append({
                    "filename": cert.filename,
                    "file_path": cert.file_path,
                    "file_size": cert.file_size,
                    "is_processed": cert.is_processed,
                    "processed_at": cert.processed_at.isoformat() if cert.processed_at else None,
                    "has_text_content": bool(cert.extracted_text)
                })
            
            return cert_details
            
        except Exception as e:
            logger.error(f"Failed to get certificate details: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    agent = StructuredDataAgent()
    
    # Test search with correct field mapping
    test_criteria = {
        "extracted_entities": {
            "location": "Bangalore",
            "budget": 5000000,  # 50 lakhs
            "num_rooms": 2      # Using actual field name
        },
        "query": "Find 2BHK apartment in Bangalore under 50 lakhs"
    }
    
    print(" Testing Structured Data Agent with actual database schema")
    result = agent.search_properties(test_criteria)
    print(f" Found {result['count']} properties")
    
    if result["properties"]:
        first_prop = result['properties'][0]
        print(f" First property: {first_prop['title']}")
        print(f" Price: ₹{first_prop['price']:,}")
        print(f" Location: {first_prop['location']}")
        print(f" Rooms: {first_prop['num_rooms']}")
        print(f" Size: {first_prop['property_size_sqft']} sqft")
        print(f" Seller: {first_prop['seller_type']}")
    else:
        print("ℹ No properties found. Database might be empty.")
        print(" Run the data ingestion script to populate the database first.")