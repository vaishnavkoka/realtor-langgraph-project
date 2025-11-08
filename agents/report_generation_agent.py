"""
Report Generation Agent - Comprehensive Real Estate Reports with Visualizations
Generates detailed analysis reports, charts, graphs, and downloadable PDF reports

Features:
1. Market analysis reports with charts
2. Property comparison reports  
3. Investment analysis with ROI calculations
4. Price trend visualizations
5. Location-based analysis
6. PDF export functionality
7. Interactive charts and graphs
8. Custom report templates
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import os
import sys
from dataclasses import dataclass, asdict
import sqlite3
import pandas as pd
import numpy as np

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print("⚠️ Visualization libraries not available. Installing requirements...")

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False
    print("⚠️ PDF generation libraries not available. Installing requirements...")

# LangChain and LLM imports
from langchain_core.messages import HumanMessage
from langchain.tools import Tool

# Project imports
from models.free_models import get_primary_llm
from utils.rate_limiter import groq_rate_limiter, rate_limited

# Import renovation estimation agent
try:
    from agents.renovation_estimation_agent import RenovationEstimationAgent
    RENOVATION_ESTIMATION_AVAILABLE = True
except ImportError:
    RENOVATION_ESTIMATION_AVAILABLE = False
    print("⚠️ Renovation estimation agent not available")

logger = logging.getLogger(__name__)

@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    report_type: str
    title: str
    description: str
    generated_at: datetime
    user_id: str
    query: str
    data_sources: List[str]
    charts_included: List[str]
    file_paths: Dict[str, str]  # {"pdf": "path", "html": "path", etc.}

@dataclass
class PropertyAnalysis:
    """Analysis data for a property"""
    property_id: str
    title: str
    price: float
    location: str
    property_type: str
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    area: Optional[float]
    price_per_sqft: Optional[float]
    roi_estimate: Optional[float]
    rental_yield: Optional[float]
    appreciation_potential: str

@dataclass
class MarketTrends:
    """Market trend data"""
    location: str
    avg_price: float
    price_change_6m: float
    price_change_1y: float
    inventory_count: int
    demand_score: float
    supply_score: float
    trend_direction: str

class ReportGenerationAgent:
    """
    Comprehensive Report Generation Agent for Real Estate Analysis
    Generates detailed reports with visualizations and PDF export
    """
    
    def __init__(self):
        self.llm = get_primary_llm()
        self.reports_dir = self._setup_reports_directory()
        self.charts_dir = os.path.join(self.reports_dir, "charts")
        self.db_path = "realestate.db"
        
        # Initialize renovation estimation agent
        if RENOVATION_ESTIMATION_AVAILABLE:
            self.renovation_agent = RenovationEstimationAgent()
        else:
            self.renovation_agent = None
        
        # Report templates
        self.report_templates = self._initialize_report_templates()
        
        # Chart configurations
        self.chart_configs = self._initialize_chart_configs()
        
        # Create tools
        self.tools = self._create_tools()
        
        logger.info("🏠📊 Report Generation Agent initialized")

    def _setup_reports_directory(self) -> str:
        """Setup directory structure for reports"""
        base_dir = "generated_reports"
        subdirs = ["pdf", "charts", "html", "data"]
        
        os.makedirs(base_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        
        return base_dir

    def _initialize_report_templates(self) -> Dict[str, Dict]:
        """Initialize report templates"""
        return {
            "market_analysis": {
                "sections": [
                    "executive_summary",
                    "market_overview", 
                    "price_analysis",
                    "trend_analysis",
                    "investment_outlook",
                    "recommendations"
                ],
                "charts": ["price_trends", "location_comparison", "property_distribution"]
            },
            "property_comparison": {
                "sections": [
                    "comparison_overview",
                    "property_details",
                    "price_analysis",
                    "location_analysis", 
                    "investment_potential",
                    "final_recommendation"
                ],
                "charts": ["price_comparison", "feature_comparison", "roi_analysis"]
            },
            "investment_analysis": {
                "sections": [
                    "investment_summary",
                    "roi_calculations",
                    "risk_assessment",
                    "market_positioning",
                    "cash_flow_projections",
                    "exit_strategies"
                ],
                "charts": ["roi_projection", "cash_flow", "risk_return_matrix"]
            },
            "location_report": {
                "sections": [
                    "location_overview",
                    "infrastructure_analysis",
                    "price_trends",
                    "demand_supply",
                    "growth_prospects",
                    "investment_rating"
                ],
                "charts": ["price_heatmap", "growth_trends", "amenity_scores"]
            }
        }

    def _initialize_chart_configs(self) -> Dict[str, Dict]:
        """Initialize chart configuration templates"""
        return {
            "price_trends": {
                "type": "line",
                "title": "Price Trends Over Time",
                "x_axis": "Time Period",
                "y_axis": "Average Price (₹)",
                "color_scheme": "viridis"
            },
            "location_comparison": {
                "type": "bar",
                "title": "Average Prices by Location",
                "x_axis": "Location",
                "y_axis": "Average Price (₹)",
                "color_scheme": "Set2"
            },
            "property_distribution": {
                "type": "pie",
                "title": "Property Type Distribution",
                "color_scheme": "pastel"
            },
            "roi_analysis": {
                "type": "scatter",
                "title": "ROI vs Risk Analysis",
                "x_axis": "Risk Score",
                "y_axis": "Expected ROI (%)",
                "color_scheme": "RdYlGn"
            }
        }

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for report generation"""
        return [
            Tool(
                name="generate_market_report",
                description="Generate comprehensive market analysis report with charts",
                func=self.generate_market_analysis_report
            ),
            Tool(
                name="generate_comparison_report", 
                description="Generate property comparison report with visualizations",
                func=self.generate_property_comparison_report
            ),
            Tool(
                name="generate_investment_report",
                description="Generate investment analysis report with ROI calculations",
                func=self.generate_investment_analysis_report
            ),
            Tool(
                name="generate_location_report",
                description="Generate location-based analysis report",
                func=self.generate_location_report
            ),
            Tool(
                name="create_custom_chart",
                description="Create custom charts and visualizations",
                func=self.create_custom_visualization
            ),
            Tool(
                name="export_to_pdf",
                description="Export any report to PDF format",
                func=self.export_report_to_pdf
            )
        ]

    def generate_report(self, query: str, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main report generation method"""
        try:
            logger.info(f"📊 Generating report for query: {query}")
            
            # Determine report type based on query
            report_type = self._classify_report_type(query)
            
            # Generate report based on type
            if report_type == "market_analysis":
                return self.generate_market_analysis_report(data, user_id, query)
            elif report_type == "user_preference_report":
                return self.generate_user_preference_report(data, user_id, query)
            elif report_type == "investment_analysis":
                return self.generate_investment_analysis_report(data, user_id, query)
            elif report_type == "renovation_estimate":
                return self.generate_renovation_estimate_report(data, user_id, query)
            elif report_type == "location_report":
                return self.generate_location_report(data, user_id, query)
            else:
                return self.generate_custom_report(data, user_id, query)
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_summary": self._generate_basic_summary(data)
            }

    @rate_limited(groq_rate_limiter)
    def _classify_report_type(self, query: str) -> str:
        """Classify the type of report needed based on query"""
        try:
            classification_prompt = f"""
            Analyze this real estate query and classify the type of report needed.
            
            Query: "{query}"
            
            Choose ONE of these report types:
            - user_preference_report: For analyzing user preferences, behavior patterns, personalized insights
            - market_analysis: For market trends, price analysis, area insights
            - investment_analysis: For ROI, investment potential, financial analysis
            - renovation_estimate: For renovation cost estimation based on property size and rooms
            - location_report: For location-specific analysis and insights
            - custom: For other specialized reports
            
            PRIORITIZE user_preference_report for queries about:
            - User behavior, preferences, patterns
            - Personalized summaries and insights
            - "My" preferences or profile analysis
            - General reports without specific market/investment focus
            
            PRIORITIZE renovation_estimate for queries about:
            - Renovation costs, estimates, pricing
            - BHK-wise cost calculation
            - Property improvement costs
            - Construction or renovation budgets
            
            Return ONLY the report type (no additional text).
            """
            
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            report_type = response.content.strip().lower()
            
            valid_types = ["user_preference_report", "market_analysis", "investment_analysis", "renovation_estimate", "location_report", "custom"]
            if report_type in valid_types:
                return report_type
            else:
                return "custom"
                
        except Exception as e:
            logger.error(f"Report classification failed: {e}")
            return "custom"

    def generate_market_analysis_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate comprehensive market analysis report"""
        try:
            logger.info("📈 Generating market analysis report...")
            
            # Extract and analyze data
            properties = data.get("properties", [])
            market_data = self._analyze_market_data(properties)
            
            # Generate visualizations
            chart_paths = self._create_market_analysis_charts(market_data)
            
            # Generate report content using LLM
            report_content = self._generate_market_analysis_content(market_data, query)
            
            # Create report metadata
            report_metadata = ReportMetadata(
                report_id=f"market_analysis_{int(time.time())}",
                report_type="market_analysis",
                title="Real Estate Market Analysis Report",
                description=f"Comprehensive market analysis based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["property_database", "market_analysis"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "market_analysis",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "summary": f"Generated market analysis report with {len(chart_paths)} visualizations"
            }
            
        except Exception as e:
            logger.error(f"Market analysis report generation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_user_preference_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate user preference analysis report focusing on behavior patterns and preferences"""
        try:
            logger.info("👤 Generating user preference analysis report...")
            
            properties = data.get("properties", [])
            user_preferences = data.get("user_preferences", {})
            search_history = data.get("search_history", [])
            
            # Analyze user behavior and preferences
            preference_data = self._analyze_user_preferences(properties, user_preferences, search_history, user_id)
            
            # Generate preference visualization charts
            chart_paths = self._create_preference_charts(preference_data, user_id)
            
            # Generate personalized report content
            report_content = self._generate_preference_content(preference_data, query, user_id)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"user_pref_{user_id}_{int(time.time())}",
                report_type="user_preference_report",
                title=f"Personal Real Estate Preference Analysis - {user_id}",
                description=f"Personalized insights and preference analysis based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["user_behavior", "search_history", "property_interactions"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "user_preference_report",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "preferences_analyzed": len(preference_data.get('preference_categories', [])),
                "summary": f"Personalized preference analysis with {len(chart_paths)} visualizations of user behavior patterns"
            }
            
        except Exception as e:
            logger.error(f"User preference report generation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_renovation_estimate_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate renovation cost estimation report"""
        try:
            logger.info("🔨 Generating renovation cost estimation report...")
            
            if not RENOVATION_ESTIMATION_AVAILABLE or not self.renovation_agent:
                return {
                    "success": False,
                    "error": "Renovation estimation not available",
                    "fallback_message": "Renovation cost estimation requires additional components"
                }
            
            # Extract property details from data or query
            property_details = self._extract_property_details_for_renovation(data, query)
            
            if not property_details.get("total_area") or not property_details.get("bhk_config"):
                return {
                    "success": False,
                    "error": "Insufficient property details for renovation estimation",
                    "requirement": "Need property area and BHK configuration"
                }
            
            # Generate renovation estimate
            estimate = self.renovation_agent.estimate_renovation_cost(
                property_type=property_details.get("property_type", "apartment"),
                bhk_config=property_details.get("bhk_config"),
                total_area=property_details.get("total_area"),
                renovation_level=property_details.get("renovation_level", "premium"),
                room_details=property_details.get("room_details"),
                additional_requirements=property_details.get("additional_requirements", [])
            )
            
            # Generate detailed report content
            report_content = self.renovation_agent.generate_detailed_estimate_report(estimate)
            
            # Create renovation-specific charts
            chart_paths = self._create_renovation_charts(estimate)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"renovation_{int(time.time())}",
                report_type="renovation_estimate",
                title=f"Renovation Cost Estimation - {estimate.bhk_config}",
                description=f"Detailed renovation cost analysis: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["renovation_formulas", "bhk_pricing"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "renovation_estimate",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "estimation_details": {
                    "total_cost": estimate.total_cost,
                    "cost_per_sqft": estimate.cost_per_sqft,
                    "timeline_weeks": estimate.timeline_weeks,
                    "renovation_level": estimate.renovation_level,
                    "property_config": f"{estimate.bhk_config} {estimate.property_type}",
                    "total_area": estimate.total_area
                },
                "summary": f"Renovation cost estimate: ₹{estimate.total_cost:,.0f} for {estimate.total_area:,.0f} sq ft {estimate.bhk_config}"
            }
            
        except Exception as e:
            logger.error(f"Renovation estimation report failed: {e}")
            return {"success": False, "error": str(e)}

    def _extract_property_details_for_renovation(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract property details from data or query for renovation estimation"""
        property_details = {}
        
        # Try to extract from properties in data
        properties = data.get("properties", [])
        if properties:
            prop = properties[0]  # Use first property as reference
            
            # Extract area
            area_str = prop.get("area", "")
            if area_str:
                try:
                    # Extract numeric value from area string
                    area = float(''.join(filter(str.isdigit, str(area_str))))
                    if area > 0:
                        property_details["total_area"] = area
                except:
                    pass
            
            # Extract BHK from title or property_type
            title = prop.get("title", "").lower()
            prop_type = prop.get("property_type", "").lower()
            
            for bhk in ["1bhk", "2bhk", "3bhk", "4bhk", "5bhk", "studio", "villa", "penthouse"]:
                if bhk in title or bhk in prop_type:
                    property_details["bhk_config"] = bhk.upper() if "bhk" in bhk else bhk
                    break
            
            # Set property type
            property_details["property_type"] = prop.get("property_type", "apartment").lower()
        
        # Try to extract from query text
        query_lower = query.lower()
        
        # Extract BHK from query
        if not property_details.get("bhk_config"):
            for bhk in ["1 bhk", "2 bhk", "3 bhk", "4 bhk", "5 bhk", "studio", "villa", "penthouse"]:
                if bhk in query_lower:
                    bhk_clean = bhk.replace(" ", "").upper() if "bhk" in bhk else bhk
                    property_details["bhk_config"] = bhk_clean
                    break
        
        # Extract area from query (look for sq ft, sqft, square feet)
        if not property_details.get("total_area"):
            import re
            area_pattern = r'(\d+)\s*(sq\s*ft|sqft|square\s*feet)'
            match = re.search(area_pattern, query_lower)
            if match:
                property_details["total_area"] = float(match.group(1))
        
        # Extract renovation level from query
        for level in ["basic", "premium", "luxury", "complete"]:
            if level in query_lower:
                property_details["renovation_level"] = level
                break
        
        # Set defaults if not found
        if not property_details.get("total_area"):
            # Default area based on BHK
            bhk = property_details.get("bhk_config", "2BHK").lower()
            default_areas = {
                "studio": 450, "1bhk": 650, "2bhk": 1000, "3bhk": 1400, 
                "4bhk": 1800, "5bhk": 2200, "villa": 2500, "penthouse": 2000
            }
            property_details["total_area"] = default_areas.get(bhk, 1000)
        
        if not property_details.get("bhk_config"):
            property_details["bhk_config"] = "2BHK"  # Default
        
        # Extract additional requirements from query
        additional_reqs = []
        requirements = [
            "smart home", "solar panels", "home theater", "gym", "swimming pool",
            "garden", "security system", "ac installation", "modular furniture"
        ]
        for req in requirements:
            if req in query_lower:
                additional_reqs.append(req.replace(" ", "_"))
        
        if additional_reqs:
            property_details["additional_requirements"] = additional_reqs
        
        return property_details

    def _create_renovation_charts(self, estimate) -> Dict[str, str]:
        """Create charts for renovation cost estimation"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                logger.warning("Visualization libraries not available, skipping renovation charts")
                return chart_paths
            
            # Cost breakdown pie chart
            if estimate.category_breakdown:
                plt.figure(figsize=(10, 8))
                
                categories = list(estimate.category_breakdown.keys())
                costs = list(estimate.category_breakdown.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                
                plt.pie(costs, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
                plt.title(f'Renovation Cost Breakdown - {estimate.bhk_config}', fontsize=14, fontweight='bold')
                
                chart_path = os.path.join(self.charts_dir, f"renovation_breakdown_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["cost_breakdown"] = chart_path
            
            # Room-wise cost bar chart
            if estimate.room_breakdown:
                plt.figure(figsize=(12, 6))
                
                rooms = list(estimate.room_breakdown.keys())
                costs = list(estimate.room_breakdown.values())
                
                bars = plt.bar(rooms, costs, color='lightcoral', alpha=0.8, edgecolor='darkred')
                plt.title(f'Room-wise Renovation Costs - {estimate.bhk_config}', fontsize=14, fontweight='bold')
                plt.xlabel('Room Type')
                plt.ylabel('Cost (₹)')
                plt.xticks(rotation=45, ha='right')
                
                # Add cost labels on bars
                for bar, cost in zip(bars, costs):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                            f'₹{cost:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                chart_path = os.path.join(self.charts_dir, f"room_costs_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["room_breakdown"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} renovation charts")
            
        except Exception as e:
            logger.error(f"Renovation chart generation failed: {e}")
        
        return chart_paths

    def generate_property_comparison_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate property comparison report"""
        try:
            logger.info("🏠 Generating property comparison report...")
            
            properties = data.get("properties", [])
            if len(properties) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 properties for comparison",
                    "suggestion": "Please provide more properties to compare"
                }
            
            # Analyze properties for comparison
            comparison_data = self._analyze_property_comparison(properties)
            
            # Generate comparison charts
            chart_paths = self._create_comparison_charts(comparison_data)
            
            # Generate report content
            report_content = self._generate_comparison_content(comparison_data, query)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"comparison_{int(time.time())}",
                report_type="property_comparison", 
                title=f"Property Comparison Report - {len(properties)} Properties",
                description=f"Detailed comparison based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["property_database"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "property_comparison",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "properties_compared": len(properties),
                "summary": f"Compared {len(properties)} properties with detailed analysis"
            }
            
        except Exception as e:
            logger.error(f"Property comparison report failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_investment_analysis_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate investment analysis report with ROI calculations"""
        try:
            logger.info("💰 Generating investment analysis report...")
            
            properties = data.get("properties", [])
            market_data = data.get("market_data", {})
            
            # Perform investment analysis
            investment_data = self._analyze_investment_potential(properties, market_data)
            
            # Generate investment charts
            chart_paths = self._create_investment_charts(investment_data)
            
            # Generate report content
            report_content = self._generate_investment_content(investment_data, query)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"investment_{int(time.time())}",
                report_type="investment_analysis",
                title="Real Estate Investment Analysis Report",
                description=f"Investment analysis based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["property_database", "market_data", "roi_calculations"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "investment_analysis",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "investment_recommendations": investment_data.get("recommendations", []),
                "summary": f"Investment analysis with ROI calculations for {len(properties)} properties"
            }
            
        except Exception as e:
            logger.error(f"Investment analysis report failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_location_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate location-based analysis report"""
        try:
            logger.info("📍 Generating location analysis report...")
            
            properties = data.get("properties", [])
            location_data = self._analyze_location_data(properties, query)
            
            # Generate location charts
            chart_paths = self._create_location_charts(location_data)
            
            # Generate report content
            report_content = self._generate_location_content(location_data, query)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"location_{int(time.time())}",
                report_type="location_report",
                title=f"Location Analysis Report - {location_data.get('location', 'Multi-Location')}",
                description=f"Location analysis based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["property_database", "location_analysis"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "location_report",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "locations_analyzed": len(location_data.get("locations", [])),
                "summary": f"Location analysis for {location_data.get('location', 'multiple locations')}"
            }
            
        except Exception as e:
            logger.error(f"Location report generation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_custom_report(self, data: Dict[str, Any], user_id: str, query: str) -> Dict[str, Any]:
        """Generate custom report based on specific query"""
        try:
            logger.info("🎨 Generating custom report...")
            
            # Analyze the data and query to create custom report
            custom_data = self._analyze_custom_requirements(data, query)
            
            # Generate appropriate charts
            chart_paths = self._create_custom_charts(custom_data)
            
            # Generate custom content
            report_content = self._generate_custom_content(custom_data, query)
            
            # Create metadata
            report_metadata = ReportMetadata(
                report_id=f"custom_{int(time.time())}",
                report_type="custom",
                title="Custom Real Estate Report",
                description=f"Custom analysis based on: {query}",
                generated_at=datetime.now(),
                user_id=user_id,
                query=query,
                data_sources=["property_database", "custom_analysis"],
                charts_included=list(chart_paths.keys()),
                file_paths=chart_paths
            )
            
            # Generate PDF if requested
            pdf_path = None
            if "pdf" in query.lower() or "download" in query.lower():
                pdf_path = self.export_report_to_pdf(report_content, report_metadata)
                report_metadata.file_paths["pdf"] = pdf_path
            
            return {
                "success": True,
                "report_type": "custom",
                "content": report_content,
                "metadata": asdict(report_metadata),
                "charts": chart_paths,
                "pdf_path": pdf_path,
                "summary": "Custom report generated based on specific requirements"
            }
            
        except Exception as e:
            logger.error(f"Custom report generation failed: {e}")
            return {"success": False, "error": str(e)}

    # Data Analysis Methods
    def _analyze_user_preferences(self, properties: List[Dict], user_preferences: Dict, search_history: List, user_id: str) -> Dict[str, Any]:
        """Analyze user preferences and behavior patterns"""
        try:
            # Extract preference patterns from user data and properties
            preference_data = {
                "preferred_locations": [],
                "preferred_price_ranges": [],
                "preferred_property_types": [],
                "search_patterns": {},
                "behavior_insights": {},
                "preference_categories": []
            }
            
            # Analyze location preferences
            if properties:
                locations = [p.get('location', '').split(',')[0] for p in properties if p.get('location')]
                location_counts = {}
                for loc in locations:
                    if loc.strip():
                        location_counts[loc.strip()] = location_counts.get(loc.strip(), 0) + 1
                
                preference_data["preferred_locations"] = [
                    {"location": loc, "frequency": count, "preference_score": count/len(locations)*100}
                    for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            
            # Analyze price range preferences
            if properties:
                prices = []
                for p in properties:
                    price_str = p.get('price', '0')
                    try:
                        # Extract numeric price
                        price = float(''.join(filter(str.isdigit, str(price_str))))
                        if price > 0:
                            prices.append(price)
                    except:
                        continue
                
                if prices:
                    prices.sort()
                    preference_data["preferred_price_ranges"] = [
                        {
                            "range": f"₹{min(prices):,.0f} - ₹{max(prices):,.0f}",
                            "average": f"₹{sum(prices)/len(prices):,.0f}",
                            "properties_count": len(prices)
                        }
                    ]
            
            # Analyze property type preferences
            if properties:
                type_counts = {}
                for p in properties:
                    prop_type = p.get('property_type', 'Unknown')
                    type_counts[prop_type] = type_counts.get(prop_type, 0) + 1
                
                preference_data["preferred_property_types"] = [
                    {"type": ptype, "count": count, "percentage": round(count/len(properties)*100, 1)}
                    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                ]
            
            # Analyze search patterns from user preferences
            if user_preferences:
                preference_data["search_patterns"] = {
                    "total_searches": len(search_history) if search_history else 0,
                    "learned_preferences": len(user_preferences),
                    "preference_evolution": user_preferences
                }
            
            # Generate behavior insights
            preference_data["behavior_insights"] = {
                "search_focus": self._determine_search_focus(properties),
                "price_sensitivity": self._analyze_price_sensitivity(properties),
                "location_loyalty": self._analyze_location_patterns(preference_data["preferred_locations"]),
                "property_diversity": self._analyze_property_diversity(preference_data["preferred_property_types"])
            }
            
            # Define preference categories for visualization
            preference_data["preference_categories"] = [
                "Location Preferences", "Price Range Analysis", "Property Type Distribution", 
                "Search Behavior", "Market Preferences"
            ]
            
            return preference_data
            
        except Exception as e:
            logger.error(f"User preference analysis failed: {e}")
            return {"error": str(e), "preference_categories": []}
    
    def _determine_search_focus(self, properties: List[Dict]) -> str:
        """Determine the user's primary search focus"""
        if not properties:
            return "General browsing"
        
        # Analyze property characteristics to determine focus
        price_ranges = len(set([p.get('price', '') for p in properties[:5]]))
        locations = len(set([p.get('location', '').split(',')[0] for p in properties[:5]]))
        
        if price_ranges == 1:
            return "Budget-focused"
        elif locations == 1:
            return "Location-focused" 
        else:
            return "Exploration-focused"
    
    def _analyze_price_sensitivity(self, properties: List[Dict]) -> str:
        """Analyze user's price sensitivity"""
        if not properties:
            return "Unknown"
        
        prices = []
        for p in properties:
            try:
                price_str = str(p.get('price', '0'))
                price = float(''.join(filter(str.isdigit, price_str)))
                if price > 0:
                    prices.append(price)
            except:
                continue
        
        if len(prices) < 2:
            return "Needs more data"
        
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        
        if price_range / avg_price < 0.3:
            return "Consistent budget range"
        elif price_range / avg_price < 0.8:
            return "Moderate flexibility"
        else:
            return "Highly flexible budget"
    
    def _analyze_location_patterns(self, location_prefs: List[Dict]) -> str:
        """Analyze location preference patterns"""
        if not location_prefs:
            return "No clear pattern"
        
        top_location_score = location_prefs[0].get('preference_score', 0) if location_prefs else 0
        
        if top_location_score > 60:
            return "Strong location preference"
        elif top_location_score > 30:
            return "Moderate location focus"
        else:
            return "Open to multiple locations"
    
    def _analyze_property_diversity(self, type_prefs: List[Dict]) -> str:
        """Analyze property type diversity"""
        if not type_prefs:
            return "No data"
        
        if len(type_prefs) == 1:
            return "Single property type focus"
        elif len(type_prefs) <= 3:
            return "Limited property type exploration"
        else:
            return "Diverse property interests"

    def _analyze_market_data(self, properties: List[Dict]) -> Dict[str, Any]:
        """Analyze market data from properties"""
        try:
            if not properties:
                return {"error": "No properties provided for analysis"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(properties)
            
            # Basic market statistics
            analysis = {
                "total_properties": len(properties),
                "avg_price": df['price'].mean() if 'price' in df.columns else 0,
                "median_price": df['price'].median() if 'price' in df.columns else 0,
                "price_range": {
                    "min": df['price'].min() if 'price' in df.columns else 0,
                    "max": df['price'].max() if 'price' in df.columns else 0
                }
            }
            
            # Location analysis
            if 'location' in df.columns:
                location_stats = df.groupby('location').agg({
                    'price': ['count', 'mean', 'median']
                }).round(2)
                analysis["location_stats"] = location_stats.to_dict()
            
            # Property type analysis
            if 'property_type' in df.columns:
                type_stats = df.groupby('property_type').agg({
                    'price': ['count', 'mean']
                }).round(2)
                analysis["property_type_stats"] = type_stats.to_dict()
            
            # Price trends simulation (in real implementation, this would use historical data)
            analysis["price_trends"] = self._simulate_price_trends()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market data analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_property_comparison(self, properties: List[Dict]) -> Dict[str, Any]:
        """Analyze properties for comparison"""
        try:
            comparison_data = {
                "properties": [],
                "comparison_metrics": {},
                "rankings": {}
            }
            
            for prop in properties[:10]:  # Limit to 10 for comparison
                analysis = PropertyAnalysis(
                    property_id=prop.get('id', ''),
                    title=prop.get('title', 'Property'),
                    price=float(prop.get('price', 0)),
                    location=prop.get('location', ''),
                    property_type=prop.get('property_type', ''),
                    bedrooms=prop.get('bedrooms'),
                    bathrooms=prop.get('bathrooms'),
                    area=prop.get('area'),
                    price_per_sqft=self._calculate_price_per_sqft(prop),
                    roi_estimate=self._estimate_roi(prop),
                    rental_yield=self._estimate_rental_yield(prop),
                    appreciation_potential=self._assess_appreciation_potential(prop)
                )
                comparison_data["properties"].append(asdict(analysis))
            
            # Calculate comparison metrics
            comparison_data["comparison_metrics"] = self._calculate_comparison_metrics(properties)
            comparison_data["rankings"] = self._rank_properties(properties)
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Property comparison analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_investment_potential(self, properties: List[Dict], market_data: Dict) -> Dict[str, Any]:
        """Analyze investment potential of properties"""
        try:
            investment_analysis = {
                "properties": [],
                "market_overview": market_data,
                "investment_recommendations": [],
                "risk_assessment": {},
                "roi_projections": {}
            }
            
            for prop in properties:
                roi = self._calculate_detailed_roi(prop)
                risk_score = self._calculate_risk_score(prop)
                
                investment_analysis["properties"].append({
                    "property": prop,
                    "roi_analysis": roi,
                    "risk_score": risk_score,
                    "investment_rating": self._get_investment_rating(roi, risk_score)
                })
            
            # Generate investment recommendations
            investment_analysis["investment_recommendations"] = self._generate_investment_recommendations(
                investment_analysis["properties"]
            )
            
            return investment_analysis
            
        except Exception as e:
            logger.error(f"Investment analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_location_data(self, properties: List[Dict], query: str) -> Dict[str, Any]:
        """Analyze location-specific data"""
        try:
            # Group properties by location
            location_groups = {}
            for prop in properties:
                location = prop.get('location', 'Unknown')
                if location not in location_groups:
                    location_groups[location] = []
                location_groups[location].append(prop)
            
            location_analysis = {
                "locations": [],
                "location_comparison": {},
                "top_locations": []
            }
            
            for location, props in location_groups.items():
                if not props:
                    continue
                    
                avg_price = sum(p.get('price', 0) for p in props) / len(props)
                
                location_data = MarketTrends(
                    location=location,
                    avg_price=avg_price,
                    price_change_6m=np.random.uniform(-5, 15),  # Simulated
                    price_change_1y=np.random.uniform(-10, 25),  # Simulated
                    inventory_count=len(props),
                    demand_score=np.random.uniform(3, 9),  # Simulated
                    supply_score=np.random.uniform(2, 8),  # Simulated
                    trend_direction="up" if avg_price > 20000000 else "stable"
                )
                
                location_analysis["locations"].append(asdict(location_data))
            
            # Sort by average price to get top locations
            location_analysis["top_locations"] = sorted(
                location_analysis["locations"],
                key=lambda x: x["avg_price"],
                reverse=True
            )[:5]
            
            return location_analysis
            
        except Exception as e:
            logger.error(f"Location analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_custom_requirements(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Analyze data based on custom query requirements"""
        try:
            custom_analysis = {
                "query_analysis": query,
                "data_summary": {},
                "insights": [],
                "recommendations": []
            }
            
            properties = data.get("properties", [])
            if properties:
                custom_analysis["data_summary"] = {
                    "total_properties": len(properties),
                    "property_types": list(set(p.get('property_type', '') for p in properties)),
                    "locations": list(set(p.get('location', '') for p in properties)),
                    "price_range": {
                        "min": min(p.get('price', 0) for p in properties),
                        "max": max(p.get('price', 0) for p in properties)
                    }
                }
            
            # Generate insights based on query keywords
            custom_analysis["insights"] = self._generate_query_specific_insights(query, properties)
            
            return custom_analysis
            
        except Exception as e:
            logger.error(f"Custom analysis failed: {e}")
            return {"error": str(e)}

    # Utility calculation methods
    def _calculate_price_per_sqft(self, property_data: Dict) -> Optional[float]:
        """Calculate price per square foot"""
        try:
            price = property_data.get('price', 0)
            area = property_data.get('area', 0)
            if price and area:
                return round(price / area, 2)
            return None
        except:
            return None

    def _estimate_roi(self, property_data: Dict) -> Optional[float]:
        """Estimate ROI for property"""
        try:
            price = property_data.get('price', 0)
            # Simple ROI estimation based on property type and location
            base_roi = 8.0  # Base 8% ROI
            
            # Adjust based on property type
            prop_type = property_data.get('property_type', '').lower()
            if 'apartment' in prop_type:
                base_roi += 1.0
            elif 'villa' in prop_type:
                base_roi += 2.0
            
            # Add some randomness for demo
            return round(base_roi + np.random.uniform(-2, 4), 2)
            
        except:
            return None

    def _estimate_rental_yield(self, property_data: Dict) -> Optional[float]:
        """Estimate rental yield"""
        try:
            # Simplified rental yield calculation
            return round(np.random.uniform(3, 7), 2)
        except:
            return None

    def _assess_appreciation_potential(self, property_data: Dict) -> str:
        """Assess appreciation potential"""
        try:
            location = property_data.get('location', '').lower()
            
            # Simple heuristic based on location
            if any(city in location for city in ['mumbai', 'bangalore', 'delhi', 'pune']):
                return "High"
            elif any(city in location for city in ['hyderabad', 'chennai', 'kolkata']):
                return "Medium"
            else:
                return "Moderate"
                
        except:
            return "Unknown"

    def _calculate_comparison_metrics(self, properties: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for property comparison"""
        try:
            if not properties:
                return {}
            
            prices = [p.get('price', 0) for p in properties]
            
            return {
                "price_variance": np.var(prices),
                "price_std": np.std(prices),
                "best_value": min(prices) if prices else 0,
                "premium_property": max(prices) if prices else 0,
                "avg_price": np.mean(prices) if prices else 0
            }
            
        except:
            return {}

    def _rank_properties(self, properties: List[Dict]) -> Dict[str, List]:
        """Rank properties by different criteria"""
        try:
            # Sort by price (ascending)
            price_ranking = sorted(properties, key=lambda x: x.get('price', 0))
            
            # Sort by estimated value (price per sqft)
            value_ranking = sorted(
                properties,
                key=lambda x: self._calculate_price_per_sqft(x) or float('inf')
            )
            
            return {
                "by_price": [p.get('title', 'Unknown') for p in price_ranking[:5]],
                "by_value": [p.get('title', 'Unknown') for p in value_ranking[:5]]
            }
            
        except:
            return {"by_price": [], "by_value": []}

    def _calculate_detailed_roi(self, property_data: Dict) -> Dict[str, float]:
        """Calculate detailed ROI analysis"""
        try:
            price = property_data.get('price', 0)
            
            # Simulated ROI calculations
            annual_rental = price * 0.06  # Assume 6% rental yield
            maintenance_cost = price * 0.01  # Assume 1% maintenance
            net_annual_return = annual_rental - maintenance_cost
            
            return {
                "annual_rental_income": annual_rental,
                "maintenance_costs": maintenance_cost,
                "net_annual_return": net_annual_return,
                "roi_percentage": (net_annual_return / price) * 100,
                "payback_period": price / net_annual_return if net_annual_return > 0 else 0
            }
            
        except:
            return {}

    def _calculate_risk_score(self, property_data: Dict) -> float:
        """Calculate risk score for investment"""
        try:
            # Simple risk scoring based on price and location
            price = property_data.get('price', 0)
            location = property_data.get('location', '').lower()
            
            risk_score = 5.0  # Base risk score (out of 10)
            
            # Adjust based on price (higher price = higher risk)
            if price > 50000000:
                risk_score += 2
            elif price > 20000000:
                risk_score += 1
            
            # Adjust based on location
            if any(city in location for city in ['mumbai', 'delhi', 'bangalore']):
                risk_score -= 1  # Lower risk in major cities
            
            return max(1.0, min(10.0, risk_score))
            
        except:
            return 5.0

    def _get_investment_rating(self, roi_data: Dict, risk_score: float) -> str:
        """Get investment rating based on ROI and risk"""
        try:
            roi = roi_data.get('roi_percentage', 0)
            
            if roi > 12 and risk_score < 5:
                return "Excellent"
            elif roi > 10 and risk_score < 6:
                return "Good"
            elif roi > 8 and risk_score < 7:
                return "Fair"
            else:
                return "Poor"
                
        except:
            return "Unknown"

    def _generate_investment_recommendations(self, investment_properties: List[Dict]) -> List[str]:
        """Generate investment recommendations"""
        try:
            recommendations = []
            
            # Find best ROI
            best_roi_prop = max(
                investment_properties,
                key=lambda x: x.get('roi_analysis', {}).get('roi_percentage', 0),
                default=None
            )
            
            if best_roi_prop:
                recommendations.append(
                    f"Best ROI: {best_roi_prop['property'].get('title', 'Property')} "
                    f"with {best_roi_prop.get('roi_analysis', {}).get('roi_percentage', 0):.2f}% returns"
                )
            
            # Find lowest risk
            lowest_risk_prop = min(
                investment_properties,
                key=lambda x: x.get('risk_score', 10),
                default=None
            )
            
            if lowest_risk_prop:
                recommendations.append(
                    f"Lowest Risk: {lowest_risk_prop['property'].get('title', 'Property')} "
                    f"with risk score {lowest_risk_prop.get('risk_score', 0):.1f}/10"
                )
            
            return recommendations
            
        except:
            return ["Unable to generate specific recommendations"]

    def _simulate_price_trends(self) -> Dict[str, List]:
        """Simulate price trends for demonstration"""
        try:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            base_price = 25000000
            
            # Simulate upward trend with some fluctuation
            prices = []
            for i in range(6):
                price = base_price * (1 + (i * 0.02) + np.random.uniform(-0.01, 0.01))
                prices.append(int(price))
            
            return {
                "months": months,
                "prices": prices,
                "trend": "upward"
            }
            
        except:
            return {"months": [], "prices": [], "trend": "unknown"}

    def _generate_query_specific_insights(self, query: str, properties: List[Dict]) -> List[str]:
        """Generate insights specific to the query"""
        try:
            insights = []
            query_lower = query.lower()
            
            if 'investment' in query_lower:
                insights.append("Investment properties show strong potential in current market")
            
            if 'rental' in query_lower:
                insights.append("Rental yields vary significantly across different locations")
            
            if 'price' in query_lower or 'cost' in query_lower:
                insights.append("Price analysis reveals competitive opportunities in the market")
            
            if not insights:
                insights.append("Market analysis shows diverse opportunities across property types")
            
            return insights
            
        except:
            return ["Market analysis completed successfully"]

    # Chart Generation Methods
    def _create_market_analysis_charts(self, market_data: Dict) -> Dict[str, str]:
        """Create charts for market analysis"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                logger.warning("Visualization libraries not available, skipping chart generation")
                return chart_paths
            
            # Price trends chart
            trends = market_data.get("price_trends", {})
            if trends.get("months") and trends.get("prices"):
                plt.figure(figsize=(10, 6))
                plt.plot(trends["months"], trends["prices"], marker='o', linewidth=2, markersize=6)
                plt.title("Price Trends Over Time", fontsize=16, fontweight='bold')
                plt.xlabel("Month", fontsize=12)
                plt.ylabel("Average Price (₹)", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.ticklabel_format(style='plain', axis='y')
                
                chart_path = os.path.join(self.reports_dir, "charts", f"price_trends_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["price_trends"] = chart_path
            
            # Location comparison chart
            location_stats = market_data.get("location_stats", {})
            if location_stats:
                locations = list(location_stats.keys())[:8]  # Top 8 locations
                prices = [location_stats[loc]['price']['mean'] for loc in locations if 'price' in location_stats[loc]]
                
                if prices:
                    plt.figure(figsize=(12, 8))
                    bars = plt.bar(locations, prices, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
                    plt.title("Average Prices by Location", fontsize=16, fontweight='bold')
                    plt.xlabel("Location", fontsize=12)
                    plt.ylabel("Average Price (₹)", fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for bar, price in zip(bars, prices):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prices)*0.01,
                               f'₹{price/10000000:.1f}Cr', ha='center', va='bottom', fontsize=10)
                    
                    chart_path = os.path.join(self.reports_dir, "charts", f"location_comparison_{int(time.time())}.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths["location_comparison"] = chart_path
            
            # Property type distribution
            type_stats = market_data.get("property_type_stats", {})
            if type_stats:
                types = list(type_stats.keys())
                counts = [type_stats[t]['price']['count'] for t in types if 'price' in type_stats[t]]
                
                if counts:
                    plt.figure(figsize=(10, 8))
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                    wedges, texts, autotexts = plt.pie(counts, labels=types, autopct='%1.1f%%', 
                                                     colors=colors[:len(types)], startangle=90)
                    plt.title("Property Type Distribution", fontsize=16, fontweight='bold')
                    
                    chart_path = os.path.join(self.reports_dir, "charts", f"property_distribution_{int(time.time())}.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths["property_distribution"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} market analysis charts")
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
        
        return chart_paths

    def _create_preference_charts(self, preference_data: Dict, user_id: str) -> Dict[str, str]:
        """Create charts for user preference analysis"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                logger.warning("Visualization libraries not available, skipping preference charts")
                return chart_paths
            
            # Location preferences pie chart
            location_prefs = preference_data.get("preferred_locations", [])
            if location_prefs:
                plt.figure(figsize=(10, 8))
                
                locations = [loc["location"] for loc in location_prefs[:6]]
                frequencies = [loc["frequency"] for loc in location_prefs[:6]]
                colors = plt.cm.Set3(np.linspace(0, 1, len(locations)))
                
                plt.pie(frequencies, labels=locations, autopct='%1.1f%%', startangle=90, colors=colors)
                plt.title(f'Location Preferences - {user_id}', fontsize=14, fontweight='bold')
                
                chart_path = os.path.join(self.charts_dir, f"location_preferences_{user_id}_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["location_preferences"] = chart_path
            
            # Property type preferences bar chart
            type_prefs = preference_data.get("preferred_property_types", [])
            if type_prefs:
                plt.figure(figsize=(12, 6))
                
                types = [ptype["type"] for ptype in type_prefs[:8]]
                counts = [ptype["count"] for ptype in type_prefs[:8]]
                percentages = [ptype["percentage"] for ptype in type_prefs[:8]]
                
                bars = plt.bar(types, counts, color='skyblue', alpha=0.8, edgecolor='navy')
                plt.title(f'Property Type Preferences - {user_id}', fontsize=14, fontweight='bold')
                plt.xlabel('Property Type')
                plt.ylabel('Number of Properties Viewed')
                plt.xticks(rotation=45, ha='right')
                
                # Add percentage labels on bars
                for bar, percentage in zip(bars, percentages):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{percentage}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                chart_path = os.path.join(self.charts_dir, f"property_types_{user_id}_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["property_types"] = chart_path
            
            # Behavior insights radar chart
            behavior_data = preference_data.get("behavior_insights", {})
            if behavior_data:
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                categories = ['Search Focus', 'Price Sensitivity', 'Location Loyalty', 'Property Diversity']
                # Convert text insights to numeric scores for visualization
                scores = [
                    self._behavior_to_score(behavior_data.get("search_focus", "")),
                    self._behavior_to_score(behavior_data.get("price_sensitivity", "")), 
                    self._behavior_to_score(behavior_data.get("location_loyalty", "")),
                    self._behavior_to_score(behavior_data.get("property_diversity", ""))
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                scores += scores[:1]  # Complete the circle
                angles += angles[:1]
                
                ax.plot(angles, scores, 'o-', linewidth=2, color='blue', alpha=0.8)
                ax.fill(angles, scores, alpha=0.25, color='blue')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 5)
                ax.set_title(f'User Behavior Profile - {user_id}', size=14, fontweight='bold', y=1.08)
                
                chart_path = os.path.join(self.charts_dir, f"behavior_radar_{user_id}_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["behavior_insights"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} preference analysis charts")
            
        except Exception as e:
            logger.error(f"Preference chart generation failed: {e}")
        
        return chart_paths
    
    def _behavior_to_score(self, behavior_text: str) -> float:
        """Convert behavior text description to numeric score for radar chart"""
        behavior_lower = behavior_text.lower()
        
        # Map behavior patterns to scores (1-5 scale)
        if any(word in behavior_lower for word in ['strong', 'focused', 'consistent', 'single']):
            return 5.0
        elif any(word in behavior_lower for word in ['moderate', 'limited', 'flexibility']):
            return 3.5
        elif any(word in behavior_lower for word in ['diverse', 'open', 'exploration', 'flexible']):
            return 2.0
        elif any(word in behavior_lower for word in ['needs more', 'unknown', 'no data']):
            return 1.0
        else:
            return 3.0

    def _create_comparison_charts(self, comparison_data: Dict) -> Dict[str, str]:
        """Create charts for property comparison"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                return chart_paths
            
            properties = comparison_data.get("properties", [])[:8]  # Limit to 8 for readability
            
            if properties:
                # Price comparison chart
                titles = [prop["title"][:30] + "..." if len(prop["title"]) > 30 else prop["title"] 
                         for prop in properties]
                prices = [prop["price"] for prop in properties]
                
                plt.figure(figsize=(14, 8))
                bars = plt.bar(range(len(titles)), prices, color='lightcoral', alpha=0.8, edgecolor='darkred')
                plt.title("Property Price Comparison", fontsize=16, fontweight='bold')
                plt.xlabel("Properties", fontsize=12)
                plt.ylabel("Price (₹)", fontsize=12)
                plt.xticks(range(len(titles)), titles, rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add price labels
                for i, (bar, price) in enumerate(zip(bars, prices)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prices)*0.01,
                           f'₹{price/10000000:.1f}Cr', ha='center', va='bottom', fontsize=9)
                
                chart_path = os.path.join(self.reports_dir, "charts", f"price_comparison_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["price_comparison"] = chart_path
                
                # ROI comparison chart
                roi_values = [prop.get("roi_estimate", 0) for prop in properties]
                if any(roi_values):
                    plt.figure(figsize=(12, 6))
                    plt.scatter(range(len(titles)), roi_values, s=100, c=roi_values, 
                              cmap='RdYlGn', alpha=0.7, edgecolors='black')
                    plt.title("Expected ROI Comparison", fontsize=16, fontweight='bold')
                    plt.xlabel("Properties", fontsize=12)
                    plt.ylabel("Expected ROI (%)", fontsize=12)
                    plt.xticks(range(len(titles)), [f"P{i+1}" for i in range(len(titles))])
                    plt.colorbar(label='ROI %')
                    plt.grid(True, alpha=0.3)
                    
                    chart_path = os.path.join(self.reports_dir, "charts", f"roi_comparison_{int(time.time())}.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths["roi_comparison"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} comparison charts")
            
        except Exception as e:
            logger.error(f"Comparison chart generation failed: {e}")
        
        return chart_paths

    def _create_investment_charts(self, investment_data: Dict) -> Dict[str, str]:
        """Create charts for investment analysis"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                return chart_paths
            
            properties = investment_data.get("properties", [])[:10]
            
            if properties:
                # ROI vs Risk scatter plot
                roi_values = [prop.get("roi_analysis", {}).get("roi_percentage", 0) for prop in properties]
                risk_values = [prop.get("risk_score", 5) for prop in properties]
                
                if roi_values and risk_values:
                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(risk_values, roi_values, s=150, alpha=0.7, 
                                        c=roi_values, cmap='RdYlGn', edgecolors='black')
                    plt.title("Investment Analysis: ROI vs Risk", fontsize=16, fontweight='bold')
                    plt.xlabel("Risk Score (1-10)", fontsize=12)
                    plt.ylabel("Expected ROI (%)", fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.colorbar(scatter, label='ROI %')
                    
                    # Add quadrant labels
                    plt.axhline(y=np.mean(roi_values), color='gray', linestyle='--', alpha=0.5)
                    plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
                    plt.text(2, max(roi_values)*0.9, 'Low Risk\nHigh ROI', ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
                    plt.text(8, max(roi_values)*0.9, 'High Risk\nHigh ROI', ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
                    
                    chart_path = os.path.join(self.reports_dir, "charts", f"investment_analysis_{int(time.time())}.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths["investment_analysis"] = chart_path
                
                # Investment ratings pie chart
                ratings = [prop.get("investment_rating", "Unknown") for prop in properties]
                rating_counts = pd.Series(ratings).value_counts()
                
                if len(rating_counts) > 0:
                    plt.figure(figsize=(10, 8))
                    colors = ['#2E8B57', '#FFD700', '#FFA500', '#CD5C5C']
                    plt.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                           colors=colors[:len(rating_counts)], startangle=90)
                    plt.title("Investment Rating Distribution", fontsize=16, fontweight='bold')
                    
                    chart_path = os.path.join(self.reports_dir, "charts", f"investment_ratings_{int(time.time())}.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths["investment_ratings"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} investment charts")
            
        except Exception as e:
            logger.error(f"Investment chart generation failed: {e}")
        
        return chart_paths

    def _create_location_charts(self, location_data: Dict) -> Dict[str, str]:
        """Create charts for location analysis"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                return chart_paths
            
            locations = location_data.get("locations", [])[:10]
            
            if locations:
                # Location price comparison
                location_names = [loc["location"] for loc in locations]
                avg_prices = [loc["avg_price"] for loc in locations]
                
                plt.figure(figsize=(14, 8))
                bars = plt.bar(location_names, avg_prices, color='lightblue', alpha=0.8, edgecolor='navy')
                plt.title("Average Property Prices by Location", fontsize=16, fontweight='bold')
                plt.xlabel("Location", fontsize=12)
                plt.ylabel("Average Price (₹)", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add price labels
                for bar, price in zip(bars, avg_prices):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_prices)*0.01,
                           f'₹{price/10000000:.1f}Cr', ha='center', va='bottom', fontsize=9)
                
                chart_path = os.path.join(self.reports_dir, "charts", f"location_prices_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["location_prices"] = chart_path
                
                # Demand vs Supply analysis
                demand_scores = [loc.get("demand_score", 5) for loc in locations]
                supply_scores = [loc.get("supply_score", 5) for loc in locations]
                
                plt.figure(figsize=(12, 8))
                plt.scatter(supply_scores, demand_scores, s=150, alpha=0.7, 
                          c=avg_prices, cmap='viridis', edgecolors='black')
                plt.title("Location Analysis: Demand vs Supply", fontsize=16, fontweight='bold')
                plt.xlabel("Supply Score", fontsize=12)
                plt.ylabel("Demand Score", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.colorbar(label='Avg Price (₹)')
                
                # Add location labels
                for i, location in enumerate(location_names):
                    plt.annotate(location[:10], (supply_scores[i], demand_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                chart_path = os.path.join(self.reports_dir, "charts", f"demand_supply_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["demand_supply"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} location charts")
            
        except Exception as e:
            logger.error(f"Location chart generation failed: {e}")
        
        return chart_paths

    def _create_custom_charts(self, custom_data: Dict) -> Dict[str, str]:
        """Create custom charts based on data"""
        chart_paths = {}
        
        try:
            if not VISUALIZATIONS_AVAILABLE:
                return chart_paths
            
            # Create a simple summary chart
            data_summary = custom_data.get("data_summary", {})
            
            if data_summary.get("property_types"):
                types = data_summary["property_types"]
                # Create a simple count chart (simulated data)
                counts = [np.random.randint(1, 20) for _ in types]
                
                plt.figure(figsize=(10, 6))
                plt.bar(types, counts, color='gold', alpha=0.8, edgecolor='orange')
                plt.title("Property Analysis Overview", fontsize=16, fontweight='bold')
                plt.xlabel("Property Type", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                chart_path = os.path.join(self.reports_dir, "charts", f"custom_overview_{int(time.time())}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths["custom_overview"] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} custom charts")
            
        except Exception as e:
            logger.error(f"Custom chart generation failed: {e}")
        
        return chart_paths

    def create_custom_visualization(self, chart_type: str, data: Dict, title: str = "Custom Chart") -> str:
        """Create custom visualization based on parameters"""
        try:
            if not VISUALIZATIONS_AVAILABLE:
                logger.warning("Visualization libraries not available")
                return ""
            
            chart_path = os.path.join(self.reports_dir, "charts", f"custom_{chart_type}_{int(time.time())}.png")
            
            plt.figure(figsize=(12, 8))
            
            if chart_type == "bar":
                x_data = data.get("x_data", [])
                y_data = data.get("y_data", [])
                plt.bar(x_data, y_data, alpha=0.8)
                
            elif chart_type == "line":
                x_data = data.get("x_data", [])
                y_data = data.get("y_data", [])
                plt.plot(x_data, y_data, marker='o', linewidth=2, markersize=6)
                
            elif chart_type == "pie":
                labels = data.get("labels", [])
                values = data.get("values", [])
                plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                
            elif chart_type == "scatter":
                x_data = data.get("x_data", [])
                y_data = data.get("y_data", [])
                plt.scatter(x_data, y_data, s=100, alpha=0.7)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Custom visualization creation failed: {e}")
            return ""

    # Content Generation Methods
    @rate_limited(groq_rate_limiter)
    def _generate_market_analysis_content(self, market_data: Dict, query: str) -> str:
        """Generate market analysis report content using LLM"""
        try:
            prompt = f"""
            Generate a comprehensive real estate market analysis report based on the following data:
            
            Query: {query}
            
            Market Data:
            - Total Properties Analyzed: {market_data.get('total_properties', 0)}
            - Average Price: ₹{market_data.get('avg_price', 0):,.0f}
            - Median Price: ₹{market_data.get('median_price', 0):,.0f}
            - Price Range: ₹{market_data.get('price_range', {}).get('min', 0):,.0f} - ₹{market_data.get('price_range', {}).get('max', 0):,.0f}
            - Price Trend: {market_data.get('price_trends', {}).get('trend', 'Unknown')}
            
            Generate a professional report with the following sections:
            1. Executive Summary
            2. Market Overview
            3. Price Analysis
            4. Trend Analysis  
            5. Investment Outlook
            6. Key Recommendations
            
            Use proper formatting with headers, bullet points, and professional language.
            Include specific numbers and insights from the data provided.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Market analysis content generation failed: {e}")
            return self._generate_fallback_market_content(market_data)

    @rate_limited(groq_rate_limiter)
    def _generate_preference_content(self, preference_data: Dict, query: str, user_id: str) -> str:
        """Generate personalized user preference analysis content"""
        try:
            location_prefs = preference_data.get("preferred_locations", [])
            price_ranges = preference_data.get("preferred_price_ranges", [])
            type_prefs = preference_data.get("preferred_property_types", [])
            behavior_insights = preference_data.get("behavior_insights", {})
            search_patterns = preference_data.get("search_patterns", {})
            
            prompt = f"""
            Generate a comprehensive personalized real estate preference analysis report for user: {user_id}
            
            Query: {query}
            
            USER PREFERENCE ANALYSIS DATA:
            
            Location Preferences:
            {self._format_location_preferences(location_prefs)}
            
            Price Range Preferences:
            {json.dumps(price_ranges, indent=2)}
            
            Property Type Preferences:
            {self._format_type_preferences(type_prefs)}
            
            Behavior Insights:
            - Search Focus: {behavior_insights.get('search_focus', 'Unknown')}
            - Price Sensitivity: {behavior_insights.get('price_sensitivity', 'Unknown')}
            - Location Loyalty: {behavior_insights.get('location_loyalty', 'Unknown')}
            - Property Diversity: {behavior_insights.get('property_diversity', 'Unknown')}
            
            Search Patterns:
            - Total Searches: {search_patterns.get('total_searches', 0)}
            - Learned Preferences: {search_patterns.get('learned_preferences', 0)}
            
            Generate a PERSONALIZED report focusing on:
            1. User Profile Summary
            2. Location Preference Analysis 
            3. Budget & Price Sensitivity
            4. Property Type Inclinations
            5. Search Behavior Patterns
            6. Personalized Recommendations
            7. Preference Evolution Insights
            
            Focus on VISUALIZING and SUMMARIZING the user's preferences rather than comparing properties.
            Use personal language ("You prefer...", "Your search patterns show...").
            Include actionable insights based on their behavior patterns.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Preference content generation failed: {e}")
            return self._generate_fallback_preference_content(preference_data, user_id)

    @rate_limited(groq_rate_limiter)
    def _generate_comparison_content(self, comparison_data: Dict, query: str) -> str:
        """Generate property comparison report content"""
        try:
            properties = comparison_data.get("properties", [])
            
            prompt = f"""
            Generate a comprehensive property comparison report based on the following data:
            
            Query: {query}
            
            Properties Being Compared: {len(properties)}
            
            Property Details:
            {self._format_properties_for_prompt(properties[:5])}
            
            Comparison Metrics:
            {json.dumps(comparison_data.get('comparison_metrics', {}), indent=2)}
            
            Generate a professional report with:
            1. Comparison Overview
            2. Property Details Analysis
            3. Price Comparison
            4. Value Analysis
            5. Investment Potential
            6. Final Recommendations
            
            Include specific comparisons, highlight best values, and provide clear recommendations.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Comparison content generation failed: {e}")
            return self._generate_fallback_comparison_content(comparison_data)

    @rate_limited(groq_rate_limiter)
    def _generate_investment_content(self, investment_data: Dict, query: str) -> str:
        """Generate investment analysis report content"""
        try:
            properties = investment_data.get("properties", [])
            recommendations = investment_data.get("investment_recommendations", [])
            
            prompt = f"""
            Generate a comprehensive real estate investment analysis report:
            
            Query: {query}
            
            Investment Analysis Data:
            - Properties Analyzed: {len(properties)}
            - Key Recommendations: {recommendations}
            
            Top Investment Opportunities:
            {self._format_investment_properties_for_prompt(properties[:3])}
            
            Generate a professional investment report with:
            1. Investment Summary
            2. ROI Analysis
            3. Risk Assessment
            4. Market Positioning
            5. Financial Projections
            6. Investment Recommendations
            
            Include specific ROI numbers, risk assessments, and actionable investment advice.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Investment content generation failed: {e}")
            return self._generate_fallback_investment_content(investment_data)

    @rate_limited(groq_rate_limiter)
    def _generate_location_content(self, location_data: Dict, query: str) -> str:
        """Generate location analysis report content"""
        try:
            locations = location_data.get("locations", [])
            
            prompt = f"""
            Generate a comprehensive location analysis report for real estate:
            
            Query: {query}
            
            Location Analysis:
            - Locations Analyzed: {len(locations)}
            - Top Locations by Price: {[loc['location'] for loc in location_data.get('top_locations', [])[:3]]}
            
            Location Details:
            {self._format_locations_for_prompt(locations[:5])}
            
            Generate a professional location report with:
            1. Location Overview
            2. Market Analysis by Area
            3. Price Trends
            4. Demand & Supply Analysis
            5. Growth Prospects
            6. Investment Recommendations
            
            Focus on location-specific insights, price comparisons, and growth potential.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Location content generation failed: {e}")
            return self._generate_fallback_location_content(location_data)

    @rate_limited(groq_rate_limiter)
    def _generate_custom_content(self, custom_data: Dict, query: str) -> str:
        """Generate custom report content"""
        try:
            prompt = f"""
            Generate a custom real estate report based on the specific query and data:
            
            Query: {query}
            
            Analysis Data:
            {json.dumps(custom_data.get('data_summary', {}), indent=2)}
            
            Key Insights:
            {json.dumps(custom_data.get('insights', []), indent=2)}
            
            Generate a professional custom report addressing the specific requirements in the query.
            Include relevant analysis, insights, and recommendations based on the available data.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Custom content generation failed: {e}")
            return self._generate_fallback_custom_content(custom_data)

    def _format_location_preferences(self, location_prefs: List[Dict]) -> str:
        """Format location preferences for LLM prompt"""
        if not location_prefs:
            return "No clear location preferences identified."
        
        formatted = []
        for loc in location_prefs:
            formatted.append(f"""
            Location: {loc.get('location', 'N/A')}
            Search Frequency: {loc.get('frequency', 0)} times
            Preference Score: {loc.get('preference_score', 0):.1f}%
            """)
        return "\n".join(formatted)
    
    def _format_type_preferences(self, type_prefs: List[Dict]) -> str:
        """Format property type preferences for LLM prompt"""
        if not type_prefs:
            return "No clear property type preferences identified."
        
        formatted = []
        for ptype in type_prefs:
            formatted.append(f"""
            Property Type: {ptype.get('type', 'N/A')}
            Properties Viewed: {ptype.get('count', 0)}
            Preference Percentage: {ptype.get('percentage', 0)}%
            """)
        return "\n".join(formatted)

    def _format_investment_properties_for_prompt(self, properties: List[Dict]) -> str:
        """Format investment properties for LLM prompt"""
        if not properties:
            return "No properties available for analysis."
        
        formatted = []
        for prop in properties:
            formatted.append(f"""
            Property: {prop.get('title', 'N/A')}
            Location: {prop.get('location', 'N/A')}
            Price: {prop.get('price', 'N/A')}
            Type: {prop.get('property_type', 'N/A')}
            ROI Potential: {prop.get('roi_estimate', 'TBD')}
            """)
        return "\n".join(formatted)
    
    def _format_locations_for_prompt(self, locations: List[Dict]) -> str:
        """Format locations for LLM prompt"""
        if not locations:
            return "No location data available."
        
        formatted = []
        for loc in locations:
            formatted.append(f"""
            Location: {loc.get('location', 'N/A')}
            Average Price: {loc.get('avg_price', 'N/A')}
            Property Count: {loc.get('property_count', 'N/A')}
            Growth Rate: {loc.get('growth_rate', 'N/A')}
            """)
        return "\n".join(formatted)

    # Fallback content generation methods
    def _generate_fallback_market_content(self, market_data: Dict) -> str:
        """Generate fallback market analysis content"""
        return f"""
        # Market Analysis Report
        
        ## Executive Summary
        Analysis of the real estate market based on available data shows significant insights across {len(market_data.get('properties', []))} properties.
        
        ## Key Findings
        - Total properties analyzed: {len(market_data.get('properties', []))}
        - Market trends show varied pricing across different locations
        - Investment opportunities identified in multiple segments
        
        ## Market Overview
        The current real estate market presents diverse opportunities for both investors and homebuyers. Our analysis reveals competitive pricing and healthy market activity.
        
        ## Recommendations
        - Continue monitoring market trends
        - Focus on high-growth areas
        - Consider diversified investment strategies
        
        *This is a fallback report. For detailed analysis, please ensure proper data connectivity.*
        """
    
    def _generate_fallback_comparison_content(self, comparison_data: Dict) -> str:
        """Generate fallback comparison content"""
        properties = comparison_data.get('properties', [])
        return f"""
        # Property Comparison Report
        
        ## Comparison Overview
        Analysis of {len(properties)} properties for comparative evaluation.
        
        ## Properties Analyzed
        {chr(10).join([f"- {prop.get('title', 'Property')} at {prop.get('location', 'N/A')}: {prop.get('price', 'N/A')}" for prop in properties[:5]])}
        
        ## Key Insights
        - Multiple properties evaluated across different criteria
        - Price variations observed across locations
        - Diverse property types included in analysis
        
        ## Recommendations
        - Consider location-specific factors in decision making
        - Evaluate price-to-value ratio for each property
        - Review amenities and features for best fit
        
        *This is a fallback report. For detailed comparison, please ensure proper data connectivity.*
        """
    
    def _generate_fallback_investment_content(self, investment_data: Dict) -> str:
        """Generate fallback investment analysis content"""
        properties = investment_data.get('properties', [])
        return f"""
        # Investment Analysis Report
        
        ## Investment Overview
        Comprehensive analysis of {len(properties)} properties for investment potential.
        
        ## Investment Summary
        - Properties evaluated: {len(properties)}
        - Investment recommendations available
        - ROI potential assessed across portfolio
        
        ## Risk Assessment
        - Market risks: Moderate
        - Location risks: Varied
        - Financial risks: Under evaluation
        
        ## Financial Projections
        - Expected returns: Market dependent
        - Investment horizon: Long-term recommended
        - Diversification: Multiple locations suggested
        
        ## Investment Recommendations
        - Focus on high-growth areas
        - Consider market timing
        - Evaluate financing options
        - Plan for long-term appreciation
        
        *This is a fallback report. For detailed investment analysis, please ensure proper data connectivity.*
        """
    
    def _generate_fallback_location_content(self, location_data: Dict) -> str:
        """Generate fallback location analysis content"""
        locations = location_data.get('locations', [])
        return f"""
        # Location Analysis Report
        
        ## Location Overview
        Analysis of {len(locations)} locations for real estate opportunities.
        
        ## Market Analysis by Area
        - Multiple locations evaluated
        - Price variations across different areas
        - Growth potential identified in various regions
        
        ## Price Trends
        - Varied pricing observed across locations
        - Market dynamics differ by area
        - Opportunities exist across price ranges
        
        ## Growth Prospects
        - Several locations show positive trends
        - Infrastructure development impacts pricing
        - Long-term growth potential varies by area
        
        ## Location Recommendations
        - Research local market conditions
        - Consider proximity to amenities
        - Evaluate transportation connectivity
        - Assess future development plans
        
        *This is a fallback report. For detailed location analysis, please ensure proper data connectivity.*
        """
    
    def _generate_fallback_preference_content(self, preference_data: Dict, user_id: str) -> str:
        """Generate fallback preference analysis content"""
        location_prefs = preference_data.get('preferred_locations', [])
        type_prefs = preference_data.get('preferred_property_types', [])
        behavior_insights = preference_data.get('behavior_insights', {})
        
        return f"""
        # Personal Real Estate Preference Analysis - {user_id}
        
        ## Your Profile Summary
        Based on your search activity, we've identified several key preferences and patterns that define your real estate interests.
        
        ## Location Preferences
        You have shown interest in {len(location_prefs)} different locations:
        {chr(10).join([f"- {loc.get('location', 'N/A')} ({loc.get('frequency', 0)} searches)" for loc in location_prefs[:3]])}
        
        ## Property Type Interests  
        Your property type preferences show:
        {chr(10).join([f"- {ptype.get('type', 'N/A')}: {ptype.get('percentage', 0)}% of your searches" for ptype in type_prefs[:3]])}
        
        ## Your Search Behavior
        - Search Focus: {behavior_insights.get('search_focus', 'General exploration')}
        - Price Approach: {behavior_insights.get('price_sensitivity', 'Moderate flexibility')}
        - Location Strategy: {behavior_insights.get('location_loyalty', 'Open to options')}
        - Property Diversity: {behavior_insights.get('property_diversity', 'Varied interests')}
        
        ## Personalized Insights
        - Your search patterns indicate a {behavior_insights.get('search_focus', 'balanced').lower()} approach to property hunting
        - You demonstrate {behavior_insights.get('price_sensitivity', 'moderate').lower()} when it comes to pricing
        - Your location preferences show you are {behavior_insights.get('location_loyalty', 'open to exploring').lower()}
        
        ## Recommendations for You
        - Continue exploring properties in your preferred locations
        - Consider expanding your search to similar areas with comparable characteristics
        - Monitor market trends in your areas of interest
        - Set up alerts for properties matching your identified preferences
        
        *This is a personalized analysis based on your search behavior. Continue using the platform to refine these insights.*
        """
    
    def _generate_fallback_custom_content(self, custom_data: Dict) -> str:
        """Generate fallback custom report content"""
        return f"""
        # Custom Analysis Report
        
        ## Report Summary
        Custom analysis based on specific requirements and available data.
        
        ## Key Insights
        - Analysis completed based on available parameters
        - Multiple data points evaluated
        - Recommendations formulated based on criteria
        
        ## Analysis Results
        - Data points processed successfully
        - Patterns identified in available information
        - Trends analyzed across different metrics
        
        ## Recommendations
        - Continue monitoring relevant metrics
        - Consider additional data sources for enhanced analysis
        - Review findings against market conditions
        - Plan next steps based on insights
        
        *This is a fallback report. For detailed custom analysis, please ensure proper data connectivity and specific requirements.*
        """

    def export_report_to_pdf(self, report_content: str, metadata: ReportMetadata) -> Optional[str]:
        """Export report to PDF format"""
        try:
            if not PDF_GENERATION_AVAILABLE:
                logger.warning("PDF generation libraries not available")
                return self._create_simple_text_pdf(report_content, metadata)
            
            pdf_path = os.path.join(self.reports_dir, "pdf", f"{metadata.report_id}.pdf")
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                textColor=HexColor('#2E8B57')
            )
            story.append(Paragraph(metadata.title, title_style))
            story.append(Spacer(1, 20))
            
            # Add metadata
            meta_style = ParagraphStyle(
                'MetaStyle', 
                parent=styles['Normal'],
                fontSize=10,
                textColor=HexColor('#666666')
            )
            
            meta_info = f"""
            <b>Report ID:</b> {metadata.report_id}<br/>
            <b>Generated:</b> {metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>User ID:</b> {metadata.user_id}<br/>
            <b>Query:</b> {metadata.query}<br/>
            <b>Report Type:</b> {metadata.report_type.replace('_', ' ').title()}
            """
            story.append(Paragraph(meta_info, meta_style))
            story.append(Spacer(1, 30))
            
            # Add report content
            content_lines = report_content.split('\n')
            for line in content_lines:
                if line.strip():
                    if line.startswith('#'):
                        # Header
                        level = line.count('#')
                        text = line.replace('#', '').strip()
                        if level == 1:
                            style = styles['Heading1']
                        elif level == 2:
                            style = styles['Heading2']
                        else:
                            style = styles['Heading3']
                        story.append(Paragraph(text, style))
                        story.append(Spacer(1, 12))
                    else:
                        # Regular paragraph
                        story.append(Paragraph(line, styles['Normal']))
                        story.append(Spacer(1, 6))
            
            # Add charts if available
            if metadata.file_paths:
                story.append(Spacer(1, 30))
                story.append(Paragraph("Charts and Visualizations", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                for chart_name, chart_path in metadata.file_paths.items():
                    if chart_path.endswith('.png') and os.path.exists(chart_path):
                        try:
                            # Add chart to PDF
                            img = Image(chart_path, width=400, height=250)
                            story.append(img)
                            story.append(Paragraph(f"Figure: {chart_name.replace('_', ' ').title()}", 
                                                 styles['Caption']))
                            story.append(Spacer(1, 20))
                        except Exception as e:
                            logger.error(f"Failed to add chart {chart_name} to PDF: {e}")
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return self._create_simple_text_pdf(report_content, metadata)

    def _create_simple_text_pdf(self, content: str, metadata: ReportMetadata) -> str:
        """Create simple text-based PDF when ReportLab is not available"""
        try:
            pdf_path = os.path.join(self.reports_dir, "pdf", f"{metadata.report_id}_simple.txt")
            
            with open(pdf_path, 'w', encoding='utf-8') as f:
                f.write(f"REAL ESTATE REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Title: {metadata.title}\n")
                f.write(f"Generated: {metadata.generated_at}\n")
                f.write(f"Report ID: {metadata.report_id}\n")
                f.write(f"Query: {metadata.query}\n\n")
                f.write("REPORT CONTENT\n")
                f.write("-"*50 + "\n\n")
                f.write(content)
                f.write(f"\n\nCharts Generated: {', '.join(metadata.charts_included)}")
            
            logger.info(f"Simple text report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Simple PDF creation failed: {e}")
            return ""

    # Integration method for the main system
    def process_report_request(self, query: str, user_id: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process report generation request from the main system"""
        try:
            logger.info(f"📊 Processing report request from user {user_id}")
            
            # Prepare data for report generation
            report_data = {
                "properties": search_results.get("properties", []),
                "market_data": search_results.get("market_data", {}),
                "agent_results": search_results.get("agent_results", {}),
                "synthesis": search_results.get("synthesis", {})
            }
            
            # Generate the report
            report_result = self.generate_report(query, user_id, report_data)
            
            if report_result.get("success"):
                logger.info(f"✅ Report generated successfully: {report_result.get('report_type')}")
                
                # Return comprehensive result
                return {
                    "success": True,
                    "report_generated": True,
                    "report_type": report_result.get("report_type"),
                    "report_content": report_result.get("content"),
                    "metadata": report_result.get("metadata"),
                    "charts": report_result.get("charts", {}),
                    "pdf_path": report_result.get("pdf_path"),
                    "summary": report_result.get("summary"),
                    "download_links": self._generate_download_links(report_result),
                    "report_id": report_result.get("metadata", {}).get("report_id")
                }
            else:
                logger.error(f"❌ Report generation failed: {report_result.get('error')}")
                return {
                    "success": False,
                    "report_generated": False,
                    "error": report_result.get("error"),
                    "fallback_summary": report_result.get("fallback_summary")
                }
                
        except Exception as e:
            logger.error(f"Report request processing failed: {e}")
            return {
                "success": False,
                "report_generated": False,
                "error": str(e),
                "fallback_summary": "Report generation encountered an error"
            }

    def _generate_download_links(self, report_result: Dict) -> Dict[str, str]:
        """Generate download links for report files"""
        download_links = {}
        
        try:
            # PDF download link
            if report_result.get("pdf_path"):
                download_links["pdf"] = f"file://{report_result['pdf_path']}"
            
            # Chart download links
            charts = report_result.get("charts", {})
            for chart_name, chart_path in charts.items():
                download_links[f"chart_{chart_name}"] = f"file://{chart_path}"
            
            return download_links
            
        except Exception as e:
            logger.error(f"Download link generation failed: {e}")
            return {}

    def get_available_report_types(self) -> List[Dict[str, str]]:
        """Get list of available report types"""
        return [
            {
                "type": "market_analysis",
                "name": "Market Analysis Report",
                "description": "Comprehensive market trends and price analysis"
            },
            {
                "type": "user_preference_report", 
                "name": "Personal Preference Analysis",
                "description": "Personalized analysis of your search patterns and preferences"
            },
            {
                "type": "investment_analysis",
                "name": "Investment Analysis Report", 
                "description": "ROI calculations and investment potential analysis"
            },
            {
                "type": "renovation_estimate",
                "name": "Renovation Cost Estimation",
                "description": "BHK-wise renovation cost calculation with detailed breakdown"
            },
            {
                "type": "location_report",
                "name": "Location Analysis Report",
                "description": "Area-specific market insights and trends"
            },
            {
                "type": "custom",
                "name": "Custom Report",
                "description": "Tailored analysis based on specific requirements"
            }
        ]

    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of a generated report"""
        try:
            # Check if report files exist
            pdf_path = os.path.join(self.reports_dir, "pdf", f"{report_id}.pdf")
            charts_dir = os.path.join(self.reports_dir, "charts")
            
            return {
                "report_id": report_id,
                "pdf_exists": os.path.exists(pdf_path),
                "pdf_path": pdf_path if os.path.exists(pdf_path) else None,
                "charts_available": len([f for f in os.listdir(charts_dir) if report_id in f]) if os.path.exists(charts_dir) else 0,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Report status check failed: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    def test_report_generation():
        """Test the report generation functionality"""
        
        # Initialize agent
        agent = ReportGenerationAgent()
        
        # Test data
        test_data = {
            "properties": [
                {
                    "id": "1",
                    "title": "Luxury Apartment in Mumbai",
                    "price": 35000000,
                    "location": "Mumbai",
                    "property_type": "Apartment", 
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "area": 1200
                },
                {
                    "id": "2", 
                    "title": "Villa in Bangalore",
                    "price": 45000000,
                    "location": "Bangalore",
                    "property_type": "Villa",
                    "bedrooms": 4,
                    "bathrooms": 3,
                    "area": 2000
                }
            ]
        }
        
        # Test different report types
        test_queries = [
            "Generate a market analysis report for Mumbai properties",
            "Compare these properties and create a detailed comparison report", 
            "Provide investment analysis with ROI calculations",
            "Create a location-based analysis report"
        ]
        
        print("🏠📊 Testing Report Generation Agent")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: {query}")
            print("-"*40)
            
            try:
                result = agent.process_report_request(query, f"test_user_{i}", {"properties": test_data["properties"]})
                
                if result.get("success"):
                    print(f"✅ Report generated successfully")
                    print(f"📊 Report Type: {result.get('report_type')}")
                    print(f"📄 PDF Path: {result.get('pdf_path')}")
                    print(f"📈 Charts: {len(result.get('charts', {}))}")
                else:
                    print(f"❌ Report generation failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        print(f"\n🎉 Report generation testing completed!")
        print(f"📁 Reports saved in: {agent.reports_dir}")
    
    # Run test
    test_report_generation()