#!/usr/bin/env python3
"""
Renovation Estimation Agent

Purpose: Given property size/rooms, estimate renovation costs using BHK-wise formulas
Features:
- BHK-based cost calculation
- Room-specific pricing models  
- Different renovation levels (Basic, Premium, Luxury)
- Detailed cost breakdown by categories
- Area-based and room-count-based estimation
"""

import os
import sys
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RenovationEstimate:
    """Data class for renovation estimation results"""
    property_type: str
    total_area: float
    bhk_config: str
    renovation_level: str
    total_cost: float
    cost_per_sqft: float
    room_breakdown: Dict[str, float]
    category_breakdown: Dict[str, float]
    timeline_weeks: int
    estimated_at: datetime
    recommendations: List[str]
    cost_factors: Dict[str, Any]

class RenovationEstimationAgent:
    """
    Comprehensive Renovation Estimation Agent
    Provides accurate cost estimation for property renovations based on BHK and area
    """
    
    def __init__(self):
        self.base_costs = self._initialize_base_costs()
        self.renovation_levels = self._initialize_renovation_levels()
        self.room_multipliers = self._initialize_room_multipliers()
        self.category_weights = self._initialize_category_weights()
        
        logger.info("🔨🏠 Renovation Estimation Agent initialized")
    
    def _initialize_base_costs(self) -> Dict[str, float]:
        """Initialize base cost per sqft for different BHK configurations"""
        return {
            # Cost per square foot in INR
            "1BHK": 800,
            "2BHK": 900,
            "3BHK": 1000,
            "4BHK": 1100,
            "5BHK": 1200,
            "villa": 1300,
            "studio": 700,
            "duplex": 1150,
            "penthouse": 1400
        }
    
    def _initialize_renovation_levels(self) -> Dict[str, Dict[str, float]]:
        """Initialize renovation level multipliers and descriptions"""
        return {
            "basic": {
                "multiplier": 1.0,
                "description": "Essential repairs and basic improvements",
                "features": ["Paint refresh", "Basic fixtures", "Minor repairs"],
                "timeline_factor": 1.0
            },
            "premium": {
                "multiplier": 1.8,
                "description": "Comprehensive renovation with quality materials",
                "features": ["Quality finishes", "Modern fixtures", "Structural improvements"],
                "timeline_factor": 1.5
            },
            "luxury": {
                "multiplier": 3.2,
                "description": "High-end renovation with premium materials",
                "features": ["Premium finishes", "Designer fixtures", "Smart home features"],
                "timeline_factor": 2.2
            },
            "complete": {
                "multiplier": 4.5,
                "description": "Complete makeover with top-tier materials",
                "features": ["Luxury finishes", "Custom work", "Full automation"],
                "timeline_factor": 3.0
            }
        }
    
    def _initialize_room_multipliers(self) -> Dict[str, float]:
        """Initialize cost multipliers for different room types"""
        return {
            "living_room": 1.2,
            "bedroom": 1.0,
            "kitchen": 2.5,  # Kitchens are most expensive
            "bathroom": 2.0,  # Bathrooms are second most expensive
            "dining_room": 0.8,
            "balcony": 0.6,
            "study_room": 0.9,
            "utility_room": 0.7,
            "master_bedroom": 1.3
        }
    
    def _initialize_category_weights(self) -> Dict[str, float]:
        """Initialize weights for different renovation categories"""
        return {
            "flooring": 0.20,
            "painting": 0.15,
            "electrical": 0.18,
            "plumbing": 0.15,
            "kitchen_fixtures": 0.12,
            "bathroom_fixtures": 0.10,
            "doors_windows": 0.07,
            "miscellaneous": 0.03
        }
    
    def estimate_renovation_cost(
        self, 
        property_type: str,
        bhk_config: str,
        total_area: float,
        renovation_level: str = "premium",
        room_details: Optional[Dict[str, int]] = None,
        location_factor: float = 1.0,
        additional_requirements: Optional[List[str]] = None
    ) -> RenovationEstimate:
        """
        Main method to estimate renovation costs
        
        Args:
            property_type: Type of property (apartment, villa, etc.)
            bhk_config: BHK configuration (1BHK, 2BHK, etc.)
            total_area: Total area in square feet
            renovation_level: Level of renovation (basic, premium, luxury, complete)
            room_details: Dictionary with room counts (optional)
            location_factor: Factor for location-based pricing (1.0 = average)
            additional_requirements: List of special requirements
        """
        try:
            logger.info(f"🔨 Estimating renovation cost for {bhk_config} {property_type}")
            
            # Get base cost per sqft
            base_cost_per_sqft = self.base_costs.get(bhk_config.lower(), self.base_costs["2BHK"])
            
            # Apply renovation level multiplier
            renovation_multiplier = self.renovation_levels[renovation_level]["multiplier"]
            cost_per_sqft = base_cost_per_sqft * renovation_multiplier * location_factor
            
            # Calculate basic total cost
            basic_total_cost = total_area * cost_per_sqft
            
            # Calculate room-wise breakdown
            room_breakdown = self._calculate_room_breakdown(
                bhk_config, total_area, cost_per_sqft, room_details
            )
            
            # Calculate category-wise breakdown
            category_breakdown = self._calculate_category_breakdown(basic_total_cost)
            
            # Apply additional requirements
            additional_cost = self._calculate_additional_costs(
                additional_requirements, basic_total_cost
            )
            
            # Final total cost
            total_cost = basic_total_cost + additional_cost
            
            # Calculate timeline
            timeline_weeks = self._calculate_timeline(
                total_area, renovation_level, bhk_config
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                bhk_config, renovation_level, total_area, total_cost
            )
            
            # Prepare cost factors
            cost_factors = {
                "base_cost_per_sqft": base_cost_per_sqft,
                "renovation_multiplier": renovation_multiplier,
                "location_factor": location_factor,
                "additional_cost": additional_cost,
                "total_area": total_area
            }
            
            return RenovationEstimate(
                property_type=property_type,
                total_area=total_area,
                bhk_config=bhk_config,
                renovation_level=renovation_level,
                total_cost=total_cost,
                cost_per_sqft=cost_per_sqft,
                room_breakdown=room_breakdown,
                category_breakdown=category_breakdown,
                timeline_weeks=timeline_weeks,
                estimated_at=datetime.now(),
                recommendations=recommendations,
                cost_factors=cost_factors
            )
            
        except Exception as e:
            logger.error(f"Renovation cost estimation failed: {e}")
            raise
    
    def _calculate_room_breakdown(
        self, 
        bhk_config: str, 
        total_area: float, 
        cost_per_sqft: float,
        room_details: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """Calculate cost breakdown by rooms"""
        
        # Standard room configuration based on BHK
        standard_rooms = self._get_standard_room_config(bhk_config)
        
        if room_details:
            # Use provided room details
            rooms = room_details
        else:
            # Use standard configuration
            rooms = standard_rooms
        
        room_breakdown = {}
        
        # Estimate area per room based on total area and room count
        total_rooms = sum(rooms.values())
        
        for room_type, count in rooms.items():
            if count > 0:
                # Calculate area for this room type
                room_multiplier = self.room_multipliers.get(room_type, 1.0)
                
                if room_type == "kitchen":
                    # Kitchen gets fixed percentage of total area
                    area_per_room = total_area * 0.12  # 12% for kitchen
                elif room_type == "bathroom":
                    # Bathroom gets fixed percentage
                    area_per_room = total_area * 0.08  # 8% per bathroom
                else:
                    # Other rooms share remaining area
                    remaining_percentage = 1.0 - (0.12 + (rooms.get("bathroom", 1) * 0.08))
                    other_rooms = sum(v for k, v in rooms.items() if k not in ["kitchen", "bathroom"])
                    if other_rooms > 0:
                        area_per_room = (total_area * remaining_percentage) / other_rooms
                    else:
                        area_per_room = total_area * 0.15
                
                # Calculate cost for this room type
                room_cost = area_per_room * cost_per_sqft * room_multiplier * count
                room_breakdown[f"{room_type} ({count})"] = round(room_cost, 2)
        
        return room_breakdown
    
    def _get_standard_room_config(self, bhk_config: str) -> Dict[str, int]:
        """Get standard room configuration for BHK"""
        config_map = {
            "studio": {"living_room": 1, "kitchen": 1, "bathroom": 1},
            "1BHK": {"bedroom": 1, "living_room": 1, "kitchen": 1, "bathroom": 1},
            "2BHK": {"bedroom": 2, "living_room": 1, "kitchen": 1, "bathroom": 2},
            "3BHK": {"bedroom": 2, "master_bedroom": 1, "living_room": 1, "kitchen": 1, "bathroom": 2, "balcony": 1},
            "4BHK": {"bedroom": 2, "master_bedroom": 1, "living_room": 1, "dining_room": 1, "kitchen": 1, "bathroom": 3, "balcony": 2},
            "5BHK": {"bedroom": 3, "master_bedroom": 1, "living_room": 1, "dining_room": 1, "kitchen": 1, "bathroom": 4, "balcony": 2, "study_room": 1},
            "villa": {"bedroom": 3, "master_bedroom": 1, "living_room": 2, "dining_room": 1, "kitchen": 1, "bathroom": 4, "balcony": 3, "utility_room": 1}
        }
        
        return config_map.get(bhk_config.lower(), config_map["2BHK"])
    
    def _calculate_category_breakdown(self, total_cost: float) -> Dict[str, float]:
        """Calculate cost breakdown by renovation categories"""
        category_breakdown = {}
        
        for category, weight in self.category_weights.items():
            cost = total_cost * weight
            category_breakdown[category.replace('_', ' ').title()] = round(cost, 2)
        
        return category_breakdown
    
    def _calculate_additional_costs(
        self, 
        additional_requirements: Optional[List[str]], 
        base_cost: float
    ) -> float:
        """Calculate additional costs for special requirements"""
        if not additional_requirements:
            return 0.0
        
        additional_cost_map = {
            "smart_home": base_cost * 0.15,
            "solar_panels": base_cost * 0.20,
            "home_theater": base_cost * 0.08,
            "gym_setup": base_cost * 0.06,
            "swimming_pool": base_cost * 0.40,
            "garden_landscaping": base_cost * 0.10,
            "security_system": base_cost * 0.05,
            "ac_installation": base_cost * 0.12,
            "modular_furniture": base_cost * 0.25
        }
        
        total_additional = 0.0
        for requirement in additional_requirements:
            cost = additional_cost_map.get(requirement.lower().replace(' ', '_'), 0.0)
            total_additional += cost
        
        return total_additional
    
    def _calculate_timeline(self, area: float, renovation_level: str, bhk_config: str) -> int:
        """Calculate renovation timeline in weeks"""
        # Base timeline calculation
        base_weeks = 4 + (area / 200)  # 4 weeks base + 1 week per 200 sqft
        
        # Apply renovation level factor
        level_factor = self.renovation_levels[renovation_level]["timeline_factor"]
        
        # Apply BHK complexity factor
        bhk_factors = {
            "studio": 0.7, "1BHK": 0.8, "2BHK": 1.0, "3BHK": 1.2, 
            "4BHK": 1.4, "5BHK": 1.6, "villa": 1.8, "penthouse": 1.5
        }
        bhk_factor = bhk_factors.get(bhk_config.lower(), 1.0)
        
        total_weeks = base_weeks * level_factor * bhk_factor
        
        return max(int(round(total_weeks)), 2)  # Minimum 2 weeks
    
    def _generate_recommendations(
        self, 
        bhk_config: str, 
        renovation_level: str, 
        area: float, 
        total_cost: float
    ) -> List[str]:
        """Generate personalized renovation recommendations"""
        recommendations = []
        
        # Cost-based recommendations
        cost_per_sqft = total_cost / area
        if cost_per_sqft > 3000:
            recommendations.append("Consider phased renovation to spread costs over time")
        
        # BHK-based recommendations
        if "BHK" in bhk_config:
            bhk_num = int(bhk_config[0])
            if bhk_num >= 3:
                recommendations.append("Focus on master bedroom and living areas for maximum impact")
            if bhk_num >= 4:
                recommendations.append("Consider converting one bedroom to a multipurpose room")
        
        # Level-based recommendations
        if renovation_level == "luxury":
            recommendations.append("Invest in smart home automation for modern convenience")
            recommendations.append("Consider premium lighting solutions for ambiance")
        elif renovation_level == "basic":
            recommendations.append("Prioritize kitchen and bathroom upgrades for best ROI")
        
        # Area-based recommendations
        if area > 2000:
            recommendations.append("Plan for efficient storage solutions in larger spaces")
        elif area < 800:
            recommendations.append("Use space-saving furniture and light colors to enhance space perception")
        
        # General recommendations
        recommendations.extend([
            "Get multiple quotes from certified contractors",
            "Plan for 10-15% buffer in budget for unforeseen expenses",
            "Consider energy-efficient fixtures to reduce long-term costs"
        ])
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def get_bhk_wise_cost_comparison(self, area: float, renovation_level: str = "premium") -> Dict[str, Dict[str, Any]]:
        """Get cost comparison across different BHK configurations"""
        bhk_options = ["1BHK", "2BHK", "3BHK", "4BHK", "5BHK"]
        comparison = {}
        
        for bhk in bhk_options:
            estimate = self.estimate_renovation_cost(
                property_type="apartment",
                bhk_config=bhk,
                total_area=area,
                renovation_level=renovation_level
            )
            
            comparison[bhk] = {
                "total_cost": estimate.total_cost,
                "cost_per_sqft": estimate.cost_per_sqft,
                "timeline_weeks": estimate.timeline_weeks,
                "estimated_rooms": len(estimate.room_breakdown)
            }
        
        return comparison
    
    def get_renovation_level_comparison(
        self, 
        property_type: str, 
        bhk_config: str, 
        area: float
    ) -> Dict[str, Dict[str, Any]]:
        """Get cost comparison across renovation levels"""
        levels = ["basic", "premium", "luxury", "complete"]
        comparison = {}
        
        for level in levels:
            estimate = self.estimate_renovation_cost(
                property_type=property_type,
                bhk_config=bhk_config,
                total_area=area,
                renovation_level=level
            )
            
            comparison[level] = {
                "total_cost": estimate.total_cost,
                "cost_per_sqft": estimate.cost_per_sqft,
                "timeline_weeks": estimate.timeline_weeks,
                "description": self.renovation_levels[level]["description"],
                "features": self.renovation_levels[level]["features"]
            }
        
        return comparison
    
    def generate_detailed_estimate_report(self, estimate: RenovationEstimate) -> str:
        """Generate a detailed text report of the renovation estimate"""
        report = f"""
🔨 RENOVATION COST ESTIMATION REPORT
{'='*50}

🏠 PROPERTY DETAILS:
• Property Type: {estimate.property_type.title()}
• Configuration: {estimate.bhk_config}
• Total Area: {estimate.total_area:,.0f} sq ft
• Renovation Level: {estimate.renovation_level.title()}

💰 COST SUMMARY:
• Total Renovation Cost: ₹{estimate.total_cost:,.0f}
• Cost per Square Foot: ₹{estimate.cost_per_sqft:,.0f}
• Timeline: {estimate.timeline_weeks} weeks

🏗️ ROOM-WISE BREAKDOWN:
{self._format_breakdown(estimate.room_breakdown)}

📋 CATEGORY-WISE BREAKDOWN:
{self._format_breakdown(estimate.category_breakdown)}

💡 RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in estimate.recommendations])}

📊 COST FACTORS:
• Base Cost per sq ft: ₹{estimate.cost_factors['base_cost_per_sqft']:,.0f}
• Renovation Level Multiplier: {estimate.cost_factors['renovation_multiplier']:.1f}x
• Location Factor: {estimate.cost_factors['location_factor']:.1f}x
• Additional Features Cost: ₹{estimate.cost_factors['additional_cost']:,.0f}

📅 Estimated on: {estimate.estimated_at.strftime('%Y-%m-%d %H:%M:%S')}

Note: Estimates are indicative and may vary based on material quality, 
contractor rates, and market conditions. Always get professional quotes.
"""
        return report
    
    def _format_breakdown(self, breakdown: Dict[str, float]) -> str:
        """Format cost breakdown for display"""
        if not breakdown:
            return "• No detailed breakdown available"
        
        formatted = []
        for item, cost in breakdown.items():
            formatted.append(f"• {item}: ₹{cost:,.0f}")
        
        return '\n'.join(formatted)

# Example usage and testing
if __name__ == "__main__":
    def test_renovation_estimation():
        """Test the renovation estimation functionality"""
        print("🔨🏠 Testing Renovation Estimation Agent")
        print("="*50)
        
        # Initialize agent
        agent = RenovationEstimationAgent()
        
        # Test scenarios
        test_cases = [
            {
                "property_type": "apartment",
                "bhk_config": "2BHK", 
                "total_area": 1000,
                "renovation_level": "premium",
                "description": "Standard 2BHK apartment renovation"
            },
            {
                "property_type": "villa",
                "bhk_config": "4BHK",
                "total_area": 2500,
                "renovation_level": "luxury",
                "additional_requirements": ["smart_home", "swimming_pool"],
                "description": "Luxury villa renovation with extras"
            },
            {
                "property_type": "apartment",
                "bhk_config": "1BHK",
                "total_area": 650,
                "renovation_level": "basic",
                "description": "Budget 1BHK renovation"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🧪 TEST CASE {i}: {case['description']}")
            print("-"*40)
            
            try:
                estimate = agent.estimate_renovation_cost(
                    property_type=case["property_type"],
                    bhk_config=case["bhk_config"],
                    total_area=case["total_area"],
                    renovation_level=case["renovation_level"],
                    additional_requirements=case.get("additional_requirements")
                )
                
                print(f"✅ Estimate Generated Successfully!")
                print(f"💰 Total Cost: ₹{estimate.total_cost:,.0f}")
                print(f"📏 Cost per sq ft: ₹{estimate.cost_per_sqft:,.0f}")
                print(f"⏱️ Timeline: {estimate.timeline_weeks} weeks")
                print(f"🏠 Rooms: {len(estimate.room_breakdown)}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Test comparisons
        print(f"\n📊 BHK-wise Comparison (1500 sq ft, Premium):")
        comparison = agent.get_bhk_wise_cost_comparison(1500, "premium")
        for bhk, data in comparison.items():
            print(f"• {bhk}: ₹{data['total_cost']:,.0f} ({data['timeline_weeks']} weeks)")
        
        print(f"\n🎯 Renovation Level Comparison (2BHK, 1200 sq ft):")
        level_comparison = agent.get_renovation_level_comparison("apartment", "2BHK", 1200)
        for level, data in level_comparison.items():
            print(f"• {level.title()}: ₹{data['total_cost']:,.0f}")
    
    # Run tests
    test_renovation_estimation()