"""
Memory Component - Real Estate Search Engine
Persist user preferences across sessions (locations, budget, saved properties)

This component demonstrates multiple types of memory across chat sessions:
1. User Preference Memory (location, budget, property type preferences)
2. Search History Memory (past queries and results)  
3. Saved Properties Memory (bookmarked properties)
4. Session Context Memory (current conversation state)
5. Interaction Pattern Memory (user behavior analysis)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    """User preference data structure"""
    user_id: str
    preferred_locations: List[str]
    budget_range: Dict[str, float]  # {"min": 50_00_000, "max": 1_00_00_000}
    preferred_property_types: List[str]  # ["apartment", "villa", "studio"]
    preferred_bhk_configs: List[str]  # ["2BHK", "3BHK"]
    preferred_amenities: List[str]  # ["parking", "gym", "pool"]
    created_at: str
    updated_at: str

@dataclass
class SearchHistoryItem:
    """Search history item"""
    search_id: str
    user_id: str
    query: str
    results_count: int
    timestamp: str
    query_type: str  # "search", "analysis", "renovation"
    satisfied: Optional[bool] = None  # User satisfaction with results

@dataclass
class SavedProperty:
    """Saved/bookmarked property"""
    property_id: str
    user_id: str
    title: str
    location: str
    price: float
    property_type: str
    saved_at: str
    notes: Optional[str] = None
    tags: List[str] = None

@dataclass
class SessionContext:
    """Current session conversation context"""
    session_id: str
    user_id: str
    current_intent: str
    active_filters: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    started_at: str
    last_activity: str

@dataclass
class InteractionPattern:
    """User interaction pattern analysis"""
    user_id: str
    total_sessions: int
    total_queries: int
    common_query_types: List[str]
    preferred_response_format: str  # "detailed", "concise", "charts"
    peak_usage_times: List[str]
    analyzed_at: str

class MemoryComponent:
    """
    Comprehensive Memory Component for Real Estate Search Engine
    
    Features:
    - Multi-type memory persistence across sessions
    - User preference learning and adaptation
    - Search history tracking and analysis
    - Property bookmarking and management
    - Session context maintenance
    - Interaction pattern recognition
    """
    
    def __init__(self, memory_dir: str = "memory_storage"):
        """Initialize memory component with persistent storage"""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory storage files
        self.preferences_file = self.memory_dir / "user_preferences.json"
        self.search_history_file = self.memory_dir / "search_history.json" 
        self.saved_properties_file = self.memory_dir / "saved_properties.json"
        self.session_contexts_file = self.memory_dir / "session_contexts.json"
        self.interaction_patterns_file = self.memory_dir / "interaction_patterns.json"
        
        # In-memory caches for performance
        self.preferences_cache = {}
        self.active_sessions = {}
        
        # Load existing data
        self._load_memory_data()
        
        logger.info("🧠 Memory Component initialized with persistent storage")

    def _load_memory_data(self):
        """Load existing memory data from files"""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    self.preferences_cache = {
                        user_id: UserPreference(**prefs) 
                        for user_id, prefs in data.items()
                    }
            logger.info(f"📚 Loaded {len(self.preferences_cache)} user preferences")
        except Exception as e:
            logger.error(f"Failed to load memory data: {e}")
            self.preferences_cache = {}

    def _save_preferences(self):
        """Save user preferences to persistent storage"""
        try:
            data = {
                user_id: asdict(prefs) 
                for user_id, prefs in self.preferences_cache.items()
            }
            with open(self.preferences_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def _append_to_file(self, file_path: Path, data: Dict):
        """Append data to JSON file (for history tracking)"""
        try:
            existing_data = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            
            existing_data.append(data)
            
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to append to {file_path}: {e}")

    # === USER PREFERENCE MANAGEMENT ===

    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences from memory"""
        return self.preferences_cache.get(user_id)

    def update_user_preferences(self, user_id: str, **preference_updates) -> UserPreference:
        """Update user preferences and persist to storage"""
        now = datetime.now().isoformat()
        
        if user_id in self.preferences_cache:
            # Update existing preferences
            prefs = self.preferences_cache[user_id]
            for key, value in preference_updates.items():
                if hasattr(prefs, key):
                    setattr(prefs, key, value)
            prefs.updated_at = now
        else:
            # Create new preferences
            prefs = UserPreference(
                user_id=user_id,
                preferred_locations=preference_updates.get('preferred_locations', []),
                budget_range=preference_updates.get('budget_range', {}),
                preferred_property_types=preference_updates.get('preferred_property_types', []),
                preferred_bhk_configs=preference_updates.get('preferred_bhk_configs', []),
                preferred_amenities=preference_updates.get('preferred_amenities', []),
                created_at=now,
                updated_at=now
            )
        
        self.preferences_cache[user_id] = prefs
        self._save_preferences()
        
        logger.info(f"✅ Updated preferences for user {user_id}")
        return prefs

    def learn_preferences_from_query(self, user_id: str, query: str, selected_results: List[Dict] = None):
        """Learn user preferences from their queries and selections"""
        query_lower = query.lower()
        
        # Extract preferences from query
        inferred_prefs = {}
        
        # Location preferences
        locations = []
        for city in ["mumbai", "delhi", "bangalore", "pune", "chennai", "hyderabad", "kolkata"]:
            if city in query_lower:
                locations.append(city.title())
        if locations:
            inferred_prefs['preferred_locations'] = locations
        
        # BHK preferences  
        bhk_configs = []
        for bhk in ["studio", "1bhk", "2bhk", "3bhk", "4bhk", "5bhk"]:
            if bhk in query_lower:
                bhk_configs.append(bhk.upper())
        if bhk_configs:
            inferred_prefs['preferred_bhk_configs'] = bhk_configs
        
        # Property type preferences
        property_types = []
        for ptype in ["apartment", "villa", "house", "studio", "flat"]:
            if ptype in query_lower:
                property_types.append(ptype)
        if property_types:
            inferred_prefs['preferred_property_types'] = property_types
        
        # Budget extraction (simplified)
        if "under" in query_lower or "below" in query_lower:
            # Try to extract budget range
            import re
            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:lakh|crore|lac|cr)', query_lower)
            if numbers:
                budget_val = float(numbers[0])
                if "crore" in query_lower or "cr" in query_lower:
                    budget_val *= 10000000
                elif "lakh" in query_lower or "lac" in query_lower:
                    budget_val *= 100000
                inferred_prefs['budget_range'] = {"max": budget_val}
        
        # Update preferences if any were inferred
        if inferred_prefs:
            existing_prefs = self.get_user_preferences(user_id)
            if existing_prefs:
                # Merge with existing preferences
                for key, value in inferred_prefs.items():
                    if key in ['preferred_locations', 'preferred_bhk_configs', 'preferred_property_types']:
                        # Merge lists
                        existing_list = getattr(existing_prefs, key, [])
                        merged_list = list(set(existing_list + value))
                        inferred_prefs[key] = merged_list
            
            self.update_user_preferences(user_id, **inferred_prefs)
            logger.info(f"🎯 Learned preferences for {user_id}: {inferred_prefs}")

    # === SEARCH HISTORY MANAGEMENT ===

    def add_search_history(self, user_id: str, query: str, results_count: int, query_type: str = "search") -> str:
        """Add search to history"""
        search_id = f"search_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        search_item = SearchHistoryItem(
            search_id=search_id,
            user_id=user_id,
            query=query,
            results_count=results_count,
            timestamp=datetime.now().isoformat(),
            query_type=query_type
        )
        
        self._append_to_file(self.search_history_file, asdict(search_item))
        
        # Learn preferences from this search
        self.learn_preferences_from_query(user_id, query)
        
        logger.info(f"📝 Added search history: {search_id}")
        return search_id

    def get_search_history(self, user_id: str, limit: int = 10) -> List[SearchHistoryItem]:
        """Get user's search history"""
        try:
            if not self.search_history_file.exists():
                return []
            
            with open(self.search_history_file, 'r') as f:
                all_history = json.load(f)
            
            user_history = [
                SearchHistoryItem(**item) for item in all_history 
                if item.get('user_id') == user_id
            ]
            
            # Sort by timestamp, most recent first
            user_history.sort(key=lambda x: x.timestamp, reverse=True)
            
            return user_history[:limit]
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []

    # === SAVED PROPERTIES MANAGEMENT ===

    def save_property(self, user_id: str, property_data: Dict, notes: str = None, tags: List[str] = None) -> str:
        """Save/bookmark a property"""
        property_id = f"prop_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        saved_property = SavedProperty(
            property_id=property_id,
            user_id=user_id,
            title=property_data.get('title', 'Untitled Property'),
            location=property_data.get('location', 'Unknown'),
            price=property_data.get('price', 0),
            property_type=property_data.get('property_type', 'Unknown'),
            saved_at=datetime.now().isoformat(),
            notes=notes,
            tags=tags or []
        )
        
        self._append_to_file(self.saved_properties_file, asdict(saved_property))
        
        logger.info(f"💾 Saved property: {property_id} for user {user_id}")
        return property_id

    def get_saved_properties(self, user_id: str, tags: List[str] = None) -> List[SavedProperty]:
        """Get user's saved properties, optionally filtered by tags"""
        try:
            if not self.saved_properties_file.exists():
                return []
            
            with open(self.saved_properties_file, 'r') as f:
                all_properties = json.load(f)
            
            user_properties = [
                SavedProperty(**prop) for prop in all_properties 
                if prop.get('user_id') == user_id
            ]
            
            # Filter by tags if provided
            if tags:
                user_properties = [
                    prop for prop in user_properties 
                    if any(tag in (prop.tags or []) for tag in tags)
                ]
            
            return user_properties
        except Exception as e:
            logger.error(f"Failed to get saved properties: {e}")
            return []

    # === SESSION CONTEXT MANAGEMENT ===

    def start_session(self, user_id: str) -> str:
        """Start a new conversation session"""
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            current_intent="",
            active_filters={},
            conversation_history=[],
            started_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat()
        )
        
        self.active_sessions[session_id] = session_context
        
        logger.info(f"🚀 Started session: {session_id} for user {user_id}")
        return session_id

    def update_session_context(self, session_id: str, **context_updates):
        """Update session context"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            for key, value in context_updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_activity = datetime.now().isoformat()
            
            logger.info(f"📝 Updated session context: {session_id}")

    def add_conversation_turn(self, session_id: str, user_message: str, assistant_response: str):
        """Add conversation turn to session history"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_message,
                "assistant": assistant_response
            })
            session.last_activity = datetime.now().isoformat()

    def end_session(self, session_id: str):
        """End session and persist to storage"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            self._append_to_file(self.session_contexts_file, asdict(session))
            del self.active_sessions[session_id]
            
            logger.info(f"🔚 Ended session: {session_id}")

    # === INTERACTION PATTERN ANALYSIS ===

    def analyze_interaction_patterns(self, user_id: str) -> InteractionPattern:
        """Analyze user interaction patterns from history"""
        search_history = self.get_search_history(user_id, limit=100)
        
        # Analyze patterns
        total_queries = len(search_history)
        query_types = [item.query_type for item in search_history]
        common_types = list(set(query_types))
        
        # Simple pattern analysis
        pattern = InteractionPattern(
            user_id=user_id,
            total_sessions=len(set(item.timestamp[:10] for item in search_history)),  # Approximate
            total_queries=total_queries,
            common_query_types=common_types,
            preferred_response_format="detailed",  # Could be inferred
            peak_usage_times=[],  # Could analyze timestamps
            analyzed_at=datetime.now().isoformat()
        )
        
        # Save pattern analysis
        self._append_to_file(self.interaction_patterns_file, asdict(pattern))
        
        return pattern

    # === MEMORY RETRIEVAL FOR AGENTS ===

    def get_user_context_for_query(self, user_id: str, current_query: str) -> Dict[str, Any]:
        """Get comprehensive user context for current query"""
        context = {
            "user_id": user_id,
            "current_query": current_query,
            "preferences": None,
            "recent_searches": [],
            "saved_properties": [],
            "interaction_patterns": None
        }
        
        # Get user preferences
        prefs = self.get_user_preferences(user_id)
        if prefs:
            context["preferences"] = asdict(prefs)
        
        # Get recent search history
        recent_searches = self.get_search_history(user_id, limit=5)
        context["recent_searches"] = [asdict(search) for search in recent_searches]
        
        # Get saved properties
        saved_props = self.get_saved_properties(user_id)
        context["saved_properties"] = [asdict(prop) for prop in saved_props[-5:]]  # Last 5
        
        # Get interaction patterns
        try:
            patterns = self.analyze_interaction_patterns(user_id)
            context["interaction_patterns"] = asdict(patterns)
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
        
        return context

    def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations based on user memory"""
        context = self.get_user_context_for_query(user_id, "")
        
        recommendations = {
            "preferred_searches": [],
            "suggested_properties": [],
            "budget_insights": {},
            "location_trends": []
        }
        
        prefs = context.get("preferences")
        if prefs:
            # Generate search suggestions based on preferences
            if prefs["preferred_locations"] and prefs["preferred_bhk_configs"]:
                for location in prefs["preferred_locations"]:
                    for bhk in prefs["preferred_bhk_configs"]:
                        recommendations["preferred_searches"].append(
                            f"Find {bhk} apartments in {location}"
                        )
        
        return recommendations

    # === MEMORY STATISTICS AND INSIGHTS ===

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory component statistics"""
        stats = {
            "total_users": len(self.preferences_cache),
            "active_sessions": len(self.active_sessions),
            "memory_files": {
                "preferences": self.preferences_file.exists(),
                "search_history": self.search_history_file.exists(),
                "saved_properties": self.saved_properties_file.exists(),
                "session_contexts": self.session_contexts_file.exists(),
                "interaction_patterns": self.interaction_patterns_file.exists()
            }
        }
        
        # Count records in each file
        for file_name, file_path in [
            ("search_history_count", self.search_history_file),
            ("saved_properties_count", self.saved_properties_file),
            ("session_contexts_count", self.session_contexts_file),
            ("interaction_patterns_count", self.interaction_patterns_file)
        ]:
            try:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        stats[file_name] = len(data)
                else:
                    stats[file_name] = 0
            except:
                stats[file_name] = 0
        
        return stats

if __name__ == "__main__":
    # Demo of memory component functionality
    print("🧠 MEMORY COMPONENT DEMONSTRATION")
    print("=" * 50)
    
    # Initialize memory component
    memory = MemoryComponent()
    
    # Demo user
    user_id = "demo_user_123"
    
    print(f"\n1. 👤 USER PREFERENCE MANAGEMENT")
    print("-" * 30)
    
    # Update user preferences
    prefs = memory.update_user_preferences(
        user_id,
        preferred_locations=["Mumbai", "Pune"], 
        budget_range={"min": 5000000, "max": 10000000},
        preferred_bhk_configs=["2BHK", "3BHK"],
        preferred_property_types=["apartment"]
    )
    print(f"✅ Created preferences for {user_id}")
    
    # Get preferences
    retrieved_prefs = memory.get_user_preferences(user_id)
    print(f"📚 Retrieved preferences: {retrieved_prefs.preferred_locations}")
    
    print(f"\n2. 📝 SEARCH HISTORY TRACKING")
    print("-" * 30)
    
    # Add search history
    memory.add_search_history(user_id, "Find 2BHK apartments in Mumbai under 80 lakhs", 15, "search")
    memory.add_search_history(user_id, "Estimate renovation cost for 2BHK", 1, "renovation")
    
    # Get search history
    history = memory.get_search_history(user_id)
    print(f"📜 Search history: {len(history)} searches")
    for search in history:
        print(f"   - {search.query} ({search.query_type})")
    
    print(f"\n3. 💾 SAVED PROPERTIES MANAGEMENT")
    print("-" * 30)
    
    # Save a property
    property_data = {
        "title": "2BHK Apartment in Bandra",
        "location": "Mumbai",
        "price": 8500000,
        "property_type": "apartment"
    }
    prop_id = memory.save_property(user_id, property_data, notes="Great location near station", tags=["favorite", "urgent"])
    print(f"💾 Saved property: {prop_id}")
    
    # Get saved properties
    saved = memory.get_saved_properties(user_id)
    print(f"📋 Saved properties: {len(saved)} properties")
    
    print(f"\n4. 🗣️ SESSION CONTEXT MANAGEMENT")
    print("-" * 30)
    
    # Start session
    session_id = memory.start_session(user_id)
    print(f"🚀 Started session: {session_id}")
    
    # Add conversation
    memory.add_conversation_turn(session_id, "Find apartments in Pune", "Found 20 apartments in Pune matching your preferences")
    memory.update_session_context(session_id, current_intent="property_search", active_filters={"location": "Pune"})
    
    print("💬 Added conversation turn and updated context")
    
    print(f"\n5. 🎯 COMPREHENSIVE USER CONTEXT")
    print("-" * 30)
    
    # Get complete context
    context = memory.get_user_context_for_query(user_id, "Show me similar properties")
    print(f"📊 User context includes:")
    print(f"   - Preferences: {bool(context['preferences'])}")
    print(f"   - Recent searches: {len(context['recent_searches'])}")
    print(f"   - Saved properties: {len(context['saved_properties'])}")
    print(f"   - Interaction patterns: {bool(context['interaction_patterns'])}")
    
    print(f"\n6. 📈 MEMORY STATISTICS")
    print("-" * 30)
    
    stats = memory.get_memory_statistics()
    print(f"👥 Total users: {stats['total_users']}")
    print(f"🔄 Active sessions: {stats['active_sessions']}")
    print(f"📝 Search history entries: {stats['search_history_count']}")
    print(f"💾 Saved properties: {stats['saved_properties_count']}")
    
    print(f"\n🎉 MEMORY COMPONENT DEMONSTRATION COMPLETE!")
    print("✅ All memory types working across sessions")