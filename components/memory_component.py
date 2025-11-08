"""
Memory Component for Multi-Agent Real Estate Search Engine
Provides conversation memory, user preferences, context retention, and personalized recommendations

Features:
1. Conversation History Management
2. User Preference Learning
3. Search Context Retention
4. Personalized Property Recommendations
5. User Interaction Analytics
6. Long-term Memory Storage
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    """User preference data structure"""
    user_id: str
    preference_type: str  # location, price_range, property_type, amenities, etc.
    preference_value: Any
    confidence_score: float
    created_at: datetime
    updated_at: datetime
    frequency: int = 1

@dataclass
class ConversationTurn:
    """Individual conversation turn"""
    turn_id: str
    user_id: str
    timestamp: datetime
    user_query: str
    user_intent: str
    extracted_entities: Dict[str, Any]
    agent_response: str
    agents_used: List[str]
    properties_shown: List[str]
    user_feedback: Optional[str] = None
    satisfaction_score: Optional[float] = None

@dataclass
class SearchContext:
    """Search session context"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    search_criteria: Dict[str, Any]
    viewed_properties: List[str]
    saved_properties: List[str]
    search_refinements: List[Dict[str, Any]]
    is_active: bool = True

@dataclass
class UserProfile:
    """Comprehensive user profile"""
    user_id: str
    created_at: datetime
    updated_at: datetime
    total_searches: int
    total_views: int
    preferred_locations: List[str]
    budget_range: Tuple[int, int]
    preferred_property_types: List[str]
    lifestyle_preferences: Dict[str, Any]
    investment_profile: str  # conservative, moderate, aggressive
    communication_style: str  # brief, detailed, technical


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_turns_per_session: int = 50):
        self.max_turns_per_session = max_turns_per_session
        self.active_sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns_per_session))
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to memory"""
        session_key = f"{turn.user_id}_{turn.timestamp.date()}"
        self.active_sessions[session_key].append(turn)
    
    def get_recent_context(self, user_id: str, num_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context for a user"""
        session_key = f"{user_id}_{datetime.now().date()}"
        turns = list(self.active_sessions[session_key])
        return turns[-num_turns:] if turns else []
    
    def get_session_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of current session"""
        session_key = f"{user_id}_{datetime.now().date()}"
        turns = list(self.active_sessions[session_key])
        
        if not turns:
            return {"total_turns": 0, "search_intents": [], "properties_viewed": []}
        
        return {
            "total_turns": len(turns),
            "search_intents": [turn.user_intent for turn in turns],
            "properties_viewed": sum([turn.properties_shown for turn in turns], []),
            "agents_used": list(set(sum([turn.agents_used for turn in turns], []))),
            "session_duration": (turns[-1].timestamp - turns[0].timestamp).total_seconds() / 60
        }


class PreferenceEngine:
    """Learns and manages user preferences"""
    
    def __init__(self):
        self.user_preferences: Dict[str, List[UserPreference]] = defaultdict(list)
    
    def extract_preferences_from_query(self, user_id: str, query: str, entities: Dict[str, Any]) -> List[UserPreference]:
        """Extract implicit preferences from user query"""
        preferences = []
        timestamp = datetime.now()
        
        # Location preferences
        if entities.get("location"):
            pref = UserPreference(
                user_id=user_id,
                preference_type="location",
                preference_value=entities["location"],
                confidence_score=0.8,
                created_at=timestamp,
                updated_at=timestamp
            )
            preferences.append(pref)
        
        # Price range preferences
        if entities.get("max_price"):
            pref = UserPreference(
                user_id=user_id,
                preference_type="max_budget",
                preference_value=entities["max_price"],
                confidence_score=0.9,
                created_at=timestamp,
                updated_at=timestamp
            )
            preferences.append(pref)
        
        # Property type preferences
        if entities.get("property_type"):
            pref = UserPreference(
                user_id=user_id,
                preference_type="property_type",
                preference_value=entities["property_type"],
                confidence_score=0.7,
                created_at=timestamp,
                updated_at=timestamp
            )
            preferences.append(pref)
        
        # Room preferences
        if entities.get("rooms"):
            pref = UserPreference(
                user_id=user_id,
                preference_type="rooms",
                preference_value=entities["rooms"],
                confidence_score=0.8,
                created_at=timestamp,
                updated_at=timestamp
            )
            preferences.append(pref)
        
        # Investment-related preferences
        investment_keywords = ["investment", "yield", "rental", "portfolio", "roi"]
        if any(keyword in query.lower() for keyword in investment_keywords):
            pref = UserPreference(
                user_id=user_id,
                preference_type="investment_focus",
                preference_value=True,
                confidence_score=0.6,
                created_at=timestamp,
                updated_at=timestamp
            )
            preferences.append(pref)
        
        return preferences
    
    def update_preferences(self, user_id: str, preferences: List[UserPreference]):
        """Update user preferences with reinforcement learning"""
        for new_pref in preferences:
            existing_prefs = self.user_preferences[user_id]
            
            # Find existing preference of same type
            existing_pref = None
            for i, pref in enumerate(existing_prefs):
                if pref.preference_type == new_pref.preference_type and pref.preference_value == new_pref.preference_value:
                    existing_pref = pref
                    break
            
            if existing_pref:
                # Reinforce existing preference
                existing_pref.frequency += 1
                existing_pref.confidence_score = min(0.95, existing_pref.confidence_score + 0.1)
                existing_pref.updated_at = datetime.now()
            else:
                # Add new preference
                self.user_preferences[user_id].append(new_pref)
    
    def get_user_preferences(self, user_id: str, preference_type: Optional[str] = None) -> List[UserPreference]:
        """Get user preferences, optionally filtered by type"""
        preferences = self.user_preferences.get(user_id, [])
        
        if preference_type:
            preferences = [p for p in preferences if p.preference_type == preference_type]
        
        # Sort by confidence score and frequency
        return sorted(preferences, key=lambda x: (x.confidence_score, x.frequency), reverse=True)


class MemoryStore:
    """Persistent storage for memory data"""
    
    def __init__(self, db_path: str = "memory_store.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for memory storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    turn_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TIMESTAMP,
                    turn_data TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    preference_data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            # Search contexts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_contexts (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    context_data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            conn.commit()
    
    def save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, profile_data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                profile.user_id,
                json.dumps(asdict(profile), default=str),
                profile.created_at,
                profile.updated_at
            ))
            conn.commit()
    
    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT profile_data FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            if result:
                profile_data = json.loads(result[0])
                # Convert string timestamps back to datetime objects
                profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                profile_data['updated_at'] = datetime.fromisoformat(profile_data['updated_at'])
                return UserProfile(**profile_data)
            return None
    
    def save_conversation_turn(self, turn: ConversationTurn):
        """Save conversation turn to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_history 
                (turn_id, user_id, timestamp, turn_data)
                VALUES (?, ?, ?, ?)
            """, (
                turn.turn_id,
                turn.user_id,
                turn.timestamp,
                json.dumps(asdict(turn), default=str)
            ))
            conn.commit()
    
    def load_conversation_history(self, user_id: str, days: int = 7) -> List[ConversationTurn]:
        """Load recent conversation history"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT turn_data FROM conversation_history 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (user_id, since_date))
            
            turns = []
            for (turn_data,) in cursor.fetchall():
                data = json.loads(turn_data)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                turns.append(ConversationTurn(**data))
            
            return turns


class MemoryComponent:
    """Main memory component integrating all memory functionalities"""
    
    def __init__(self, db_path: str = "real_estate_memory.db"):
        self.conversation_memory = ConversationMemory()
        self.preference_engine = PreferenceEngine()
        self.memory_store = MemoryStore(db_path)
        self.active_users: Dict[str, datetime] = {}
        
        logger.info("🧠 Memory Component initialized successfully!")
    
    def create_user_session(self, user_id: str) -> str:
        """Create a new user session"""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_users[user_id] = datetime.now()
        
        # Load existing user profile or create new one
        profile = self.memory_store.load_user_profile(user_id)
        if not profile:
            profile = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                total_searches=0,
                total_views=0,
                preferred_locations=[],
                budget_range=(0, 10000000),
                preferred_property_types=[],
                lifestyle_preferences={},
                investment_profile="moderate",
                communication_style="detailed"
            )
            self.memory_store.save_user_profile(profile)
        
        logger.info(f"📝 Created session {session_id} for user {user_id}")
        return session_id
    
    def process_user_interaction(
        self, 
        user_id: str,
        query: str,
        entities: Dict[str, Any],
        intent: str,
        agents_used: List[str],
        response: str,
        properties_shown: List[str]
    ) -> Dict[str, Any]:
        """Process a user interaction and update memory"""
        
        # Create conversation turn
        turn_id = f"{user_id}_{datetime.now().timestamp()}"
        turn = ConversationTurn(
            turn_id=turn_id,
            user_id=user_id,
            timestamp=datetime.now(),
            user_query=query,
            user_intent=intent,
            extracted_entities=entities,
            agent_response=response,
            agents_used=agents_used,
            properties_shown=properties_shown
        )
        
        # Add to conversation memory
        self.conversation_memory.add_turn(turn)
        
        # Save to persistent storage
        self.memory_store.save_conversation_turn(turn)
        
        # Extract and update preferences
        preferences = self.preference_engine.extract_preferences_from_query(user_id, query, entities)
        self.preference_engine.update_preferences(user_id, preferences)
        
        # Update user profile
        profile = self.memory_store.load_user_profile(user_id)
        if profile:
            profile.total_searches += 1
            profile.total_views += len(properties_shown)
            profile.updated_at = datetime.now()
            
            # Update preferred locations
            if entities.get("location") and entities["location"] not in profile.preferred_locations:
                profile.preferred_locations.append(entities["location"])
                if len(profile.preferred_locations) > 10:  # Keep only top 10
                    profile.preferred_locations = profile.preferred_locations[-10:]
            
            self.memory_store.save_user_profile(profile)
        
        # Return memory context for current interaction
        return self.get_memory_context(user_id)
    
    def get_memory_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory context for a user"""
        
        # Recent conversation context
        recent_turns = self.conversation_memory.get_recent_context(user_id, num_turns=3)
        session_summary = self.conversation_memory.get_session_summary(user_id)
        
        # User preferences
        location_prefs = self.preference_engine.get_user_preferences(user_id, "location")
        budget_prefs = self.preference_engine.get_user_preferences(user_id, "max_budget")
        type_prefs = self.preference_engine.get_user_preferences(user_id, "property_type")
        
        # User profile
        profile = self.memory_store.load_user_profile(user_id)
        
        context = {
            "user_id": user_id,
            "session_summary": session_summary,
            "recent_queries": [turn.user_query for turn in recent_turns],
            "recent_intents": [turn.user_intent for turn in recent_turns],
            "preferred_locations": [p.preference_value for p in location_prefs[:3]],
            "budget_preferences": [p.preference_value for p in budget_prefs[:2]],
            "property_type_preferences": [p.preference_value for p in type_prefs[:3]],
            "user_profile": {
                "total_searches": profile.total_searches if profile else 0,
                "investment_profile": profile.investment_profile if profile else "moderate",
                "communication_style": profile.communication_style if profile else "detailed",
                "preferred_locations": profile.preferred_locations if profile else []
            } if profile else None,
            "context_timestamp": datetime.now().isoformat()
        }
        
        return context
    
    def get_personalized_recommendations(self, user_id: str, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Provide personalized property recommendations based on memory"""
        
        if not properties:
            return []
        
        preferences = self.preference_engine.get_user_preferences(user_id)
        profile = self.memory_store.load_user_profile(user_id)
        
        # Score properties based on user preferences
        scored_properties = []
        
        for prop in properties:
            score = 0.0
            reasons = []
            
            # Location matching
            location_prefs = [p for p in preferences if p.preference_type == "location"]
            for loc_pref in location_prefs:
                if loc_pref.preference_value.lower() in prop.get("location", "").lower():
                    score += loc_pref.confidence_score * 0.3
                    reasons.append(f"Matches preferred location: {loc_pref.preference_value}")
            
            # Budget matching
            budget_prefs = [p for p in preferences if p.preference_type == "max_budget"]
            for budget_pref in budget_prefs:
                if prop.get("price", 0) <= budget_pref.preference_value:
                    score += budget_pref.confidence_score * 0.25
                    reasons.append(f"Within budget preference: ₹{budget_pref.preference_value:,}")
            
            # Property type matching
            type_prefs = [p for p in preferences if p.preference_type == "property_type"]
            for type_pref in type_prefs:
                if type_pref.preference_value.lower() in prop.get("title", "").lower():
                    score += type_pref.confidence_score * 0.2
                    reasons.append(f"Matches preferred type: {type_pref.preference_value}")
            
            # Investment focus
            investment_prefs = [p for p in preferences if p.preference_type == "investment_focus"]
            if investment_prefs and profile and profile.investment_profile in ["moderate", "aggressive"]:
                score += 0.15
                reasons.append("Aligns with investment focus")
            
            prop_with_score = prop.copy()
            prop_with_score["personalization_score"] = score
            prop_with_score["recommendation_reasons"] = reasons
            scored_properties.append(prop_with_score)
        
        # Sort by personalization score
        scored_properties.sort(key=lambda x: x["personalization_score"], reverse=True)
        
        return scored_properties
    
    def update_user_feedback(self, user_id: str, turn_id: str, feedback: str, satisfaction_score: float):
        """Update user feedback for learning"""
        # This would be used to improve future recommendations
        # For now, we'll store it for analytics
        logger.info(f"📝 Received feedback from {user_id}: {feedback} (satisfaction: {satisfaction_score})")
    
    def get_memory_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics about user memory and behavior"""
        
        profile = self.memory_store.load_user_profile(user_id)
        preferences = self.preference_engine.get_user_preferences(user_id)
        session_summary = self.conversation_memory.get_session_summary(user_id)
        
        return {
            "user_activity": {
                "total_searches": profile.total_searches if profile else 0,
                "total_views": profile.total_views if profile else 0,
                "session_turns": session_summary.get("total_turns", 0)
            },
            "learned_preferences": {
                "total_preferences": len(preferences),
                "location_preferences": len([p for p in preferences if p.preference_type == "location"]),
                "budget_preferences": len([p for p in preferences if p.preference_type == "max_budget"]),
                "type_preferences": len([p for p in preferences if p.preference_type == "property_type"])
            },
            "memory_strength": {
                "conversation_memory": "Active" if session_summary.get("total_turns", 0) > 0 else "Empty",
                "preference_confidence": sum(p.confidence_score for p in preferences) / len(preferences) if preferences else 0,
                "profile_completeness": 0.8 if profile and profile.preferred_locations else 0.3
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize memory component
    memory = MemoryComponent()
    
    # Simulate user interactions
    user_id = "test_user_123"
    session_id = memory.create_user_session(user_id)
    
    # Simulate several search interactions
    interactions = [
        {
            "query": "Find 3 BHK apartments in Mumbai under 2 crores",
            "entities": {"location": "Mumbai", "rooms": 3, "max_price": 20000000, "property_type": "apartment"},
            "intent": "search",
            "agents_used": ["StructuredDataAgent", "RAGAgent"],
            "response": "Found 5 apartments matching your criteria...",
            "properties_shown": ["PROP-001", "PROP-002", "PROP-003"]
        },
        {
            "query": "Show me investment properties in Mumbai with good rental yield",
            "entities": {"location": "Mumbai", "property_type": "investment"},
            "intent": "investment_search",
            "agents_used": ["StructuredDataAgent", "RAGAgent", "WebResearchAgent"],
            "response": "Here are some investment opportunities...",
            "properties_shown": ["PROP-004", "PROP-005"]
        }
    ]
    
    for interaction in interactions:
        context = memory.process_user_interaction(
            user_id=user_id,
            **interaction
        )
        print(f"Memory Context: {json.dumps(context, indent=2, default=str)}")
    
    # Get analytics
    analytics = memory.get_memory_analytics(user_id)
    print(f"Memory Analytics: {json.dumps(analytics, indent=2)}")