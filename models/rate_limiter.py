"""
Rate limiter for free APIs to prevent exceeding limits
"""

import time
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FreeAPIRateLimiter:
    """Smart rate limiter for free API services"""
    
    def __init__(self, cache_file: str = "data/api_usage.json"):
        self.cache_file = cache_file
        self.usage_data = self._load_usage_data()
        
        # API Limits (conservative estimates)
        self.limits = {
            "groq": {
                "daily": 25000,     # requests per day
                "minute": 6000,     # requests per minute
                "concurrent": 10    # concurrent requests
            },
            "serper": {
                "monthly": 2400,    # searches per month
                "daily": 80,        # searches per day
                "minute": 10        # searches per minute
            },
            "tavily": {
                "monthly": 950,     # searches per month
                "daily": 32,        # searches per day
                "minute": 5         # searches per minute
            },
            "huggingface": {
                "monthly": 950,     # API calls per month
                "daily": 32,        # API calls per day
                "minute": 10        # API calls per minute
            }
        }
    
    def _load_usage_data(self) -> Dict:
        """Load usage data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load usage data: {e}")
        
        return defaultdict(lambda: defaultdict(int))
    
    def _save_usage_data(self):
        """Save usage data to cache file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.usage_data, f)
        except Exception as e:
            logger.error(f"Could not save usage data: {e}")
    
    def _get_time_keys(self):
        """Get time-based keys for tracking usage"""
        now = datetime.now()
        return {
            "year_month": now.strftime("%Y-%m"),
            "date": now.strftime("%Y-%m-%d"),
            "hour_minute": now.strftime("%Y-%m-%d_%H:%M")
        }
    
    def can_make_request(self, service: str) -> bool:
        """Check if we can make a request to the service"""
        if service not in self.limits:
            return True
        
        time_keys = self._get_time_keys()
        
        # Initialize service if not exists
        if service not in self.usage_data:
            self.usage_data[service] = defaultdict(int)
        
        # Ensure the service data is a defaultdict
        if not isinstance(self.usage_data[service], defaultdict):
            self.usage_data[service] = defaultdict(int, self.usage_data[service])
        
        current_usage = self.usage_data[service]
        limits = self.limits[service]
        
        # Check monthly limit
        if "monthly" in limits:
            monthly_usage = current_usage.get(time_keys["year_month"], 0)
            if monthly_usage >= limits["monthly"] * 0.9:  # 90% safety margin
                logger.warning(f"Monthly limit approaching for {service}: {monthly_usage}/{limits['monthly']}")
                return False
        
        # Check daily limit
        if "daily" in limits:
            daily_usage = current_usage.get(time_keys["date"], 0)
            if daily_usage >= limits["daily"] * 0.9:  # 90% safety margin
                logger.warning(f"Daily limit approaching for {service}: {daily_usage}/{limits['daily']}")
                return False
        
        # Check minute limit
        if "minute" in limits:
            minute_usage = current_usage.get(time_keys["hour_minute"], 0)
            if minute_usage >= limits["minute"]:
                logger.warning(f"Minute limit reached for {service}: {minute_usage}/{limits['minute']}")
                return False
        
        return True
    
    def record_request(self, service: str):
        """Record a successful request"""
        time_keys = self._get_time_keys()
        
        # Initialize service if not exists
        if service not in self.usage_data:
            self.usage_data[service] = defaultdict(int)
        
        # Ensure the service data is a defaultdict
        if not isinstance(self.usage_data[service], defaultdict):
            self.usage_data[service] = defaultdict(int, self.usage_data[service])
        
        # Increment counters
        self.usage_data[service][time_keys["year_month"]] += 1
        self.usage_data[service][time_keys["date"]] += 1
        self.usage_data[service][time_keys["hour_minute"]] += 1
        
        # Save periodically
        self._save_usage_data()
        
        logger.debug(f"Recorded request for {service}")
    
    def get_best_available_service(self, service_type: str) -> Optional[str]:
        """Get the best available service for a given type"""
        if service_type == "llm":
            if self.can_make_request("groq"):
                return "groq"
            else:
                logger.warning("Groq limit reached, no backup LLM available")
                return None
        
        elif service_type == "search":
            if self.can_make_request("serper"):
                return "serper"
            elif self.can_make_request("tavily"):
                return "tavily"
            else:
                logger.warning("All search APIs limit reached")
                return None
        
        elif service_type == "embedding":
            # HuggingFace is local, no limits
            return "huggingface"
        
        return None
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics"""
        time_keys = self._get_time_keys()
        stats = {}
        
        for service, limits in self.limits.items():
            service_usage = self.usage_data.get(service, {})
            
            stats[service] = {
                "monthly": {
                    "used": service_usage.get(time_keys["year_month"], 0),
                    "limit": limits.get("monthly", "unlimited"),
                    "remaining": limits.get("monthly", float('inf')) - service_usage.get(time_keys["year_month"], 0)
                },
                "daily": {
                    "used": service_usage.get(time_keys["date"], 0),
                    "limit": limits.get("daily", "unlimited"),
                    "remaining": limits.get("daily", float('inf')) - service_usage.get(time_keys["date"], 0)
                }
            }
        
        return stats
    
    def wait_if_needed(self, service: str, max_wait: int = 60):
        """Wait if rate limit is hit (for minute-based limits)"""
        if not self.can_make_request(service):
            # If it's a minute limit, wait for next minute
            if "minute" in self.limits.get(service, {}):
                wait_time = min(max_wait, 60)
                logger.info(f"Rate limit hit for {service}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return True
        return False

# Global rate limiter instance
rate_limiter = FreeAPIRateLimiter()