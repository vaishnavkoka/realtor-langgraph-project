"""
Rate Limiter Utility for API Requests
Handles rate limiting with exponential backoff and request queuing
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter with exponential backoff and request queuing
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 30,
        max_requests_per_hour: int = 1000,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        self.requests_this_minute = []
        self.requests_this_hour = []
        self.current_delay = initial_delay
        self.last_request_time = 0
        
    def _clean_old_requests(self):
        """Remove requests older than the time window"""
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        self.requests_this_minute = [
            req_time for req_time in self.requests_this_minute 
            if req_time > one_minute_ago
        ]
        
        self.requests_this_hour = [
            req_time for req_time in self.requests_this_hour 
            if req_time > one_hour_ago
        ]
    
    def _should_wait(self) -> tuple[bool, float]:
        """Check if we should wait and return wait time"""
        self._clean_old_requests()
        
        # Check per-minute limit
        if len(self.requests_this_minute) >= self.max_requests_per_minute:
            oldest_minute = min(self.requests_this_minute)
            wait_time = 60 - (time.time() - oldest_minute) + 1
            return True, wait_time
        
        # Check per-hour limit
        if len(self.requests_this_hour) >= self.max_requests_per_hour:
            oldest_hour = min(self.requests_this_hour)
            wait_time = 3600 - (time.time() - oldest_hour) + 1
            return True, wait_time
        
        # Check minimum delay between requests
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.current_delay:
            wait_time = self.current_delay - time_since_last
            return True, wait_time
        
        return False, 0
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        should_wait, wait_time = self._should_wait()
        
        if should_wait:
            logger.info(f"🕐 Rate limit protection: Waiting {wait_time:.1f}s before next request")
            time.sleep(wait_time)
    
    def record_request(self, success: bool = True):
        """Record a request and update delay"""
        now = time.time()
        self.requests_this_minute.append(now)
        self.requests_this_hour.append(now)
        self.last_request_time = now
        
        if success:
            # Reset delay on successful request
            self.current_delay = max(self.initial_delay, self.current_delay * 0.8)
        else:
            # Increase delay on failed request
            self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)
            logger.warning(f"🚫 Request failed, increasing delay to {self.current_delay:.1f}s")
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
        self._clean_old_requests()
        return {
            "requests_last_minute": len(self.requests_this_minute),
            "requests_last_hour": len(self.requests_this_hour),
            "current_delay": self.current_delay,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour
        }

def rate_limited(rate_limiter: RateLimiter):
    """Decorator to apply rate limiting to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            rate_limiter.wait_if_needed()
            
            try:
                result = func(*args, **kwargs)
                rate_limiter.record_request(success=True)
                return result
            except Exception as e:
                # Check if it's a rate limit error
                if "429" in str(e) or "Too Many Requests" in str(e):
                    rate_limiter.record_request(success=False)
                    logger.warning(f"⚠️ Rate limit hit: {e}")
                    # Wait and retry once
                    rate_limiter.wait_if_needed()
                    try:
                        result = func(*args, **kwargs)
                        rate_limiter.record_request(success=True)
                        return result
                    except Exception as retry_e:
                        rate_limiter.record_request(success=False)
                        raise retry_e
                else:
                    raise e
        
        return wrapper
    return decorator

class GroqRateLimiter(RateLimiter):
    """Specialized rate limiter for Groq API"""
    
    def __init__(self):
        # Groq free tier limits: ~30 requests/minute, conservative settings
        super().__init__(
            max_requests_per_minute=25,  # Conservative limit
            max_requests_per_hour=800,   # Conservative hourly limit
            initial_delay=2.0,           # Start with 2 second delay
            max_delay=30.0,              # Max 30 second delay
            backoff_factor=1.5           # Moderate backoff
        )

# Global rate limiter instance
groq_rate_limiter = GroqRateLimiter()

def log_rate_limit_stats():
    """Log current rate limiter statistics"""
    stats = groq_rate_limiter.get_stats()
    logger.info(f"📊 Rate Limiter Stats: {stats['requests_last_minute']}/{stats['max_requests_per_minute']} req/min, "
                f"{stats['requests_last_hour']}/{stats['max_requests_per_hour']} req/hour, "
                f"delay: {stats['current_delay']:.1f}s")