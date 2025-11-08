"""
Configuration for Rate Limiting in Real Estate Search Engine
Adjust these settings based on your API limits and requirements
"""

# Groq API Rate Limiting Configuration
GROQ_RATE_LIMITS = {
    # Conservative settings for free tier
    "free_tier": {
        "max_requests_per_minute": 20,  # Very conservative
        "max_requests_per_hour": 600,   # Conservative hourly limit
        "initial_delay": 3.0,           # Start with 3 second delay
        "max_delay": 60.0,              # Max 1 minute delay
        "backoff_factor": 1.8           # Moderate backoff
    },
    
    # Standard settings for paid tier
    "paid_tier": {
        "max_requests_per_minute": 50,
        "max_requests_per_hour": 2000,
        "initial_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 1.5
    },
    
    # Aggressive settings for high-volume usage
    "enterprise": {
        "max_requests_per_minute": 100,
        "max_requests_per_hour": 5000,
        "initial_delay": 0.5,
        "max_delay": 15.0,
        "backoff_factor": 1.3
    }
}

# Current rate limiting tier (change as needed)
CURRENT_TIER = "free_tier"

# Demo configuration
DEMO_CONFIG = {
    "delay_between_queries": 4.0,      # Seconds to wait between queries
    "max_demo_queries": 5,             # Maximum queries in demo
    "enable_verbose_logging": True,    # Show detailed rate limit info
    "fallback_on_rate_limit": True     # Use fallback responses on rate limits
}

# Monitoring configuration
MONITORING_CONFIG = {
    "log_rate_limit_stats": True,      # Log rate limiter statistics
    "track_api_usage": True,           # Track API usage patterns
    "alert_on_rate_limit": True,       # Alert when rate limits are hit
    "save_usage_metrics": True         # Save usage metrics to file
}

def get_rate_limit_config(tier: str = None) -> dict:
    """Get rate limiting configuration for specified tier"""
    if tier is None:
        tier = CURRENT_TIER
    
    return GROQ_RATE_LIMITS.get(tier, GROQ_RATE_LIMITS["free_tier"])

def is_conservative_mode() -> bool:
    """Check if we're running in conservative rate limiting mode"""
    return CURRENT_TIER == "free_tier"

def get_demo_delay() -> float:
    """Get recommended delay between demo queries"""
    if is_conservative_mode():
        return DEMO_CONFIG["delay_between_queries"] * 1.5  # Extra conservative for free tier
    else:
        return DEMO_CONFIG["delay_between_queries"]

# API Usage Guidelines
USAGE_GUIDELINES = {
    "free_tier": {
        "description": "Conservative usage for free API tiers",
        "recommended_concurrent_users": 1,
        "recommended_demo_duration": "5-10 minutes",
        "notes": "Use longer delays between requests to avoid rate limits"
    },
    
    "paid_tier": {
        "description": "Standard usage for paid API subscriptions",
        "recommended_concurrent_users": "3-5",
        "recommended_demo_duration": "3-5 minutes",
        "notes": "Balanced performance and API usage"
    },
    
    "enterprise": {
        "description": "High-volume usage for enterprise applications",
        "recommended_concurrent_users": "10+",
        "recommended_demo_duration": "1-3 minutes",
        "notes": "Optimized for performance with high API limits"
    }
}