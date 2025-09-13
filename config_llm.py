"""Configuration for LLM integration and conversation management."""

import os
from pathlib import Path
from typing import Dict, List

# Model Configuration
DEFAULT_MODEL = "gpt-4"  # Primary model for conversations
FALLBACK_MODEL = "gpt-3.5-turbo"  # Fallback for simpler tasks
EMBEDDING_MODEL = "text-embedding-ada-002"  # For query embeddings if needed

# API Configuration
MAX_TOKENS = 4000  # Maximum tokens per response
TEMPERATURE = 0.3  # Lower temperature for more consistent medical responses
TOP_P = 0.9
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

# Request Configuration
REQUEST_TIMEOUT = 60  # Seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Seconds between retries

# Conversation Configuration
MAX_CONVERSATION_HISTORY = 10  # Number of previous exchanges to remember
MAX_CONTEXT_LENGTH = 8000  # Maximum tokens for conversation context
CONVERSATION_TIMEOUT = 3600  # Seconds (1 hour) before conversation expires

# Safety and Content Configuration
MEDICAL_DISCLAIMER = """
⚠️ **Important Medical Disclaimer**: This information is for educational purposes only and should not be considered medical advice. Always consult with qualified healthcare professionals before making any medical decisions. Clinical trials have risks and benefits that must be carefully evaluated with your doctor.
"""

# Response Configuration
MAX_RESPONSE_LENGTH = 2000  # Maximum characters in response
INCLUDE_SOURCES = True  # Whether to include trial sources in responses
SHOW_CONFIDENCE = False  # Whether to show confidence levels

# Rate Limiting
REQUESTS_PER_MINUTE = 50
REQUESTS_PER_HOUR = 1000

# Cost Management
MAX_DAILY_COST = 50.0  # USD - daily spending limit
TRACK_USAGE = True  # Monitor API usage and costs

# File Paths
BASE_DIR = Path(__file__).parent
CONVERSATIONS_DIR = BASE_DIR / "conversations"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [CONVERSATIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Conversation Types and Their Configurations
CONVERSATION_TYPES = {
    "general_inquiry": {
        "system_prompt_key": "general_medical",
        "max_tokens": 3000,
        "temperature": 0.3,
        "include_search": True
    },
    "trial_explanation": {
        "system_prompt_key": "trial_explainer",
        "max_tokens": 4000,
        "temperature": 0.2,
        "include_search": False
    },
    "eligibility_assessment": {
        "system_prompt_key": "eligibility_helper",
        "max_tokens": 3500,
        "temperature": 0.2,
        "include_search": True
    },
    "search_assistance": {
        "system_prompt_key": "search_helper",
        "max_tokens": 2500,
        "temperature": 0.4,
        "include_search": True
    }
}

# Medical Content Guidelines
PROHIBITED_TOPICS = [
    "specific medical diagnosis",
    "treatment recommendations",
    "medication dosage advice",
    "emergency medical situations",
    "self-diagnosis guidance"
]

ENCOURAGED_TOPICS = [
    "general trial information",
    "understanding eligibility criteria", 
    "explaining medical terms",
    "research process education",
    "connecting to healthcare providers"
]

# Response Quality Settings
REQUIRE_SOURCES = True  # Always include sources for medical claims
FACT_CHECK_MEDICAL_CLAIMS = True  # Extra validation for medical information
INCLUDE_UNCERTAINTIES = True  # Acknowledge when information is unclear

# Integration Settings
SEARCH_INTEGRATION = True  # Enable integration with clinical trial search
MAX_SEARCH_RESULTS = 5  # Maximum trials to consider in responses
SEARCH_RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score

# Logging Configuration
LOG_CONVERSATIONS = True  # Save conversation logs (anonymized)
LOG_SEARCH_QUERIES = True  # Log search patterns
LOG_LEVEL = "INFO"

# Environment Variables (loaded from .env)
def get_env_config() -> Dict[str, str]:
    """Get configuration from environment variables."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_ORG_ID": os.getenv("OPENAI_ORG_ID"),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true"
    }

# Model Costs (USD per 1K tokens) - Update as needed
MODEL_COSTS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0}
}