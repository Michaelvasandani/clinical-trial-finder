"""Unified configuration settings for Clinical Trial Finder system."""

import os
from pathlib import Path
from typing import Dict, List

# =============================================================================
# DATA INGESTION CONFIGURATION
# =============================================================================

# API Configuration
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 100  # Number of results per page (max 1000)
RATE_LIMIT_DELAY = 1.5  # Seconds between requests to respect rate limits

# Target conditions for data collection
CONDITIONS = [
    "cancer",
    "diabetes", 
    "heart disease",
    "hypertension",
    "alzheimer disease"
]

# Data collection limits
MAX_TRIALS_PER_CONDITION = 400
TOTAL_TARGET_TRIALS = 1500

# Fields to extract from API response
REQUIRED_FIELDS = [
    "NCTId",
    "BriefTitle", 
    "OfficialTitle",
    "OverallStatus",
    "Phase",
    "StudyType",
    "Condition",
    "InterventionName",
    "InterventionType",
    "InterventionDescription",
    "BriefSummary",
    "DetailedDescription",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge", 
    "Gender",
    "HealthyVolunteers",
    "LocationCity",
    "LocationState",
    "LocationCountry",
    "LocationFacility",
    "LocationStatus",
    "StartDate",
    "PrimaryCompletionDate",
    "CompletionDate",
    "StudyFirstPostDate",
    "LastUpdatePostDate",
    "EnrollmentCount",
    "EnrollmentType",
    "PrimaryOutcomeMeasure",
    "PrimaryOutcomeDescription",
    "PrimaryOutcomeTimeFrame",
    "SecondaryOutcomeMeasure",
    "SecondaryOutcomeDescription",
    "SecondaryOutcomeTimeFrame",
    "ResponsiblePartyType",
    "LeadSponsorName",
    "LeadSponsorClass",
    "CollaboratorName",
    "CollaboratorClass"
]

# API query fields parameter (formatted for API request)
API_FIELDS = "|".join(REQUIRED_FIELDS)

# Output formats
OUTPUT_FORMATS = ["csv", "json", "parquet"]

# Retry configuration for API requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds to wait before retrying
REQUEST_TIMEOUT = 30  # Seconds

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

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
LLM_REQUEST_TIMEOUT = 60  # Seconds
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.0  # Seconds between retries

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

# Model Costs (USD per 1K tokens) - Update as needed
MODEL_COSTS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0}
}

# =============================================================================
# EMBEDDINGS CONFIGURATION
# =============================================================================

# Model Configuration
EMBEDDING_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBEDDING_DIM = 768  # BioBERT output dimension
MAX_TOKENS_EMBED = 256  # BioBERT max input length
CHUNK_OVERLAP = 50  # Token overlap between chunks
BATCH_SIZE = 32  # Number of documents to process at once

# Document Processing
FIELDS_TO_EMBED = {
    "title": ["BriefTitle", "OfficialTitle"],
    "summary": ["BriefSummary", "DetailedDescription"],
    "eligibility": ["EligibilityCriteria"],
    "medical": ["Condition", "InterventionName", "InterventionType", "Phase", "StudyType"],
    "outcomes": ["PrimaryOutcomeMeasure", "SecondaryOutcomeMeasure"]
}

# Metadata fields to store but not embed
METADATA_FIELDS = [
    "NCTId", "BriefTitle", "OverallStatus", "Phase", "StudyType", 
    "Condition", "MinimumAge", "MaximumAge", "Gender", "HealthyVolunteers",
    "LocationCity", "LocationState", "LocationCountry", "LocationFacility", "LocationStatus",
    "StartDate", "PrimaryCompletionDate", "CompletionDate", 
    "EnrollmentCount", "EnrollmentType", "LeadSponsorName", "LeadSponsorClass",
    "InterventionName", "InterventionType", "SearchCondition"
]

# Vector Store Configuration
VECTOR_STORE_TYPE = "faiss"  # Options: "faiss" or "chromadb"
SIMILARITY_METRIC = "cosine"  # Options: "cosine", "euclidean", "inner_product"

# Processing Configuration
USE_GPU = True  # Will fallback to CPU if GPU not available
SHOW_PROGRESS = True
SAVE_INTERMEDIATE = True  # Save embeddings after each batch

# Search Configuration
DEFAULT_TOP_K = 10  # Default number of results to return
MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity score to include in results

# =============================================================================
# PATIENT EXTRACTION CONFIGURATION
# =============================================================================

# Extraction settings
EXTRACTION_TEMPERATURE = 0.1  # Low temperature for consistent extraction
EXTRACTION_MAX_TOKENS = 1000  # Max tokens for extraction response

# Common medical conditions for recognition
COMMON_CONDITIONS = [
    # Cancers
    "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer",
    "melanoma", "lymphoma", "leukemia", "pancreatic cancer",
    
    # Chronic conditions
    "diabetes", "type 1 diabetes", "type 2 diabetes",
    "hypertension", "high blood pressure",
    "heart disease", "coronary artery disease", "heart failure",
    "COPD", "chronic obstructive pulmonary disease", "asthma",
    "kidney disease", "chronic kidney disease",
    "liver disease", "cirrhosis", "hepatitis",
    
    # Neurological
    "alzheimer's disease", "dementia", "parkinson's disease",
    "multiple sclerosis", "epilepsy", "migraine",
    
    # Mental health
    "depression", "anxiety", "bipolar disorder", "schizophrenia", "PTSD",
    
    # Autoimmune
    "rheumatoid arthritis", "lupus", "psoriasis", "crohn's disease",
    "ulcerative colitis", "multiple sclerosis",
    
    # Other common conditions
    "HIV", "AIDS", "obesity", "osteoporosis", "fibromyalgia"
]

# Common medications (for recognition)
COMMON_MEDICATIONS = [
    # Diabetes medications
    "metformin", "insulin", "glipizide", "januvia", "ozempic",
    
    # Heart/Blood pressure
    "lisinopril", "metoprolol", "amlodipine", "atorvastatin", "aspirin",
    "warfarin", "eliquis", "plavix",
    
    # Pain/Inflammation
    "ibuprofen", "acetaminophen", "tylenol", "advil", "naproxen",
    
    # Mental health
    "sertraline", "zoloft", "lexapro", "prozac", "xanax", "ativan",
    
    # Cancer treatments
    "chemotherapy", "radiation", "immunotherapy", "herceptin", "keytruda",
    
    # Other common
    "levothyroxine", "omeprazole", "gabapentin", "prednisone"
]

# Location keywords for extraction
LOCATION_KEYWORDS = [
    "in", "from", "near", "located in", "living in", "based in",
    "resident of", "staying in"
]

# Age extraction patterns
AGE_PATTERNS = [
    r'(\d+)[\s-]*(?:year|yr)s?[\s-]*old',
    r'age[d]?[\s:]+(\d+)',
    r"i[']?m\s+(\d+)",
    r"patient is (\d+)",
    r"(\d+) y/?o"
]

# Gender extraction patterns
GENDER_PATTERNS = {
    "female": ["woman", "female", "girl", "lady", "she", "her"],
    "male": ["man", "male", "boy", "gentleman", "he", "his", "him"]
}

# Validation rules
VALID_AGE_RANGE = (0, 120)  # Valid age range for patients
MAX_CONDITIONS = 10  # Maximum number of conditions to extract
MAX_MEDICATIONS = 15  # Maximum number of medications to extract

# Search query optimization
MAX_QUERY_CONDITIONS = 3  # Max conditions to include in search query
MAX_QUERY_MEDICATIONS = 2  # Max medications to include in search query
MAX_QUERY_LENGTH = 200  # Maximum characters in search query

# Patient summary template
SUMMARY_TEMPLATE = """
Patient Profile Summary:
- Age: {age}
- Gender: {gender}
- Primary Conditions: {conditions}
- Current Medications: {medications}
- Location: {location}
- Additional Notes: {notes}
"""

# Privacy settings
ANONYMIZE_NAMES = True  # Remove personal names from extracted data
REMOVE_IDENTIFIERS = True  # Remove SSN, phone numbers, etc.

# Extraction prompts
GPT4_EXTRACTION_PROMPT = """You are a medical information extraction specialist. Extract patient information from the provided text and return it as structured JSON.

Focus on:
1. Demographics (age, gender)
2. Medical conditions (all mentioned)
3. Current medications
4. Location (if mentioned)
5. Relevant medical history
6. Symptoms
7. Allergies

Return ONLY valid JSON with these fields:
- age (number or null)
- gender (male/female/other or null)
- conditions (array of strings)
- medications (array of strings)
- location (string or null)
- symptoms (array of strings)
- allergies (array of strings)
- diagnosis_dates (array of strings)
- previous_treatments (array of strings)
- family_history (array of strings)

Use null for missing fields. Keep medical terminology as stated."""

# Matching configuration
MATCH_SCORE_WEIGHTS = {
    "condition_match": 0.4,
    "age_match": 0.2,
    "location_match": 0.2,
    "medication_relevance": 0.1,
    "phase_preference": 0.1
}

# Default number of trials to return
DEFAULT_NUM_MATCHES = 10

# Confidence thresholds
MIN_EXTRACTION_CONFIDENCE = 0.7  # Minimum confidence for extracted data
MIN_MATCH_SCORE = 0.5  # Minimum score for trial matches

# =============================================================================
# DIRECTORY PATHS AND SETUP
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"
CONVERSATIONS_DIR = BASE_DIR / "conversations"
LOGS_DIR = BASE_DIR / "logs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR, CONVERSATIONS_DIR, LOGS_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths for embeddings
INPUT_CSV = PROCESSED_DATA_DIR / "clinical_trials_20250912_175505.csv"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "clinical_trials.index"
METADATA_PATH = EMBEDDINGS_DIR / "clinical_trials_metadata.json"
CHUNK_MAPPING_PATH = EMBEDDINGS_DIR / "chunk_mapping.json"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_CONVERSATIONS = True  # Save conversation logs (anonymized)
LOG_SEARCH_QUERIES = True  # Log search patterns

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

def get_env_config() -> Dict[str, str]:
    """Get configuration from environment variables."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_ORG_ID": os.getenv("OPENAI_ORG_ID"),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", ""),
        "MAX_REQUESTS_PER_MINUTE": int(os.getenv("MAX_REQUESTS_PER_MINUTE", "50")),
        "MAX_REQUESTS_PER_HOUR": int(os.getenv("MAX_REQUESTS_PER_HOUR", "1000")),
        "MAX_DAILY_COST_USD": float(os.getenv("MAX_DAILY_COST_USD", "50.0")),
        "ALERT_COST_THRESHOLD_USD": float(os.getenv("ALERT_COST_THRESHOLD_USD", "40.0"))
    }