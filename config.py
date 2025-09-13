"""Configuration settings for ClinicalTrials.gov data ingestion."""

from pathlib import Path

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

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Output formats
OUTPUT_FORMATS = ["csv", "json", "parquet"]

# Retry configuration for API requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds to wait before retrying

# Request timeout
REQUEST_TIMEOUT = 30  # Seconds