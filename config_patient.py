"""Configuration for patient information extraction."""

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