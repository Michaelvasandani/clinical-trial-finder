"""Configuration for embedding generation and vector storage."""

from pathlib import Path

# Model Configuration
MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBEDDING_DIM = 768  # BioBERT output dimension
MAX_TOKENS = 256  # BioBERT max input length
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

# File Paths
BASE_DIR = Path(__file__).parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Input/Output
INPUT_CSV = BASE_DIR / "data" / "processed" / "clinical_trials_20250912_175505.csv"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "clinical_trials.index"
METADATA_PATH = EMBEDDINGS_DIR / "clinical_trials_metadata.json"
CHUNK_MAPPING_PATH = EMBEDDINGS_DIR / "chunk_mapping.json"

# Processing Configuration
USE_GPU = True  # Will fallback to CPU if GPU not available
SHOW_PROGRESS = True
SAVE_INTERMEDIATE = True  # Save embeddings after each batch

# Search Configuration
DEFAULT_TOP_K = 10  # Default number of results to return
MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity score to include in results