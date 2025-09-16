# Clinical Trial Finder 🔬

A conversational AI system that helps users find and understand clinical trials through intelligent search, patient profile matching, and natural language interaction.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Local Installation](#local-installation)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

The Clinical Trial Finder combines cutting-edge AI technologies to make clinical trial discovery accessible and intelligent. It processes over 26,000 clinical trials from ClinicalTrials.gov, uses BioBERT embeddings for semantic search, and leverages GPT-4 for natural language understanding and patient profile extraction.

### What it does:
- **Intelligent Search**: Find relevant clinical trials using natural language queries
- **Patient Matching**: Extract patient information from text and match with suitable trials
- **Conversational AI**: Ask questions about trials in plain English
- **Detailed Analysis**: Get AI-powered explanations of why trials match patient profiles
- **Real-time Data**: Access up-to-date information from ClinicalTrials.gov

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   PostgreSQL    │
│   (Frontend)    │    │   (Backend)     │    │   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   GPT-4 Client  │
                       │ (OpenAI API)    │
                       └─────────────────┘
```

### Data Flow

```
ClinicalTrials.gov API
         │
         ▼
   Data Ingestion ─────► CSV/JSON Storage
         │                      │
         ▼                      ▼
   BioBERT Embeddings ─────► PostgreSQL + pgvector
         │                      │
         ▼                      ▼
   Patient Text ─────► GPT-4 ─────► Semantic Search ─────► Ranked Results
                        │                                       │
                        ▼                                       ▼
                 Conversation Manager ─────────────────► AI Response
```

### Core Components

1. **Core Business Logic** (`/core/`)
   - GPT-4 client and conversation management
   - Medical prompt templates
   - Patient information extraction

2. **Data & Search** (`/src/`)
   - ClinicalTrials.gov API integration
   - BioBERT embedding generation
   - PostgreSQL vector search
   - Document processing and chunking

3. **API Layer** (`chat_api.py`)
   - FastAPI server with REST endpoints
   - Request validation and error handling
   - Background task management

4. **User Interface** (`streamlit_app.py`)
   - Interactive chat interface
   - Trial visualization and comparison
   - Patient profile management

5. **Configuration** (`config.py`)
   - Unified settings management
   - Environment-aware configuration
   - Database and API settings

### Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Streamlit
- **Database**: PostgreSQL with pgvector extension
- **AI/ML**: OpenAI GPT-4, BioBERT (sentence-transformers)
- **Search**: Semantic vector search with cosine similarity
- **Data Source**: ClinicalTrials.gov API v2

## Features

### Intelligent Search
- **Natural Language Queries**: Search using everyday language
- **Semantic Understanding**: Find trials based on meaning, not just keywords
- **Advanced Filtering**: Filter by condition, location, phase, status
- **Relevance Ranking**: AI-powered result ranking and scoring

### Patient Profile Matching
- **Text-to-Profile Extraction**: Automatically extract patient information from descriptions
- **Smart Matching**: Match patients with suitable trials based on eligibility criteria
- **Compatibility Analysis**: Detailed explanations of why trials match or don't match
- **Medical History Understanding**: Parse complex medical histories and conditions

### Conversational Interface
- **Natural Conversations**: Ask questions about trials in plain English
- **Context Awareness**: Maintains conversation history and context
- **Multi-turn Dialogue**: Follow-up questions and clarifications
- **Personalized Responses**: Tailored responses based on user needs

### Comprehensive Trial Information
- **Complete Metadata**: Full trial details including contact information
- **Eligibility Criteria**: Clear presentation of inclusion/exclusion criteria
- **Location Information**: Trial sites, contact details, and geographical data
- **Study Design**: Phase, intervention type, primary outcomes

### Advanced Analytics
- **Match Scoring**: Quantitative compatibility scores
- **Trend Analysis**: Trial landscape insights
- **Performance Metrics**: Search quality and user engagement tracking

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **PostgreSQL**: 15 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for full dataset)
- **Storage**: 10GB free space for data and embeddings
- **OS**: macOS, Linux, or Windows with WSL2

### API Keys Required
- **OpenAI API Key**: Required for GPT-4 access
  - Get yours at [OpenAI Platform](https://platform.openai.com/api-keys)
  - Minimum usage limit: $10/month recommended

### Optional but Recommended
- **Git**: For cloning the repository
- **pgvector**: PostgreSQL extension for vector operations

## Local Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/clinical-trial-finder.git
cd clinical-trial-finder
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install and Configure PostgreSQL

#### macOS (using Homebrew):
```bash
# Install PostgreSQL and pgvector
brew install postgresql pgvector

# Start PostgreSQL service
brew services start postgresql

# Create database
createdb clinical_trials
```

#### Ubuntu/Debian:
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector
sudo apt install postgresql-15-pgvector

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres createdb clinical_trials
```

#### Windows:
```bash
# Install PostgreSQL from official website
# Then install pgvector extension
# Create database using pgAdmin or command line
```

### Step 4: Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use your preferred editor
```

Required environment variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=clinical_trials
DB_USER=your_username
DB_PASSWORD=your_password

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### Step 5: Generate Clinical Trial Dataset
```bash
# Start with a small dataset for testing (recommended)
python main.py --max-trials 600

# For full dataset (takes longer, ~30-60 minutes):
# python main.py --max-trials 30000
```

This will:
- Download data from ClinicalTrials.gov
- Process and clean the data
- Save to `/data/processed/clinical_trials_*.csv`
- Display progress and statistics

### Step 6: Generate Vector Embeddings
```bash
# Generate embeddings from the dataset
python generate_embeddings.py

# For testing with smaller dataset:
# python generate_embeddings.py --test --test-samples 5000
```

This process:
- Loads the clinical trial data
- Creates BioBERT embeddings for semantic search
- Stores vectors in PostgreSQL with pgvector
- Takes 10-20 minutes depending on dataset size

### Step 7: Start the Services
```bash
# Terminal 1: Start the FastAPI backend
python chat_api.py

# Terminal 2: Start the Streamlit frontend
streamlit run streamlit_app.py
```

### Step 8: Access the Application
- **Frontend**: Open http://localhost:8501 in your browser
- **API Documentation**: Visit http://localhost:8000/docs for interactive API docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### Basic Search
1. Open the application at http://localhost:8501
2. Enter a natural language query like:
   - "I'm looking for diabetes trials in California"
   - "Find cancer trials for elderly patients"
   - "What trials are available for heart disease?"

### Patient Profile Matching
1. Navigate to the Patient Profile Matching section
2. Enter a patient description:
```
I am a 45-year-old female with Type 2 diabetes, taking metformin.
I live in California and was diagnosed last month.
```
3. The system will:
   - Extract patient information automatically
   - Find matching trials
   - Explain why each trial is suitable

### Conversational Interface
1. Use the chat interface to ask follow-up questions:
   - "Tell me more about the first trial"
   - "What are the side effects?"
   - "How do I contact the researchers?"

### Advanced Search
Use the API directly for programmatic access:
```python
import requests

# Search for trials
response = requests.post("http://localhost:8000/search", json={
    "query": "diabetes clinical trials",
    "filters": {"phase": "Phase 3", "status": "RECRUITING"},
    "k": 10
})

trials = response.json()["results"]
```

## 📚 API Documentation

### Core Endpoints

#### Chat Interface
```http
POST /chat
Content-Type: application/json

{
    "message": "Tell me about diabetes trials",
    "conversation_id": "optional-uuid",
    "include_search": true
}
```

#### Search Trials
```http
POST /search
Content-Type: application/json

{
    "query": "cancer clinical trials",
    "filters": {
        "phase": "Phase 3",
        "status": "RECRUITING",
        "location": "California"
    },
    "k": 10
}
```

#### Patient Extraction and Matching
```http
POST /patient/extract-and-match
Content-Type: application/json

{
    "patient_text": "65-year-old male with heart disease",
    "num_results": 5
}
```

#### Trial Explanation
```http
POST /trials/{nct_id}/explain
Content-Type: application/json

{
    "conversation_id": "optional-uuid"
}
```

### Response Format
```json
{
    "status": "success",
    "data": {
        "results": [...],
        "total_count": 150,
        "search_time_ms": 45
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Handling
The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (trial not found)
- `500`: Internal Server Error
- `503`: Service Unavailable (system not initialized)

## ⚙️ Configuration

### Environment Variables

#### Required Settings
```env
OPENAI_API_KEY=sk-...           # OpenAI API key
```

#### Database Settings
```env
DB_HOST=localhost              # Database host
DB_PORT=5432                  # Database port
DB_NAME=clinical_trials       # Database name
DB_USER=postgres              # Database user
DB_PASSWORD=password          # Database password
```

#### Optional Settings
```env
# Application
ENVIRONMENT=development         # development, staging, production
DEBUG=true                     # Enable debug logging
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR

# API Configuration
MAX_REQUESTS_PER_MINUTE=50    # Rate limiting
MAX_DAILY_COST_USD=50.0       # Cost protection
```

### Model Configuration
Customize AI behavior in `config.py`:
```python
# GPT-4 Settings
DEFAULT_MODEL = "gpt-4"
TEMPERATURE = 0.3              # Response creativity
MAX_TOKENS = 4000             # Response length
MAX_CONVERSATION_HISTORY = 10  # Context window

# Search Settings
DEFAULT_TOP_K = 10            # Default results count
MIN_SIMILARITY_SCORE = 0.7    # Minimum match threshold
```

### Data Collection Settings
```python
# Trial Collection
CONDITIONS = ["cancer", "diabetes", "heart disease", ...]
MAX_TRIALS_PER_CONDITION = 600
TOTAL_TARGET_TRIALS = 30000

# API Settings
RATE_LIMIT_DELAY = 1.5        # Seconds between requests
MAX_RETRIES = 3               # Retry failed requests
```

## 🛠️ Development

### Project Structure
```
clinical-trial-finder/
├── core/                   # Core business logic
│   ├── conversation_manager.py
│   ├── gpt4_client.py
│   ├── medical_prompts.py
│   └── patient_extraction.py
├── src/                    # Data and search modules
│   ├── advanced_search.py
│   ├── api_client.py
│   ├── data_parser.py
│   ├── embedding_generator.py
│   └── vector_store.py
├── ui/                     # UI components
│   ├── components.py
│   ├── styles.py
│   └── api_client.py
├── data/                   # Generated data
│   ├── raw/
│   └── processed/
├── embeddings/            # Vector embeddings
├── conversations/         # Conversation logs
├── main.py               # Data ingestion
├── chat_api.py          # FastAPI server
├── streamlit_app.py     # Streamlit frontend
├── generate_embeddings.py # Embedding generation
└── config.py            # Configuration
```

### Development Commands

#### Testing
```bash
# Test full integration
python test_full_integration.py

# Test patient extraction
python test_patient_extraction.py

# Test specific functionality
python test_enhanced_explanations.py
```

#### Data Management
```bash
# Update trial data
python main.py --conditions "cancer diabetes" --max-trials 1000

# Regenerate embeddings
python generate_embeddings.py --force-rebuild

# Test with small dataset
python main.py --test
python generate_embeddings.py --test
```

#### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy src/ core/

# Linting
flake8 .
```

### Contributing Guidelines

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with proper testing
4. **Follow code style**: Use black, isort, and type hints
5. **Update documentation** as needed
6. **Submit pull request** with clear description

### Adding New Features

#### New Search Filters
1. Add filter to `config.py`:
```python
ADDITIONAL_FILTERS = ["sponsor_type", "study_design"]
```
2. Update search logic in `advanced_search.py`
3. Add UI controls in `streamlit_app.py`

#### New AI Prompts
1. Add prompt template to `medical_prompts.py`
2. Register in prompt manager
3. Test with various inputs

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Restart if needed
brew services restart postgresql

# Verify connection
psql -d clinical_trials -c "SELECT version();"
```

#### Missing Embeddings
```bash
# Check if embeddings exist
ls -la embeddings/

# Regenerate if missing
python generate_embeddings.py --force-rebuild
```

#### API Connection Issues
```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:8501

# Check logs for errors
tail -f logs/api.log
```

#### Memory Issues
```bash
# Monitor memory usage
htop

# Use smaller dataset for testing
python main.py --test --max-trials 100
```

### Performance Optimization

#### For Large Datasets
- Increase PostgreSQL memory settings
- Use connection pooling
- Implement result caching
- Consider distributed embedding generation

#### For Low Memory Systems
- Use test mode with smaller datasets
- Implement lazy loading
- Reduce embedding dimensions

### Error Messages

#### "System not initialized"
- Run data ingestion: `python main.py`
- Generate embeddings: `python generate_embeddings.py`
- Check database connection

#### "OpenAI API key not found"
- Verify `.env` file exists
- Check `OPENAI_API_KEY` is set correctly
- Ensure sufficient API credits

#### "No trials found"
- Check dataset exists in `/data/processed/`
- Verify embeddings are generated
- Try broader search terms

### Getting Help

1. **Check logs**: Look in `/logs/` directory for error details
2. **API documentation**: Visit http://localhost:8000/docs
3. **System status**: Check http://localhost:8000/stats
4. **GitHub issues**: Report bugs or request features


## Acknowledgments

- **ClinicalTrials.gov**: For providing comprehensive clinical trial data
- **OpenAI**: For GPT-4 API and embedding models
- **BioBERT**: For medical domain language understanding
- **PostgreSQL & pgvector**: For scalable vector search capabilities
- **Streamlit & FastAPI**: For rapid development frameworks

---

