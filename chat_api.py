"""FastAPI integration for clinical trial chat system."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from core.conversation_manager import ConversationManager
from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator
from src.advanced_search import AdvancedClinicalTrialSearch
from config import get_env_config
from core.patient_extraction import PatientInfoExtractor, PatientMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message", min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    conversation_type: str = Field("general_inquiry", description="Type of conversation")
    include_search: bool = Field(True, description="Whether to include clinical trial search")

class ChatResponse(BaseModel):
    content: str
    conversation_id: str
    search_results: Optional[List[Dict]] = None
    metadata: Dict[str, Any]

class ConversationStartRequest(BaseModel):
    conversation_type: str = Field("general_inquiry", description="Type of conversation to start")
    initial_context: Optional[Dict] = Field(None, description="Initial context for conversation")

class TrialExplanationRequest(BaseModel):
    nct_id: str = Field(..., description="NCT ID of trial to explain")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for clinical trials")
    filters: Optional[Dict] = Field(None, description="Additional search filters")
    k: int = Field(5, description="Number of results to return", ge=1, le=20)

class PatientExtractionRequest(BaseModel):
    patient_text: str = Field(..., description="Patient description text", min_length=10, max_length=2000)
    num_results: int = Field(10, description="Number of trials to match", ge=1, le=20)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trial Finder Chat API",
    description="Conversational AI for clinical trial information and search",
    version="1.0.0"
)

# Add CORS middleware
env_config = get_env_config()
cors_origins = env_config.get("CORS_ORIGINS", "").split(",") if env_config.get("CORS_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for system components
conversation_manager: Optional[ConversationManager] = None
search_engine: Optional[AdvancedClinicalTrialSearch] = None
patient_extractor: Optional[PatientInfoExtractor] = None
patient_matcher: Optional[PatientMatcher] = None

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global conversation_manager, search_engine, patient_extractor, patient_matcher
    
    try:
        logger.info("Initializing clinical trial search system...")
        
        # Load embeddings and search system
        vector_store = VectorStore()
        vector_store.load()
        
        embedding_generator = EmbeddingGenerator()
        search_engine = AdvancedClinicalTrialSearch(vector_store, embedding_generator)
        
        # Initialize conversation manager with search integration
        conversation_manager = ConversationManager(search_engine=search_engine)
        
        # Initialize patient extraction components
        # Note: Using conversation_manager's GPT-4 client for extraction
        patient_extractor = PatientInfoExtractor(conversation_manager.gpt4_client)
        patient_matcher = PatientMatcher(search_engine, patient_extractor)
        
        logger.info("System initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Clinical Trial Finder Chat API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "chat": "/chat",
            "start_conversation": "/conversations/start",
            "explain_trial": "/trials/{nct_id}/explain",
            "search": "/search",
            "patient_extract": "/patient/extract",
            "patient_match": "/patient/extract-and-match",
            "conversations_list": "/conversations",
            "conversation_load": "/conversations/{conversation_id}/load",
            "conversation_delete": "/conversations/{conversation_id}",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for conversational AI.
    
    Processes user messages and returns AI-generated responses with optional
    clinical trial search integration.
    """
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Start new conversation if none provided
        if not request.conversation_id:
            conversation_id = await conversation_manager.start_conversation(
                conversation_type=request.conversation_type
            )
        else:
            conversation_id = request.conversation_id
            
            # Verify conversation exists
            if not conversation_manager.get_conversation(conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Process message
        response = await conversation_manager.process_message(
            conversation_id=conversation_id,
            user_message=request.message,
            include_search=request.include_search
        )
        
        # Schedule cleanup of expired conversations
        background_tasks.add_task(conversation_manager.cleanup_expired_conversations)
        
        return ChatResponse(**response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/conversations/start")
async def start_conversation(request: ConversationStartRequest):
    """Start a new conversation with optional initial context."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        conversation_id = await conversation_manager.start_conversation(
            conversation_type=request.conversation_type,
            initial_context=request.initial_context
        )
        
        return {
            "conversation_id": conversation_id,
            "conversation_type": request.conversation_type,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/trials/{nct_id}/explain")
async def explain_trial(nct_id: str, request: TrialExplanationRequest):
    """Get AI explanation of a specific clinical trial."""
    if not conversation_manager or not search_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Search for the specific trial
        trial_query = f"NCT ID {nct_id}"
        search_results = search_engine.search(trial_query, k=1)
        
        if not search_results:
            raise HTTPException(status_code=404, detail=f"Trial {nct_id} not found")
        
        trial = search_results[0]
        
        # Create explanation request
        explanation_message = f"Please explain this clinical trial in patient-friendly language: {trial['metadata'].get('BriefTitle', 'Unknown title')} (NCT ID: {nct_id})"
        
        # Start new conversation or use existing one
        if not request.conversation_id:
            conversation_id = await conversation_manager.start_conversation(
                conversation_type="trial_explanation",
                initial_context={"nct_id": nct_id, "trial_data": trial}
            )
        else:
            conversation_id = request.conversation_id
        
        # Process explanation request
        response = await conversation_manager.process_message(
            conversation_id=conversation_id,
            user_message=explanation_message,
            include_search=False  # We already have the specific trial
        )
        
        # Add trial details to response
        response["trial_details"] = {
            "nct_id": trial["NCTId"],
            "title": trial["metadata"].get("BriefTitle", ""),
            "status": trial["metadata"].get("OverallStatus", ""),
            "phase": trial["metadata"].get("Phase", ""),
            "condition": trial["metadata"].get("Condition", ""),
            "location": trial["metadata"].get("LocationState", "")
        }
        
        return ChatResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining trial {nct_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
async def search_trials(request: SearchRequest):
    """Search for clinical trials with advanced filtering."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        results = search_engine.search(
            query=request.query,
            filters=request.filters,
            k=request.k,
            rerank=True
        )
        
        # Format results for API response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "nct_id": result["NCTId"],
                "title": result["metadata"].get("BriefTitle", ""),
                "condition": result["metadata"].get("Condition", ""),
                "status": result["metadata"].get("OverallStatus", ""),
                "phase": result["metadata"].get("Phase", ""),
                "location": result["metadata"].get("LocationState", ""),
                "age_range": f"{result['metadata'].get('MinimumAge', 'N/A')} - {result['metadata'].get('MaximumAge', 'N/A')}",
                "gender": result["metadata"].get("Gender", ""),
                "score": result.get("reranked_score", result["score"]),
                "summary": result["metadata"].get("BriefSummary", "")[:200] + "..." if result["metadata"].get("BriefSummary") else "",
                "metadata": result["metadata"]  # Include complete metadata for UI
            })
        
        return {
            "query": request.query,
            "filters": request.filters,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/patient/extract-and-match")
async def extract_and_match_patient(request: PatientExtractionRequest):
    """Extract patient information from text and match with clinical trials."""
    if not patient_matcher:
        raise HTTPException(status_code=503, detail="Patient extraction system not initialized")
    
    try:
        # Extract patient info and match trials
        result = await patient_matcher.match_patient(
            patient_text=request.patient_text,
            num_results=request.num_results
        )
        
        # Format the matched trials for response
        formatted_trials = []
        for trial in result.get("matched_trials", []):
            formatted_trials.append({
                "nct_id": trial["NCTId"],
                "title": trial["metadata"].get("BriefTitle", ""),
                "condition": trial["metadata"].get("Condition", ""),
                "status": trial["metadata"].get("OverallStatus", ""),
                "phase": trial["metadata"].get("Phase", ""),
                "location": trial["metadata"].get("LocationState", ""),
                "score": trial.get("reranked_score", trial["score"]),
                "metadata": trial["metadata"]  # Include complete metadata for UI
            })
        
        return {
            "patient_profile": result["patient_info"],
            "patient_summary": result["patient_summary"],
            "search_query": result["search_query"],
            "search_filters": result["search_filters"],
            "matched_trials": formatted_trials,
            "total_matches": result["match_count"],
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Error in patient extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/patient/extract")
async def extract_patient_info(request: PatientExtractionRequest):
    """Extract patient information from text without matching trials."""
    if not patient_extractor:
        raise HTTPException(status_code=503, detail="Patient extraction system not initialized")
    
    try:
        # Extract patient information
        patient_info = await patient_extractor.extract_from_text(request.patient_text)
        
        # Generate search query and filters
        search_query = patient_extractor.create_search_query(patient_info)
        search_filters = patient_extractor.create_filters(patient_info)
        patient_summary = patient_extractor.generate_summary(patient_info)
        
        return {
            "patient_info": patient_info,
            "patient_summary": patient_summary,
            "suggested_search_query": search_query,
            "suggested_filters": search_filters,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error extracting patient information: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversations")
async def list_conversations(limit: Optional[int] = 20):
    """List saved conversations with summary information."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        conversations = conversation_manager.list_saved_conversations(limit=limit)
        
        return {
            "conversations": conversations,
            "total_count": len(conversations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/conversations/{conversation_id}/load")
async def load_conversation(conversation_id: str):
    """Load a saved conversation back into memory."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        conversation = conversation_manager.load_conversation_from_state(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found in saved state")
        
        return {
            "conversation_id": conversation.conversation_id,
            "conversation_type": conversation.conversation_type,
            "message_count": len(conversation.messages),
            "created_at": conversation.created_at.isoformat(),
            "last_activity": conversation.last_activity.isoformat(),
            "status": "loaded",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, permanent: bool = False):
    """
    Delete a conversation. 
    
    Args:
        conversation_id: ID of conversation to delete
        permanent: If True, delete from persistent storage too
    """
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Try to end active conversation first
    conversation_manager.end_conversation(conversation_id)
    
    # If permanent deletion requested, remove from persistent state too
    if permanent:
        deleted = conversation_manager.delete_saved_conversation(conversation_id)
        if not deleted:
            # Check if it was at least in memory
            if conversation_id not in conversation_manager.conversations:
                raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "status": "permanently deleted" if permanent else "ended",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": conversation_manager is not None and search_engine is not None
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = conversation_manager.get_stats()
    
    if search_engine:
        vector_stats = search_engine.vector_store.get_stats()
        stats["search_system"] = vector_stats
    
    stats["timestamp"] = datetime.now().isoformat()
    
    return stats

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "chat_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )