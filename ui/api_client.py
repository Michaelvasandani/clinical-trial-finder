"""API client for communicating with the FastAPI backend."""

import requests
import streamlit as st
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class ClinicalTrialAPIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Unable to connect to the backend. Please make sure the FastAPI server is running.")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.error("❌ The requested resource was not found.")
            elif e.response.status_code == 500:
                st.error("🔧 Server error. Please try again later.")
            else:
                st.error(f"❌ API Error: {e.response.status_code}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"API request failed: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if the API is healthy."""
        result = self._make_request("GET", "/health")
        return result is not None and result.get("status") == "healthy"
    
    def chat(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        conversation_type: str = "general_inquiry",
        include_search: bool = True
    ) -> Optional[Dict]:
        """Send a chat message to the API."""
        payload = {
            "message": message,
            "conversation_type": conversation_type,
            "include_search": include_search
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        return self._make_request("POST", "/chat", json=payload)
    
    def start_conversation(
        self, 
        conversation_type: str = "general_inquiry",
        initial_context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Start a new conversation."""
        payload = {
            "conversation_type": conversation_type
        }
        
        if initial_context:
            payload["initial_context"] = initial_context
        
        return self._make_request("POST", "/conversations/start", json=payload)
    
    def search_trials(
        self, 
        query: str, 
        filters: Optional[Dict] = None,
        k: int = 5
    ) -> Optional[Dict]:
        """Search for clinical trials."""
        payload = {
            "query": query,
            "k": k
        }
        
        if filters:
            payload["filters"] = filters
        
        return self._make_request("POST", "/search", json=payload)
    
    def extract_patient_info(self, patient_text: str) -> Optional[Dict]:
        """Extract patient information from text."""
        payload = {
            "patient_text": patient_text
        }
        
        return self._make_request("POST", "/patient/extract", json=payload)
    
    def extract_and_match_patient(
        self, 
        patient_text: str,
        num_results: int = 10
    ) -> Optional[Dict]:
        """Extract patient info and find matching trials."""
        payload = {
            "patient_text": patient_text,
            "num_results": num_results
        }
        
        return self._make_request("POST", "/patient/extract-and-match", json=payload)
    
    def explain_trial(
        self, 
        nct_id: str,
        conversation_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Get an explanation of a specific trial."""
        payload = {
            "nct_id": nct_id
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        return self._make_request("POST", f"/trials/{nct_id}/explain", json=payload)
    
    def get_conversations(self) -> Optional[List[Dict]]:
        """Get list of conversations."""
        return self._make_request("GET", "/conversations")
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load a conversation from persistent state."""
        return self._make_request("POST", f"/conversations/{conversation_id}/load")
    
    def delete_conversation(
        self, 
        conversation_id: str,
        permanent: bool = False
    ) -> Optional[Dict]:
        """Delete a conversation."""
        params = {"permanent": permanent} if permanent else {}
        return self._make_request("DELETE", f"/conversations/{conversation_id}", params=params)
    
    def get_stats(self) -> Optional[Dict]:
        """Get system statistics."""
        return self._make_request("GET", "/stats")

# Global API client instance
@st.cache_resource
def get_api_client() -> ClinicalTrialAPIClient:
    """Get a cached API client instance."""
    # Try to get API URL from multiple sources for deployment flexibility
    try:
        api_url = st.secrets.get("API_URL", None)
    except Exception:
        api_url = None

    if not api_url:
        # Check multiple environment variable names for compatibility
        api_url = (
            os.environ.get("API_BASE_URL") or  # Render/Vercel standard
            os.environ.get("API_URL") or       # Custom
            os.environ.get("FASTAPI_URL") or   # Alternative
            "http://localhost:8000"            # Development default
        )

    return ClinicalTrialAPIClient(api_url)

def with_loading(func, *args, loading_text="Processing...", **kwargs):
    """Execute a function with a loading indicator."""
    with st.spinner(loading_text):
        return func(*args, **kwargs)