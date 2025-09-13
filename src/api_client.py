"""API client for ClinicalTrials.gov API v2."""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

import requests
from tqdm import tqdm

from ..config import (
    API_BASE_URL,
    API_FIELDS,
    MAX_RETRIES,
    PAGE_SIZE,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class ClinicalTrialsAPIClient:
    """Client for interacting with ClinicalTrials.gov API v2."""
    
    def __init__(self):
        """Initialize the API client."""
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ClinicalTrialsFinder/1.0"
        })
        
    def _make_request(
        self, 
        params: Dict[str, Any], 
        retry_count: int = 0
    ) -> Optional[Dict]:
        """
        Make a request to the API with retry logic.
        
        Args:
            params: Query parameters for the API request
            retry_count: Current retry attempt number
            
        Returns:
            JSON response from the API or None if failed
        """
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying... (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * (retry_count + 1))
                return self._make_request(params, retry_count + 1)
            
            logger.error(f"Max retries exceeded for request with params: {params}")
            return None
    
    def search_studies(
        self,
        condition: str,
        max_results: int = 100,
        additional_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for clinical trials by condition with pagination.
        
        Args:
            condition: Medical condition to search for
            max_results: Maximum number of results to retrieve
            additional_filters: Additional query filters
            
        Returns:
            List of study records
        """
        all_studies = []
        page_token = None
        
        # Base query parameters
        params = {
            "query.cond": condition,
            "pageSize": min(PAGE_SIZE, max_results),
            "format": "json",
            "fields": API_FIELDS
        }
        
        # Add any additional filters
        if additional_filters:
            params.update(additional_filters)
        
        # Progress bar for pagination
        pbar = tqdm(
            total=max_results,
            desc=f"Fetching {condition} trials",
            unit="trials"
        )
        
        while len(all_studies) < max_results:
            # Add page token if available
            if page_token:
                params["pageToken"] = page_token
            
            # Make API request
            response = self._make_request(params)
            
            if not response:
                logger.warning(f"Failed to fetch data for condition: {condition}")
                break
            
            # Extract studies from response
            studies = response.get("studies", [])
            
            if not studies:
                logger.info(f"No more studies found for condition: {condition}")
                break
            
            # Add studies to collection
            remaining = max_results - len(all_studies)
            studies_to_add = studies[:remaining]
            all_studies.extend(studies_to_add)
            
            # Update progress bar
            pbar.update(len(studies_to_add))
            
            # Check if there are more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(all_studies) >= max_results:
                break
            
            page_token = next_page_token
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
        
        pbar.close()
        
        logger.info(f"Retrieved {len(all_studies)} studies for condition: {condition}")
        return all_studies
    
    def get_study_by_nct_id(self, nct_id: str) -> Optional[Dict]:
        """
        Get a specific study by its NCT ID.
        
        Args:
            nct_id: The NCT identifier for the study
            
        Returns:
            Study record or None if not found
        """
        params = {
            "query.id": nct_id,
            "format": "json",
            "fields": API_FIELDS
        }
        
        response = self._make_request(params)
        
        if response and response.get("studies"):
            return response["studies"][0]
        
        return None
    
    def search_all_conditions(
        self,
        conditions: List[str],
        max_per_condition: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Search for trials across multiple conditions.
        
        Args:
            conditions: List of conditions to search
            max_per_condition: Maximum trials per condition
            
        Returns:
            Dictionary mapping conditions to their study lists
        """
        results = {}
        
        for condition in conditions:
            logger.info(f"Searching for condition: {condition}")
            studies = self.search_studies(condition, max_per_condition)
            results[condition] = studies
            
            # Rate limiting between conditions
            if condition != conditions[-1]:
                time.sleep(RATE_LIMIT_DELAY * 2)
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
        params = {
            "query.cond": "cancer",
            "pageSize": 1,
            "format": "json",
            "fields": "NCTId"
        }
        
        try:
            response = self._make_request(params)
            if response and "studies" in response:
                logger.info("API connection test successful")
                return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
        
        return False