"""Advanced search system with hybrid semantic and metadata filtering."""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from .vector_store import VectorStore
from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class AdvancedClinicalTrialSearch:
    """Advanced search system combining semantic search with metadata filtering."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize advanced search system.
        
        Args:
            vector_store: Initialized vector store with embeddings
            embedding_generator: Embedding generator for queries
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        k: int = 10,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Perform advanced search with semantic + metadata filtering.
        
        Args:
            query: Search query text
            filters: Metadata filters to apply
            k: Number of results to return
            rerank: Whether to apply re-ranking algorithm
            
        Returns:
            List of search results with scores and metadata
        """
        # Parse query to extract components
        parsed_query = self._parse_query(query)
        
        # Generate embedding for semantic search
        query_embedding = self.embedding_generator.generate_query_embedding(
            parsed_query.get('medical_terms', query)
        )
        
        # Combine filters from query parsing and explicit filters
        combined_filters = self._combine_filters(parsed_query.get('filters', {}), filters)
        
        # Get semantic search results
        candidates = self.vector_store.search(
            query_embedding,
            k=k * 3,  # Get more candidates for filtering/reranking
            filter_metadata=combined_filters,
            min_score=0.3
        )
        
        # Apply advanced filtering
        filtered_results = self._apply_advanced_filters(candidates, parsed_query)
        
        # Re-rank results if requested
        if rerank:
            filtered_results = self._rerank_results(filtered_results, parsed_query)
        
        # Return top k results
        return filtered_results[:k]
    
    def _parse_query(self, query: str) -> Dict:
        """
        Parse query to extract medical terms, demographics, and preferences.
        
        Args:
            query: Raw search query
            
        Returns:
            Dictionary with parsed components
        """
        parsed = {
            'medical_terms': query,
            'filters': {},
            'preferences': {}
        }
        
        # Extract age information
        age_match = re.search(r'(\d+)[\s-]*(?:year|yr)s?\s*old', query, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
            parsed['filters']['age'] = age
            # Remove age from medical terms
            parsed['medical_terms'] = re.sub(age_match.group(0), '', query).strip()
        
        # Extract age ranges
        age_range_match = re.search(r'age\s*(\d+)[\s-]*(?:to|-)[\s-]*(\d+)', query, re.IGNORECASE)
        if age_range_match:
            min_age = int(age_range_match.group(1))
            max_age = int(age_range_match.group(2))
            parsed['filters']['age_range'] = (min_age, max_age)
            parsed['medical_terms'] = re.sub(age_range_match.group(0), '', query).strip()
        
        # Extract gender
        if re.search(r'\b(?:women|female|girls?)\b', query, re.IGNORECASE):
            parsed['filters']['gender'] = 'FEMALE'
        elif re.search(r'\b(?:men|male|boys?)\b', query, re.IGNORECASE):
            parsed['filters']['gender'] = 'MALE'
        
        # Extract pediatric/children
        if re.search(r'\b(?:pediatric|children|child|kids?|adolescent)\b', query, re.IGNORECASE):
            parsed['filters']['pediatric'] = True
            parsed['preferences']['pediatric'] = True
        
        # Extract adult/elderly
        if re.search(r'\b(?:adult|elderly|senior)\b', query, re.IGNORECASE):
            parsed['filters']['adult'] = True
        
        # Extract location information
        location_match = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
        if location_match:
            location = location_match.group(1)
            parsed['filters']['location'] = location
            parsed['medical_terms'] = re.sub(location_match.group(0), '', query).strip()
        
        # Extract recruitment status preferences
        if re.search(r'\b(?:recruiting|enrolling|accepting)\b', query, re.IGNORECASE):
            parsed['preferences']['recruiting'] = True
        
        if re.search(r'\b(?:new|recent|latest)\b', query, re.IGNORECASE):
            parsed['preferences']['recent'] = True
        
        # Extract trial phase preferences
        phase_match = re.search(r'phase\s*([1-4]|I{1,3}V?)', query, re.IGNORECASE)
        if phase_match:
            phase = phase_match.group(1)
            # Normalize phase notation
            if phase.upper() in ['I', '1']:
                parsed['filters']['phase'] = 'Phase 1'
            elif phase.upper() in ['II', '2']:
                parsed['filters']['phase'] = 'Phase 2'
            elif phase.upper() in ['III', '3']:
                parsed['filters']['phase'] = 'Phase 3'
            elif phase.upper() in ['IV', '4']:
                parsed['filters']['phase'] = 'Phase 4'
        
        # Clean medical terms
        parsed['medical_terms'] = re.sub(r'\s+', ' ', parsed['medical_terms']).strip()
        
        return parsed
    
    def _combine_filters(
        self, 
        parsed_filters: Dict, 
        explicit_filters: Optional[Dict]
    ) -> Dict:
        """Combine filters from query parsing and explicit filters."""
        combined = parsed_filters.copy()
        
        if explicit_filters:
            combined.update(explicit_filters)
        
        return combined
    
    def _apply_advanced_filters(
        self, 
        candidates: List[Dict], 
        parsed_query: Dict
    ) -> List[Dict]:
        """
        Apply advanced filtering logic to candidates.
        
        Args:
            candidates: Search candidates from semantic search
            parsed_query: Parsed query components
            
        Returns:
            Filtered results
        """
        filtered = []
        filters = parsed_query.get('filters', {})
        
        for candidate in candidates:
            metadata = candidate['metadata']
            
            # Age filtering
            if 'age' in filters:
                if not self._matches_age_criteria(metadata, filters['age']):
                    continue
            
            if 'age_range' in filters:
                min_age, max_age = filters['age_range']
                if not self._matches_age_range(metadata, min_age, max_age):
                    continue
            
            # Gender filtering
            if 'gender' in filters:
                trial_gender = metadata.get('Gender', 'ALL')
                if trial_gender != 'ALL' and trial_gender != filters['gender']:
                    continue
            
            # Pediatric filtering
            if filters.get('pediatric'):
                if not self._is_pediatric_trial(metadata):
                    continue
            
            # Adult filtering
            if filters.get('adult'):
                if not self._is_adult_trial(metadata):
                    continue
            
            # Location filtering
            if 'location' in filters:
                if not self._matches_location(metadata, filters['location']):
                    continue
            
            # Phase filtering
            if 'phase' in filters:
                trial_phase = metadata.get('Phase', '')
                if filters['phase'] not in trial_phase:
                    continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _matches_age_criteria(self, metadata: Dict, target_age: int) -> bool:
        """Check if trial accepts the target age."""
        min_age_str = metadata.get('MinimumAge', '')
        max_age_str = metadata.get('MaximumAge', '')
        
        # Parse minimum age
        min_age = self._parse_age(min_age_str)
        max_age = self._parse_age(max_age_str)
        
        # Check if target age falls within range
        if min_age is not None and target_age < min_age:
            return False
        
        if max_age is not None and target_age > max_age:
            return False
        
        return True
    
    def _matches_age_range(self, metadata: Dict, min_target: int, max_target: int) -> bool:
        """Check if trial overlaps with target age range."""
        trial_min = self._parse_age(metadata.get('MinimumAge', ''))
        trial_max = self._parse_age(metadata.get('MaximumAge', ''))
        
        # Check for any overlap
        if trial_max is not None and min_target > trial_max:
            return False
        
        if trial_min is not None and max_target < trial_min:
            return False
        
        return True
    
    def _parse_age(self, age_str: str) -> Optional[int]:
        """Parse age string to integer."""
        if not age_str:
            return None
        
        # Extract number from strings like "18 Years", "65 Months"
        match = re.search(r'(\d+)', age_str)
        if match:
            age = int(match.group(1))
            
            # Convert months to years if needed
            if 'month' in age_str.lower():
                age = age // 12
            
            return age
        
        return None
    
    def _is_pediatric_trial(self, metadata: Dict) -> bool:
        """Check if trial is pediatric."""
        max_age = self._parse_age(metadata.get('MaximumAge', ''))
        
        # Consider pediatric if max age is 18 or less
        if max_age is not None and max_age <= 18:
            return True
        
        # Also check for pediatric keywords in conditions/title
        text_fields = [
            metadata.get('Condition', ''),
            metadata.get('BriefTitle', '')
        ]
        
        for text in text_fields:
            if re.search(r'\b(?:pediatric|children|child|adolescent)\b', text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_adult_trial(self, metadata: Dict) -> bool:
        """Check if trial is for adults."""
        min_age = self._parse_age(metadata.get('MinimumAge', ''))
        
        # Consider adult if min age is 18 or more
        return min_age is None or min_age >= 18
    
    def _matches_location(self, metadata: Dict, target_location: str) -> bool:
        """Check if trial is in target location."""
        location_fields = [
            metadata.get('LocationCity', ''),
            metadata.get('LocationState', ''),
            metadata.get('LocationCountry', '')
        ]
        
        target_lower = target_location.lower()
        
        for field in location_fields:
            if target_lower in field.lower():
                return True
        
        return False
    
    def _rerank_results(self, results: List[Dict], parsed_query: Dict) -> List[Dict]:
        """
        Re-rank results based on multiple factors.
        
        Args:
            results: Search results to re-rank
            parsed_query: Parsed query components
            
        Returns:
            Re-ranked results
        """
        preferences = parsed_query.get('preferences', {})
        
        for result in results:
            metadata = result['metadata']
            
            # Start with semantic similarity score
            base_score = result['score']
            
            # Apply boost factors
            boost_factor = 1.0
            
            # Boost recruiting trials
            status = metadata.get('OverallStatus', '')
            if preferences.get('recruiting') and status == 'RECRUITING':
                boost_factor *= 1.3
            elif status == 'RECRUITING':
                boost_factor *= 1.1
            elif status in ['COMPLETED', 'TERMINATED', 'WITHDRAWN']:
                boost_factor *= 0.8
            
            # Boost recent trials
            if preferences.get('recent'):
                start_date = metadata.get('StartDate', '')
                if self._is_recent_trial(start_date):
                    boost_factor *= 1.2
            
            # Boost pediatric trials if requested
            if preferences.get('pediatric') and self._is_pediatric_trial(metadata):
                boost_factor *= 1.3
            
            # Apply phase preferences
            phase = metadata.get('Phase', '')
            if phase:
                # Prefer Phase 2 and 3 trials (most informative)
                if 'Phase 2' in phase or 'Phase 3' in phase:
                    boost_factor *= 1.1
                elif 'Phase 1' in phase:
                    boost_factor *= 0.9
            
            # Calculate final score
            result['reranked_score'] = base_score * boost_factor
            result['boost_factor'] = boost_factor
        
        # Sort by reranked score
        results.sort(key=lambda x: x.get('reranked_score', x['score']), reverse=True)
        
        return results
    
    def _is_recent_trial(self, start_date: str) -> bool:
        """Check if trial started recently (within 5 years)."""
        if not start_date:
            return False
        
        try:
            # Parse date (format: YYYY-MM or YYYY)
            if len(start_date) == 4:  # Just year
                year = int(start_date)
            else:  # YYYY-MM format
                year = int(start_date[:4])
            
            current_year = datetime.now().year
            return (current_year - year) <= 5
            
        except (ValueError, IndexError):
            return False