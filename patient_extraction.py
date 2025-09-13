"""Simple patient information extraction using GPT-4."""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PatientInfoExtractor:
    """Extract structured patient information from text using GPT-4."""
    
    def __init__(self, gpt4_client=None):
        """
        Initialize the patient extractor.
        
        Args:
            gpt4_client: Optional GPT-4 client instance. If not provided,
                        will be initialized when needed.
        """
        self.gpt4_client = gpt4_client
        
    async def extract_from_text(self, patient_text: str) -> Dict[str, Any]:
        """
        Extract structured patient information from free text.
        
        Args:
            patient_text: Free text description of patient
            
        Returns:
            Dictionary containing extracted patient information
        """
        if not patient_text:
            return {}
        
        # Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(patient_text)
        
        try:
            # Use GPT-4 to extract information
            if self.gpt4_client:
                messages = [
                    {"role": "system", "content": "You are a medical information extraction assistant. Extract patient information and return it as valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ]
                
                response = await self.gpt4_client.chat_completion(
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=1000
                )
                
                # Parse the JSON response
                content = response.get("content", "{}")
                # Extract JSON from the response (in case it's wrapped in text)
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    patient_info = json.loads(json_match.group())
                else:
                    patient_info = {}
            else:
                # Fallback to regex-based extraction if no GPT-4
                patient_info = self._regex_extraction(patient_text)
            
            # Validate and clean the extracted information
            patient_info = self._validate_patient_info(patient_info)
            
            logger.info(f"Extracted patient info: {patient_info}")
            return patient_info
            
        except Exception as e:
            logger.error(f"Error extracting patient information: {e}")
            # Fallback to basic extraction
            return self._regex_extraction(patient_text)
    
    def _build_extraction_prompt(self, patient_text: str) -> str:
        """Build the prompt for GPT-4 extraction."""
        return f"""Extract the following patient information from the text below and return ONLY a JSON object with these fields:

{{
    "age": <number or null>,
    "gender": <"male", "female", "other", or null>,
    "conditions": [<list of medical conditions>],
    "medications": [<list of current medications>],
    "location": <city/state or null>,
    "diagnosis_dates": [<list of when conditions were diagnosed if mentioned>],
    "allergies": [<list of any mentioned allergies>],
    "previous_treatments": [<list of past treatments mentioned>],
    "symptoms": [<list of current symptoms>],
    "family_history": [<relevant family medical history>]
}}

Important:
- Use null for missing fields
- Keep medical terms as mentioned
- Extract ALL conditions and medications mentioned
- For age, extract the number only

Patient text:
{patient_text}

Return only the JSON object, no additional text."""
    
    def _regex_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback regex-based extraction for basic information."""
        info = {
            "age": None,
            "gender": None,
            "conditions": [],
            "medications": [],
            "location": None,
            "symptoms": []
        }
        
        # Extract age
        age_patterns = [
            r'(\d+)[\s-]*(?:year|yr)s?[\s-]*old',
            r'age[d]?[\s:]+(\d+)',
            r"i[']?m\s+(\d+)"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["age"] = int(match.group(1))
                break
        
        # Extract gender
        if re.search(r'\b(woman|female|girl|she|her)\b', text, re.IGNORECASE):
            info["gender"] = "female"
        elif re.search(r'\b(man|male|boy|he|his)\b', text, re.IGNORECASE):
            info["gender"] = "male"
        
        # Extract common conditions (basic list)
        condition_keywords = [
            "diabetes", "cancer", "heart disease", "hypertension", "high blood pressure",
            "asthma", "COPD", "arthritis", "alzheimer", "dementia", "depression",
            "anxiety", "HIV", "hepatitis", "kidney disease", "liver disease"
        ]
        for condition in condition_keywords:
            if condition.lower() in text.lower():
                info["conditions"].append(condition)
        
        # Extract location (US states and major cities)
        location_pattern = r'\b(?:in|from|live in|living in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        location_match = re.search(location_pattern, text)
        if location_match:
            info["location"] = location_match.group(1)
        
        return info
    
    def _validate_patient_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted patient information."""
        validated = {}
        
        # Validate age
        if info.get("age"):
            try:
                age = int(info["age"])
                if 0 < age < 120:
                    validated["age"] = age
            except (ValueError, TypeError):
                pass
        
        # Validate gender
        if info.get("gender"):
            gender = str(info["gender"]).lower()
            if gender in ["male", "female", "other", "m", "f"]:
                validated["gender"] = gender[0] if len(gender) == 1 else gender
        
        # Clean conditions list
        if info.get("conditions"):
            conditions = info["conditions"]
            if isinstance(conditions, list):
                validated["conditions"] = [str(c).strip() for c in conditions if c]
            elif isinstance(conditions, str):
                validated["conditions"] = [conditions.strip()]
        
        # Clean medications list
        if info.get("medications"):
            medications = info["medications"]
            if isinstance(medications, list):
                validated["medications"] = [str(m).strip() for m in medications if m]
            elif isinstance(medications, str):
                validated["medications"] = [medications.strip()]
        
        # Validate location
        if info.get("location"):
            validated["location"] = str(info["location"]).strip()
        
        # Include other fields if present
        for field in ["symptoms", "allergies", "previous_treatments", "family_history", "diagnosis_dates"]:
            if info.get(field):
                if isinstance(info[field], list):
                    validated[field] = [str(item).strip() for item in info[field] if item]
                elif info[field]:
                    validated[field] = [str(info[field]).strip()]
        
        return validated
    
    def create_search_query(self, patient_info: Dict[str, Any]) -> str:
        """
        Create an optimized search query from patient information.
        
        Args:
            patient_info: Extracted patient information dictionary
            
        Returns:
            Optimized search query string
        """
        query_parts = []
        
        # Add age and gender
        if patient_info.get("age"):
            query_parts.append(f"{patient_info['age']} year old")
        
        if patient_info.get("gender"):
            gender = patient_info["gender"]
            if gender in ["m", "male"]:
                query_parts.append("man")
            elif gender in ["f", "female"]:
                query_parts.append("woman")
        
        # Add primary conditions
        if patient_info.get("conditions"):
            # Prioritize first 2-3 conditions
            conditions = patient_info["conditions"][:3]
            query_parts.extend(conditions)
        
        # Add location if specified
        if patient_info.get("location"):
            query_parts.append(f"in {patient_info['location']}")
        
        # Add key medications if relevant
        if patient_info.get("medications") and len(query_parts) < 8:
            # Only add medications if query isn't too long
            meds = patient_info["medications"][:2]
            for med in meds:
                query_parts.append(f"taking {med}")
        
        return " ".join(query_parts)
    
    def create_filters(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create search filters from patient information.
        
        Args:
            patient_info: Extracted patient information
            
        Returns:
            Dictionary of filters for the search system
        """
        filters = {}
        
        if patient_info.get("age"):
            filters["age"] = patient_info["age"]
        
        if patient_info.get("gender"):
            gender = patient_info["gender"]
            if gender in ["m", "male"]:
                filters["gender"] = "MALE"
            elif gender in ["f", "female"]:
                filters["gender"] = "FEMALE"
        
        if patient_info.get("location"):
            filters["location"] = patient_info["location"]
        
        return filters
    
    def generate_summary(self, patient_info: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the patient profile.
        
        Args:
            patient_info: Extracted patient information
            
        Returns:
            Readable summary string
        """
        summary_parts = []
        
        # Demographics
        demo_parts = []
        if patient_info.get("age"):
            demo_parts.append(f"{patient_info['age']}-year-old")
        if patient_info.get("gender"):
            demo_parts.append(patient_info["gender"])
        if demo_parts:
            summary_parts.append(" ".join(demo_parts))
        
        # Conditions
        if patient_info.get("conditions"):
            conditions_str = ", ".join(patient_info["conditions"])
            summary_parts.append(f"with {conditions_str}")
        
        # Medications
        if patient_info.get("medications"):
            meds_str = ", ".join(patient_info["medications"])
            summary_parts.append(f"taking {meds_str}")
        
        # Location
        if patient_info.get("location"):
            summary_parts.append(f"located in {patient_info['location']}")
        
        if summary_parts:
            return "Patient: " + " ".join(summary_parts)
        else:
            return "Patient profile extracted"


class PatientMatcher:
    """Match patient profiles to clinical trials."""
    
    def __init__(self, search_engine, extractor: PatientInfoExtractor):
        """
        Initialize the matcher.
        
        Args:
            search_engine: Clinical trial search engine instance
            extractor: Patient info extractor instance
        """
        self.search_engine = search_engine
        self.extractor = extractor
    
    async def match_patient(
        self, 
        patient_text: str, 
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        Extract patient info and find matching trials.
        
        Args:
            patient_text: Patient description text
            num_results: Number of trials to return
            
        Returns:
            Dictionary with patient info and matched trials
        """
        # Extract patient information
        patient_info = await self.extractor.extract_from_text(patient_text)
        
        # Create search query and filters
        search_query = self.extractor.create_search_query(patient_info)
        search_filters = self.extractor.create_filters(patient_info)
        
        # Search for matching trials
        if self.search_engine:
            trials = self.search_engine.search(
                query=search_query,
                filters=search_filters,
                k=num_results,
                rerank=True
            )
        else:
            trials = []
        
        # Generate summary
        patient_summary = self.extractor.generate_summary(patient_info)
        
        return {
            "patient_info": patient_info,
            "patient_summary": patient_summary,
            "search_query": search_query,
            "search_filters": search_filters,
            "matched_trials": trials,
            "match_count": len(trials),
            "timestamp": datetime.now().isoformat()
        }