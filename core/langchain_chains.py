"""LangChain chains for complex clinical trial workflows."""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chains.base import Chain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from core.langchain_medical_prompts import langchain_medical_prompts
from core.langchain_memory import langchain_memory_manager
from config import DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class PatientProfile(BaseModel):
    """Structured patient profile."""
    age: Optional[int] = Field(description="Patient's age")
    gender: Optional[str] = Field(description="Patient's gender")
    location: Optional[str] = Field(description="Patient's location/state")
    conditions: List[str] = Field(description="Medical conditions", default_factory=list)
    medications: List[str] = Field(description="Current medications", default_factory=list)
    allergies: List[str] = Field(description="Known allergies", default_factory=list)
    previous_treatments: List[str] = Field(description="Previous treatments", default_factory=list)
    preferences: Dict[str, Any] = Field(description="Trial preferences", default_factory=dict)

class TrialCompatibility(BaseModel):
    """Trial compatibility assessment."""
    compatibility_score: str = Field(description="High/Medium/Low/Not Compatible")
    matching_criteria: List[str] = Field(description="Criteria that match", default_factory=list)
    potential_concerns: List[str] = Field(description="Potential exclusions", default_factory=list)
    questions_for_doctor: List[str] = Field(description="Questions to ask medical team", default_factory=list)
    next_steps: List[str] = Field(description="Recommended next steps", default_factory=list)

class SearchRefinement(BaseModel):
    """Search refinement suggestions."""
    suggested_terms: List[str] = Field(description="Alternative search terms", default_factory=list)
    recommended_filters: Dict[str, str] = Field(description="Suggested filters", default_factory=dict)
    search_strategies: List[str] = Field(description="Search strategy suggestions", default_factory=list)

class LangChainWorkflowManager:
    """Manages complex LangChain workflows for clinical trial tasks."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the workflow manager."""
        self.llm = llm or ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        # Initialize output parsers
        self.patient_parser = PydanticOutputParser(pydantic_object=PatientProfile)
        self.compatibility_parser = PydanticOutputParser(pydantic_object=TrialCompatibility)
        self.search_parser = PydanticOutputParser(pydantic_object=SearchRefinement)

        # Initialize chains
        self._initialize_chains()

        logger.info("LangChain workflow manager initialized")

    def _initialize_chains(self) -> None:
        """Initialize all workflow chains."""

        # Patient Extraction Chain
        self.patient_extraction_chain = self._create_patient_extraction_chain()

        # Trial Matching Chain
        self.trial_matching_chain = self._create_trial_matching_chain()

        # Search Refinement Chain
        self.search_refinement_chain = self._create_search_refinement_chain()

        # Conversation Chain with Memory
        self.conversation_chain = self._create_conversation_chain()

        # Complex Multi-step Chain
        self.patient_trial_workflow = self._create_patient_trial_workflow()

    def _create_patient_extraction_chain(self) -> LLMChain:
        """Create chain for extracting patient information from text."""

        prompt = langchain_medical_prompts.get_conversation_chain("patient_extraction")

        # Add format instructions
        prompt = prompt.partial(
            format_instructions=self.patient_parser.get_format_instructions()
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(
                parser=self.patient_parser,
                llm=self.llm
            )
        )

        return chain

    def _create_trial_matching_chain(self) -> LLMChain:
        """Create chain for analyzing trial compatibility."""

        prompt = langchain_medical_prompts.get_conversation_chain("trial_matching")

        # Add format instructions
        prompt = prompt.partial(
            format_instructions=self.compatibility_parser.get_format_instructions()
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(
                parser=self.compatibility_parser,
                llm=self.llm
            )
        )

        return chain

    def _create_search_refinement_chain(self) -> LLMChain:
        """Create chain for search refinement suggestions."""

        prompt = langchain_medical_prompts.get_conversation_chain("search_refinement")

        # Add format instructions
        prompt = prompt.partial(
            format_instructions=self.search_parser.get_format_instructions()
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(
                parser=self.search_parser,
                llm=self.llm
            )
        )

        return chain

    def _create_conversation_chain(self) -> LLMChain:
        """Create memory-enabled conversation chain."""

        prompt = langchain_medical_prompts.get_system_prompt("general_medical")

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=None  # Memory will be added per conversation
        )

        return chain

    def _create_patient_trial_workflow(self) -> SequentialChain:
        """Create complex multi-step workflow for patient-trial matching."""

        # Step 1: Extract patient information
        def extract_patient_transform(inputs: Dict) -> Dict:
            try:
                patient_result = self.patient_extraction_chain.run(
                    patient_text=inputs["patient_text"]
                )
                return {
                    **inputs,
                    "patient_profile": patient_result.json() if hasattr(patient_result, 'json') else str(patient_result)
                }
            except Exception as e:
                logger.error(f"Error in patient extraction: {e}")
                return {
                    **inputs,
                    "patient_profile": json.dumps({"error": str(e)})
                }

        patient_transform = TransformChain(
            input_variables=["patient_text"],
            output_variables=["patient_profile"],
            transform=extract_patient_transform
        )

        # Step 2: Analyze compatibility (requires external trial info)
        def compatibility_transform(inputs: Dict) -> Dict:
            if "trial_info" not in inputs:
                return {
                    **inputs,
                    "compatibility_analysis": "No trial information provided for analysis"
                }

            try:
                compatibility_result = self.trial_matching_chain.run(
                    patient_profile=inputs["patient_profile"],
                    trial_info=inputs["trial_info"]
                )
                return {
                    **inputs,
                    "compatibility_analysis": compatibility_result.json() if hasattr(compatibility_result, 'json') else str(compatibility_result)
                }
            except Exception as e:
                logger.error(f"Error in compatibility analysis: {e}")
                return {
                    **inputs,
                    "compatibility_analysis": json.dumps({"error": str(e)})
                }

        compatibility_transform = TransformChain(
            input_variables=["patient_profile", "trial_info"],
            output_variables=["compatibility_analysis"],
            transform=compatibility_transform
        )

        # Create sequential chain
        workflow = SequentialChain(
            chains=[patient_transform, compatibility_transform],
            input_variables=["patient_text", "trial_info"],
            output_variables=["patient_profile", "compatibility_analysis"],
            verbose=True
        )

        return workflow

    async def extract_patient_info(self, patient_text: str) -> PatientProfile:
        """Extract structured patient information from text."""
        try:
            result = await self.patient_extraction_chain.arun(patient_text=patient_text)

            if isinstance(result, PatientProfile):
                return result
            elif isinstance(result, dict):
                return PatientProfile(**result)
            else:
                # Try to parse as JSON
                parsed = json.loads(str(result))
                return PatientProfile(**parsed)

        except Exception as e:
            logger.error(f"Error extracting patient info: {e}")
            # Return basic structure
            return PatientProfile()

    async def analyze_trial_compatibility(
        self,
        patient_profile: PatientProfile,
        trial_info: Dict[str, Any]
    ) -> TrialCompatibility:
        """Analyze compatibility between patient and trial."""
        try:
            patient_str = patient_profile.json() if hasattr(patient_profile, 'json') else str(patient_profile)
            trial_str = json.dumps(trial_info, indent=2)

            result = await self.trial_matching_chain.arun(
                patient_profile=patient_str,
                trial_info=trial_str
            )

            if isinstance(result, TrialCompatibility):
                return result
            elif isinstance(result, dict):
                return TrialCompatibility(**result)
            else:
                parsed = json.loads(str(result))
                return TrialCompatibility(**parsed)

        except Exception as e:
            logger.error(f"Error analyzing trial compatibility: {e}")
            return TrialCompatibility(
                compatibility_score="Unable to determine",
                potential_concerns=[f"Analysis error: {str(e)}"],
                questions_for_doctor=["Please review this trial with your medical team"],
                next_steps=["Consult with your healthcare provider"]
            )

    async def get_search_refinement(
        self,
        user_query: str,
        current_results: List[Dict],
        constraints: Optional[Dict] = None
    ) -> SearchRefinement:
        """Get search refinement suggestions."""
        try:
            constraints_str = json.dumps(constraints or {}, indent=2)
            results_str = f"Found {len(current_results)} results"

            result = await self.search_refinement_chain.arun(
                user_query=user_query,
                current_results=results_str,
                constraints=constraints_str
            )

            if isinstance(result, SearchRefinement):
                return result
            elif isinstance(result, dict):
                return SearchRefinement(**result)
            else:
                parsed = json.loads(str(result))
                return SearchRefinement(**parsed)

        except Exception as e:
            logger.error(f"Error getting search refinement: {e}")
            return SearchRefinement(
                suggested_terms=["Try broader search terms"],
                search_strategies=["Consider expanding your search criteria"]
            )

    async def run_patient_trial_workflow(
        self,
        patient_text: str,
        trial_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the complete patient-trial matching workflow."""
        try:
            result = await self.patient_trial_workflow.arun(
                patient_text=patient_text,
                trial_info=json.dumps(trial_info, indent=2)
            )

            return result

        except Exception as e:
            logger.error(f"Error in patient-trial workflow: {e}")
            return {
                "patient_profile": f"Error extracting patient info: {e}",
                "compatibility_analysis": f"Error analyzing compatibility: {e}"
            }

    async def generate_conversation_response(
        self,
        conversation_id: str,
        user_input: str,
        conversation_type: str = "general_medical"
    ) -> str:
        """Generate response using conversation chain with memory."""
        try:
            # Get memory for this conversation
            memory = langchain_memory_manager.get_buffer_memory(conversation_id)

            # Create temporary chain with memory
            prompt = langchain_medical_prompts.get_system_prompt(conversation_type)
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=memory
            )

            # Generate response
            response = await chain.arun(input=user_input)

            logger.debug(f"Generated conversation response for {conversation_id}")
            return response

        except Exception as e:
            logger.error(f"Error generating conversation response: {e}")
            return f"I apologize, but I encountered an error processing your request: {e}"

    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about chain usage."""
        return {
            "chains_available": [
                "patient_extraction_chain",
                "trial_matching_chain",
                "search_refinement_chain",
                "conversation_chain",
                "patient_trial_workflow"
            ],
            "llm_model": self.llm.model_name,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "memory_integration": True,
            "structured_outputs": True
        }

# Global instance
langchain_workflow_manager = LangChainWorkflowManager()