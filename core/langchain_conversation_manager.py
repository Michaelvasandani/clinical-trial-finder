"""LangChain-enhanced conversation management for clinical trial AI assistant."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain.schema import HumanMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback

from core.langchain_gpt4_client import LangChainGPT4Client
from core.langchain_medical_prompts import langchain_medical_prompts
from core.langchain_memory import langchain_memory_manager
from core.langchain_chains import langchain_workflow_manager
from core.conversation_state import ConversationStateManager
from config import (
    MAX_CONVERSATION_HISTORY, MAX_CONTEXT_LENGTH, CONVERSATION_TIMEOUT,
    CONVERSATION_TYPES, CONVERSATIONS_DIR, LOG_CONVERSATIONS
)

logger = logging.getLogger(__name__)

class LangChainConversationManager:
    """Enhanced conversation manager using LangChain for improved AI interactions."""

    def __init__(self, search_engine=None, enable_persistence=True):
        """
        Initialize LangChain conversation manager.

        Args:
            search_engine: Optional AdvancedClinicalTrialSearch instance
            enable_persistence: Whether to enable conversation state persistence
        """
        self.langchain_client = LangChainGPT4Client()
        self.medical_prompts = langchain_medical_prompts
        self.memory_manager = langchain_memory_manager
        self.workflow_manager = langchain_workflow_manager
        self.search_engine = search_engine

        # Active conversations
        self.conversations: Dict[str, LangChainConversation] = {}

        # State persistence
        self.enable_persistence = enable_persistence
        self.state_manager = ConversationStateManager() if enable_persistence else None

        logger.info(f"LangChain ConversationManager initialized (persistence: {enable_persistence})")

    async def start_conversation(
        self,
        conversation_type: str = "general_inquiry",
        initial_context: Optional[Dict] = None
    ) -> str:
        """
        Start a new conversation using LangChain.

        Args:
            conversation_type: Type of conversation (general_inquiry, trial_explanation, etc.)
            initial_context: Optional context to start with

        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())

        conversation = LangChainConversation(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            config=CONVERSATION_TYPES.get(conversation_type, CONVERSATION_TYPES["general_inquiry"]),
            initial_context=initial_context,
            memory_manager=self.memory_manager
        )

        self.conversations[conversation_id] = conversation

        # Initialize LangChain memory for this conversation
        memory = self.memory_manager.get_buffer_memory(conversation_id)

        # Add initial context to memory if provided
        if initial_context:
            context_message = f"Initial conversation context: {json.dumps(initial_context, indent=2)}"
            memory.chat_memory.add_message(HumanMessage(content=context_message))

        # Save initial state if persistence enabled
        if self.enable_persistence and self.state_manager:
            # Convert to legacy format for state manager compatibility
            legacy_conversation = self._convert_to_legacy_conversation(conversation)
            self.state_manager.save_conversation(legacy_conversation)

        logger.info(f"Started new LangChain conversation: {conversation_id} (type: {conversation_type})")
        return conversation_id

    async def process_message(
        self,
        conversation_id: str,
        user_message: str,
        include_search: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message using LangChain workflows.

        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            include_search: Whether to include clinical trial search

        Returns:
            Response dictionary with content, metadata, and search results
        """
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Check if conversation has expired
        if conversation.is_expired():
            raise ValueError(f"Conversation {conversation_id} has expired")

        try:
            # Perform clinical trial search if requested and available
            search_results = None
            if include_search and self.search_engine and conversation.config.get("include_search", True):
                search_results = await self._search_trials(user_message)

            # Determine if this is a patient extraction task
            is_patient_extraction = self._is_patient_extraction_query(user_message)

            # Process using appropriate LangChain workflow
            if is_patient_extraction:
                response_content = await self._handle_patient_extraction(
                    conversation_id, user_message, search_results
                )
            else:
                response_content = await self._handle_general_conversation(
                    conversation_id, user_message, search_results, conversation.conversation_type
                )

            # Update conversation activity
            conversation.last_activity = datetime.now()

            # Get memory stats for metadata
            memory_stats = self.memory_manager.get_memory_stats()

            # Prepare full response
            full_response = {
                "content": response_content,
                "conversation_id": conversation_id,
                "search_results": search_results,
                "metadata": {
                    "model": self.langchain_client.llm.model_name,
                    "conversation_type": conversation.conversation_type,
                    "message_count": len(self.memory_manager.get_conversation_history(conversation_id)),
                    "langchain_enhanced": True,
                    "memory_stats": memory_stats.get(conversation_id, {}),
                    "workflow_used": "patient_extraction" if is_patient_extraction else "general_conversation"
                }
            }

            # Add usage information if available from LangChain client
            usage_stats = self.langchain_client.get_usage_stats()
            if usage_stats:
                full_response["metadata"]["usage_stats"] = usage_stats

            # Save updated conversation state if persistence enabled
            if self.enable_persistence and self.state_manager:
                legacy_conversation = self._convert_to_legacy_conversation(conversation)
                self.state_manager.save_conversation(legacy_conversation)

            # Log conversation if enabled
            if LOG_CONVERSATIONS:
                self._log_conversation_turn(conversation_id, user_message, response_content, search_results)

            logger.info(f"Processed LangChain message in conversation {conversation_id}")
            return full_response

        except Exception as e:
            logger.error(f"Error processing message in LangChain conversation {conversation_id}: {e}")
            # Add error to memory for context
            memory = self.memory_manager.get_buffer_memory(conversation_id)
            memory.chat_memory.add_message(HumanMessage(content=f"Error occurred: {str(e)}"))
            raise

    def _is_patient_extraction_query(self, user_message: str) -> bool:
        """Determine if the user message requires patient information extraction."""
        # Simple heuristics - could be made more sophisticated
        patient_keywords = [
            "years old", "year old", "patient", "diagnosed with", "taking medication",
            "medical history", "condition", "symptoms", "treatment", "lives in", "located in"
        ]

        return any(keyword in user_message.lower() for keyword in patient_keywords)

    async def _handle_patient_extraction(
        self,
        conversation_id: str,
        user_message: str,
        search_results: Optional[List[Dict]]
    ) -> str:
        """Handle patient extraction using LangChain workflows."""
        try:
            # Extract patient information using LangChain workflow
            patient_profile = await self.workflow_manager.extract_patient_info(user_message)

            # Add to memory
            self.memory_manager.add_exchange(
                conversation_id=conversation_id,
                user_input=user_message,
                ai_output=f"Extracted patient profile: {patient_profile.json()}"
            )

            # If we have search results, analyze compatibility
            if search_results:
                compatibility_analyses = []

                for trial in search_results[:3]:  # Limit to top 3 for detailed analysis
                    try:
                        compatibility = await self.workflow_manager.analyze_trial_compatibility(
                            patient_profile=patient_profile,
                            trial_info=trial
                        )
                        compatibility_analyses.append({
                            "trial": trial,
                            "compatibility": compatibility
                        })
                    except Exception as e:
                        logger.error(f"Error analyzing compatibility for trial {trial.get('nct_id', 'unknown')}: {e}")

                # Generate comprehensive response
                response = self._format_patient_extraction_response(
                    patient_profile, search_results, compatibility_analyses
                )
            else:
                # Generate response without search results
                response = self._format_patient_profile_response(patient_profile)

            return response

        except Exception as e:
            logger.error(f"Error in patient extraction workflow: {e}")
            # Fallback to general conversation
            return await self._handle_general_conversation(
                conversation_id, user_message, search_results, "general_medical"
            )

    async def _handle_general_conversation(
        self,
        conversation_id: str,
        user_message: str,
        search_results: Optional[List[Dict]],
        conversation_type: str
    ) -> str:
        """Handle general conversation using LangChain memory and prompts."""
        try:
            # Prepare context with search results if available
            enhanced_input = user_message
            if search_results:
                search_context = self._format_search_context(search_results)
                enhanced_input = f"{user_message}\n\nAvailable clinical trials:\n{search_context}"

            # Generate response using LangChain conversation chain with memory
            response = await self.workflow_manager.generate_conversation_response(
                conversation_id=conversation_id,
                user_input=enhanced_input,
                conversation_type=conversation_type
            )

            return response

        except Exception as e:
            logger.error(f"Error in general conversation workflow: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question."

    def _format_patient_extraction_response(
        self,
        patient_profile,
        search_results: List[Dict],
        compatibility_analyses: List[Dict]
    ) -> str:
        """Format response for patient extraction with trial analysis."""
        response_parts = []

        # Patient profile summary
        response_parts.append("I've analyzed your profile and found some relevant clinical trials for you.")
        response_parts.append("")

        # Add patient summary
        if hasattr(patient_profile, 'age') and patient_profile.age:
            age_str = f"{patient_profile.age}-year-old"
        else:
            age_str = ""

        if hasattr(patient_profile, 'gender') and patient_profile.gender:
            gender_str = patient_profile.gender.lower()
        else:
            gender_str = "individual"

        conditions = []
        if hasattr(patient_profile, 'conditions') and patient_profile.conditions:
            conditions = patient_profile.conditions

        location = ""
        if hasattr(patient_profile, 'location') and patient_profile.location:
            location = f" located in {patient_profile.location}"

        condition_str = ", ".join(conditions) if conditions else "your condition"

        response_parts.append(f"**Patient:** {age_str} {gender_str} with {condition_str}{location}.")
        response_parts.append("")

        # Search results summary
        response_parts.append(f"I found {len(search_results)} clinical trials that may be relevant to your situation. Let me show you the most promising options below.")
        response_parts.append("")

        # Detailed compatibility analyses
        for i, analysis in enumerate(compatibility_analyses, 1):
            trial = analysis["trial"]
            compatibility = analysis["compatibility"]

            response_parts.append(f"## {i}. {trial.get('title', 'Clinical Trial')}")
            response_parts.append(f"**NCT ID:** {trial.get('nct_id', 'Unknown')}")
            response_parts.append(f"**Status:** {trial.get('status', 'Unknown')}")
            response_parts.append(f"**Phase:** {trial.get('phase', 'Unknown')}")
            response_parts.append(f"**Location:** {trial.get('location', 'Unknown')}")
            response_parts.append("")

            if hasattr(compatibility, 'compatibility_score'):
                response_parts.append(f"**Compatibility:** {compatibility.compatibility_score}")

            if hasattr(compatibility, 'matching_criteria') and compatibility.matching_criteria:
                response_parts.append("**Why this trial might be suitable:**")
                for criterion in compatibility.matching_criteria:
                    response_parts.append(f"• {criterion}")
                response_parts.append("")

            if hasattr(compatibility, 'potential_concerns') and compatibility.potential_concerns:
                response_parts.append("**Important considerations:**")
                for concern in compatibility.potential_concerns:
                    response_parts.append(f"• {concern}")
                response_parts.append("")

            if hasattr(compatibility, 'questions_for_doctor') and compatibility.questions_for_doctor:
                response_parts.append("**Questions to ask your doctor:**")
                for question in compatibility.questions_for_doctor[:3]:  # Limit to top 3
                    response_parts.append(f"• {question}")
                response_parts.append("")

            response_parts.append("---")
            response_parts.append("")

        # Add disclaimer
        response_parts.extend([
            "**Important:** This analysis is for educational purposes only. Please discuss these trials with your healthcare team to determine which options might be appropriate for your specific situation.",
            "",
            "Would you like me to explain any of these trials in more detail or help you search for additional options?"
        ])

        return "\n".join(response_parts)

    def _format_patient_profile_response(self, patient_profile) -> str:
        """Format response for patient profile without search results."""
        response_parts = []

        response_parts.append("I've extracted the following information from your description:")
        response_parts.append("")

        if hasattr(patient_profile, 'age') and patient_profile.age:
            response_parts.append(f"• **Age:** {patient_profile.age}")

        if hasattr(patient_profile, 'gender') and patient_profile.gender:
            response_parts.append(f"• **Gender:** {patient_profile.gender}")

        if hasattr(patient_profile, 'location') and patient_profile.location:
            response_parts.append(f"• **Location:** {patient_profile.location}")

        if hasattr(patient_profile, 'conditions') and patient_profile.conditions:
            response_parts.append(f"• **Conditions:** {', '.join(patient_profile.conditions)}")

        if hasattr(patient_profile, 'medications') and patient_profile.medications:
            response_parts.append(f"• **Medications:** {', '.join(patient_profile.medications)}")

        response_parts.append("")
        response_parts.append("Would you like me to search for clinical trials that might be relevant to this profile?")

        return "\n".join(response_parts)

    async def _search_trials(self, query: str) -> Optional[List[Dict]]:
        """
        Search for clinical trials based on user query.

        Args:
            query: User's search query

        Returns:
            List of trial results or None if search fails
        """
        if not self.search_engine:
            return None

        try:
            results = self.search_engine.search(
                query=query,
                k=5,  # Limit to top 5 results for conversation
                rerank=True
            )

            # Format results for conversation context
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "nct_id": result["NCTId"],
                    "title": result["metadata"].get("BriefTitle", ""),
                    "condition": result["metadata"].get("Condition", ""),
                    "status": result["metadata"].get("OverallStatus", ""),
                    "phase": result["metadata"].get("Phase", ""),
                    "location": result["metadata"].get("LocationState", ""),
                    "score": result.get("reranked_score", result["score"]),
                    "metadata": result["metadata"]  # Include full metadata for analysis
                })

            logger.info(f"Found {len(formatted_results)} trials for query: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching trials: {e}")
            return None

    def _format_search_context(self, search_results: List[Dict]) -> str:
        """Format search results for inclusion in conversation context."""
        if not search_results:
            return "No relevant clinical trials found in current search."

        context = f"Found {len(search_results)} relevant clinical trials:\n\n"

        for i, trial in enumerate(search_results, 1):
            context += f"{i}. {trial['title']}\n"
            context += f"   NCT ID: {trial['nct_id']}\n"
            context += f"   Condition: {trial['condition']}\n"
            context += f"   Status: {trial['status']}\n"
            context += f"   Phase: {trial['phase']}\n"
            if trial['location']:
                context += f"   Location: {trial['location']}\n"
            context += f"   Relevance Score: {trial['score']:.3f}\n\n"

        return context

    def _convert_to_legacy_conversation(self, langchain_conversation: "LangChainConversation"):
        """Convert LangChain conversation to legacy format for state persistence."""
        from core.conversation_manager import Conversation

        # Get messages from LangChain memory
        memory_messages = self.memory_manager.get_conversation_history(langchain_conversation.conversation_id)

        # Convert to legacy message format
        legacy_messages = []
        for msg in memory_messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    role = 'user'
                elif msg.__class__.__name__ == 'AIMessage':
                    role = 'assistant'
                else:
                    role = 'system'

                legacy_messages.append({
                    "role": role,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                    "type": "message"
                })

        # Create legacy conversation object
        legacy_conv = Conversation(
            conversation_id=langchain_conversation.conversation_id,
            conversation_type=langchain_conversation.conversation_type,
            config=langchain_conversation.config,
            initial_context=langchain_conversation.initial_context
        )

        # Set timestamps and messages
        legacy_conv.created_at = langchain_conversation.created_at
        legacy_conv.last_activity = langchain_conversation.last_activity
        legacy_conv.messages = legacy_messages

        return legacy_conv

    def _log_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        search_results: Optional[List[Dict]]
    ) -> None:
        """Log a conversation turn for analysis (anonymized)."""
        if not LOG_CONVERSATIONS:
            return

        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,  # UUID, no personal info
                "user_message_length": len(user_message),
                "assistant_response_length": len(assistant_response),
                "search_results_count": len(search_results) if search_results else 0,
                "has_search_results": bool(search_results),
                "langchain_enhanced": True
            }

            # Save to daily log file
            log_file = CONVERSATIONS_DIR / f"conversations_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Error logging conversation: {e}")

    def get_conversation(self, conversation_id: str) -> Optional["LangChainConversation"]:
        """Get conversation by ID."""
        return self.conversations.get(conversation_id)

    def end_conversation(self, conversation_id: str) -> bool:
        """End and clean up a conversation."""
        if conversation_id in self.conversations:
            # Save final state before removing from memory
            if self.enable_persistence and self.state_manager:
                legacy_conversation = self._convert_to_legacy_conversation(self.conversations[conversation_id])
                self.state_manager.save_conversation(legacy_conversation)

            # Clear LangChain memory
            self.memory_manager.clear_memory(conversation_id)

            del self.conversations[conversation_id]
            logger.info(f"Ended LangChain conversation: {conversation_id}")
            return True
        return False

    def cleanup_expired_conversations(self) -> int:
        """Clean up expired conversations. Returns number cleaned up."""
        expired_ids = [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.is_expired()
        ]

        for conv_id in expired_ids:
            self.memory_manager.clear_memory(conv_id)
            del self.conversations[conv_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired LangChain conversations")

        return len(expired_ids)

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics."""
        active_conversations = len(self.conversations)

        # Get conversation types distribution
        type_distribution = {}
        for conv in self.conversations.values():
            conv_type = conv.conversation_type
            type_distribution[conv_type] = type_distribution.get(conv_type, 0) + 1

        stats = {
            "active_conversations": active_conversations,
            "conversation_types": type_distribution,
            "search_enabled": self.search_engine is not None,
            "langchain_enhanced": True,
            "langchain_client_stats": self.langchain_client.get_model_info(),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "workflow_stats": self.workflow_manager.get_chain_stats(),
            "persistence_enabled": self.enable_persistence
        }

        # Add usage stats if available
        usage_stats = self.langchain_client.get_usage_stats()
        if usage_stats:
            stats["usage_stats"] = usage_stats

        # Add persistence stats if enabled
        if self.enable_persistence and self.state_manager:
            stats["persistence_stats"] = self.state_manager.get_stats()

        return stats


class LangChainConversation:
    """Individual LangChain-enhanced conversation with memory management."""

    def __init__(
        self,
        conversation_id: str,
        conversation_type: str,
        config: Dict[str, Any],
        initial_context: Optional[Dict] = None,
        memory_manager=None
    ):
        """Initialize LangChain conversation."""
        self.conversation_id = conversation_id
        self.conversation_type = conversation_type
        self.config = config
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.initial_context = initial_context
        self.memory_manager = memory_manager

    def is_expired(self) -> bool:
        """Check if conversation has expired."""
        return datetime.now() - self.last_activity > timedelta(seconds=CONVERSATION_TIMEOUT)

    def get_duration(self) -> timedelta:
        """Get conversation duration."""
        return self.last_activity - self.created_at

    def get_message_count(self) -> int:
        """Get number of messages in conversation."""
        if self.memory_manager:
            messages = self.memory_manager.get_conversation_history(self.conversation_id)
            return len(messages)
        return 0