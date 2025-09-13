"""Conversation management for clinical trial AI assistant."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .gpt4_client import GPT4Client
from .medical_prompts import MedicalPrompts
from config_llm import (
    MAX_CONVERSATION_HISTORY, MAX_CONTEXT_LENGTH, CONVERSATION_TIMEOUT,
    CONVERSATION_TYPES, CONVERSATIONS_DIR, LOG_CONVERSATIONS
)

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages clinical trial conversations with context and safety."""
    
    def __init__(self, search_engine=None):
        """
        Initialize conversation manager.
        
        Args:
            search_engine: Optional AdvancedClinicalTrialSearch instance
        """
        self.gpt4_client = GPT4Client()
        self.prompts = MedicalPrompts()
        self.search_engine = search_engine
        
        # Active conversations
        self.conversations: Dict[str, Conversation] = {}
        
        # Load system prompts
        self.system_prompts = self.prompts.get_system_prompts()
        self.response_templates = self.prompts.get_response_templates()
        
        logger.info("ConversationManager initialized")
    
    async def start_conversation(
        self,
        conversation_type: str = "general_inquiry",
        initial_context: Optional[Dict] = None
    ) -> str:
        """
        Start a new conversation.
        
        Args:
            conversation_type: Type of conversation (general_inquiry, trial_explanation, etc.)
            initial_context: Optional context to start with
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        conversation = Conversation(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            config=CONVERSATION_TYPES.get(conversation_type, CONVERSATION_TYPES["general_inquiry"]),
            initial_context=initial_context
        )
        
        self.conversations[conversation_id] = conversation
        
        logger.info(f"Started new conversation: {conversation_id} (type: {conversation_type})")
        return conversation_id
    
    async def process_message(
        self,
        conversation_id: str,
        user_message: str,
        include_search: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
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
            
            # Prepare messages for GPT-4
            system_prompt = self._get_system_prompt(conversation.conversation_type, search_results)
            messages = self.gpt4_client.format_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation.get_context_messages()
            )
            
            # Generate response
            response = await self.gpt4_client.chat_completion(
                messages=messages,
                model=conversation.config.get("model"),
                temperature=conversation.config.get("temperature"),
                max_tokens=conversation.config.get("max_tokens"),
                conversation_id=conversation_id
            )
            
            # Add messages to conversation history
            conversation.add_message("user", user_message)
            conversation.add_message("assistant", response["content"])
            
            # Prepare full response
            full_response = {
                "content": response["content"],
                "conversation_id": conversation_id,
                "search_results": search_results,
                "metadata": {
                    "model": response["model"],
                    "usage": response["usage"],
                    "response_time": response["response_time"],
                    "conversation_type": conversation.conversation_type,
                    "message_count": len(conversation.messages)
                }
            }
            
            # Add cost information if available
            if "estimated_cost" in response:
                full_response["metadata"]["estimated_cost"] = response["estimated_cost"]
            
            # Log conversation if enabled
            if LOG_CONVERSATIONS:
                self._log_conversation_turn(conversation_id, user_message, response["content"], search_results)
            
            logger.info(f"Processed message in conversation {conversation_id}: {response['usage']['total_tokens']} tokens")
            return full_response
            
        except Exception as e:
            logger.error(f"Error processing message in conversation {conversation_id}: {e}")
            # Add error to conversation for context
            conversation.add_message("system", f"Error occurred: {str(e)}")
            raise
    
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
                    "score": result.get("reranked_score", result["score"])
                })
            
            logger.info(f"Found {len(formatted_results)} trials for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching trials: {e}")
            return None
    
    def _get_system_prompt(self, conversation_type: str, search_results: Optional[List[Dict]] = None) -> str:
        """
        Get system prompt for conversation type, optionally including search results.
        
        Args:
            conversation_type: Type of conversation
            search_results: Optional search results to include
            
        Returns:
            Complete system prompt
        """
        base_prompt = self.system_prompts.get(
            CONVERSATION_TYPES[conversation_type]["system_prompt_key"],
            self.system_prompts["general_medical"]
        )
        
        if search_results:
            search_context = self._format_search_context(search_results)
            return f"{base_prompt}\n\nCURRENT SEARCH RESULTS:\n{search_context}"
        
        return base_prompt
    
    def _format_search_context(self, search_results: List[Dict]) -> str:
        """Format search results for inclusion in system prompt."""
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
        
        context += "Use this information to provide relevant, accurate responses about available clinical trials."
        return context
    
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
                "has_search_results": bool(search_results)
            }
            
            # Save to daily log file
            log_file = CONVERSATIONS_DIR / f"conversations_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def get_conversation(self, conversation_id: str) -> Optional["Conversation"]:
        """Get conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def end_conversation(self, conversation_id: str) -> bool:
        """End and clean up a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Ended conversation: {conversation_id}")
            return True
        return False
    
    def cleanup_expired_conversations(self) -> int:
        """Clean up expired conversations. Returns number cleaned up."""
        expired_ids = [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.is_expired()
        ]
        
        for conv_id in expired_ids:
            del self.conversations[conv_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired conversations")
        
        return len(expired_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics."""
        active_conversations = len(self.conversations)
        
        # Get conversation types distribution
        type_distribution = {}
        for conv in self.conversations.values():
            conv_type = conv.conversation_type
            type_distribution[conv_type] = type_distribution.get(conv_type, 0) + 1
        
        return {
            "active_conversations": active_conversations,
            "conversation_types": type_distribution,
            "search_enabled": self.search_engine is not None,
            "gpt4_stats": self.gpt4_client.get_usage_stats()
        }


class Conversation:
    """Individual conversation with context management."""
    
    def __init__(
        self,
        conversation_id: str,
        conversation_type: str,
        config: Dict[str, Any],
        initial_context: Optional[Dict] = None
    ):
        """Initialize conversation."""
        self.conversation_id = conversation_id
        self.conversation_type = conversation_type
        self.config = config
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Message history
        self.messages: List[Dict[str, Any]] = []
        
        # Initial context if provided
        if initial_context:
            self.messages.append({
                "role": "system",
                "content": f"Initial context: {json.dumps(initial_context)}",
                "timestamp": self.created_at.isoformat(),
                "type": "context"
            })
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "message"
        })
        self.last_activity = datetime.now()
        
        # Keep only recent messages to manage context length
        if len(self.messages) > MAX_CONVERSATION_HISTORY * 2:  # *2 for user+assistant pairs
            # Keep system messages and recent exchanges
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            recent_messages = self.messages[-(MAX_CONVERSATION_HISTORY * 2):]
            self.messages = system_messages + recent_messages
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for API context."""
        context_messages = []
        
        for msg in self.messages:
            if msg["type"] == "message" and msg["role"] in ["user", "assistant"]:
                context_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return context_messages
    
    def is_expired(self) -> bool:
        """Check if conversation has expired."""
        return datetime.now() - self.last_activity > timedelta(seconds=CONVERSATION_TIMEOUT)
    
    def get_duration(self) -> timedelta:
        """Get conversation duration."""
        return self.last_activity - self.created_at
    
    def get_message_count(self) -> int:
        """Get number of messages in conversation."""
        return len([msg for msg in self.messages if msg["type"] == "message"])