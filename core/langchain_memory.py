"""LangChain-powered conversation memory for clinical trial conversations."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    MAX_CONVERSATION_HISTORY, CONVERSATIONS_DIR,
    DEFAULT_MODEL, TEMPERATURE
)

logger = logging.getLogger(__name__)

class PersistentChatMessageHistory(BaseChatMessageHistory):
    """Persistent chat message history that saves to disk."""

    def __init__(self, conversation_id: str):
        """Initialize with conversation ID for file-based persistence."""
        self.conversation_id = conversation_id
        self.state_dir = Path(CONVERSATIONS_DIR) / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.state_dir / f"{conversation_id}_messages.json"

        # Load existing messages
        self._messages: List[BaseMessage] = []
        self._load_messages()

    def _load_messages(self) -> None:
        """Load messages from persistent storage."""
        try:
            if self.file_path.exists():
                with open(self.file_path, 'r') as f:
                    data = json.load(f)

                self._messages = []
                for msg_data in data.get("messages", []):
                    if msg_data["type"] == "human":
                        self._messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "ai":
                        self._messages.append(AIMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "system":
                        self._messages.append(SystemMessage(content=msg_data["content"]))

                logger.debug(f"Loaded {len(self._messages)} messages for conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error loading messages for conversation {self.conversation_id}: {e}")

    def _save_messages(self) -> None:
        """Save messages to persistent storage."""
        try:
            data = {
                "conversation_id": self.conversation_id,
                "updated_at": datetime.now().isoformat(),
                "messages": []
            }

            for msg in self._messages:
                if isinstance(msg, HumanMessage):
                    msg_type = "human"
                elif isinstance(msg, AIMessage):
                    msg_type = "ai"
                elif isinstance(msg, SystemMessage):
                    msg_type = "system"
                else:
                    msg_type = "unknown"

                data["messages"].append({
                    "type": msg_type,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })

            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving messages for conversation {self.conversation_id}: {e}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the list of messages."""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store and save to disk."""
        self._messages.append(message)
        self._save_messages()

    def clear(self) -> None:
        """Clear all messages and remove from disk."""
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()

class LangChainMemoryManager:
    """Enhanced memory management using LangChain memory systems."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the memory manager."""
        self.llm = llm or ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE
        )

        # Active memory instances
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.summary_memories: Dict[str, ConversationSummaryBufferMemory] = {}

        logger.info("LangChain memory manager initialized")

    def get_buffer_memory(
        self,
        conversation_id: str,
        k: int = MAX_CONVERSATION_HISTORY,
        return_messages: bool = True
    ) -> ConversationBufferWindowMemory:
        """Get or create a buffer window memory for a conversation."""

        if conversation_id not in self.memories:
            # Create persistent chat message history
            chat_history = PersistentChatMessageHistory(conversation_id)

            # Create buffer window memory
            memory = ConversationBufferWindowMemory(
                k=k,
                chat_memory=chat_history,
                return_messages=return_messages,
                memory_key="chat_history",
                input_key="input",
                output_key="output"
            )

            self.memories[conversation_id] = memory
            logger.debug(f"Created buffer memory for conversation {conversation_id}")

        return self.memories[conversation_id]

    def get_summary_memory(
        self,
        conversation_id: str,
        max_token_limit: int = 2000,
        return_messages: bool = True
    ) -> ConversationSummaryBufferMemory:
        """Get or create a summary buffer memory for a conversation."""

        if conversation_id not in self.summary_memories:
            # Create persistent chat message history
            chat_history = PersistentChatMessageHistory(conversation_id)

            # Create summary buffer memory
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=chat_history,
                max_token_limit=max_token_limit,
                return_messages=return_messages,
                memory_key="chat_history",
                input_key="input",
                output_key="output"
            )

            self.summary_memories[conversation_id] = memory
            logger.debug(f"Created summary memory for conversation {conversation_id}")

        return self.summary_memories[conversation_id]

    def add_exchange(
        self,
        conversation_id: str,
        user_input: str,
        ai_output: str,
        use_summary: bool = False
    ) -> None:
        """Add a user-AI exchange to memory."""

        if use_summary:
            memory = self.get_summary_memory(conversation_id)
        else:
            memory = self.get_buffer_memory(conversation_id)

        # Add the exchange
        memory.save_context(
            inputs={"input": user_input},
            outputs={"output": ai_output}
        )

        logger.debug(f"Added exchange to {'summary' if use_summary else 'buffer'} memory for {conversation_id}")

    def get_memory_variables(
        self,
        conversation_id: str,
        use_summary: bool = False
    ) -> Dict[str, Any]:
        """Get memory variables for use in prompts."""

        if use_summary:
            memory = self.get_summary_memory(conversation_id)
        else:
            memory = self.get_buffer_memory(conversation_id)

        return memory.load_memory_variables({})

    def get_conversation_history(
        self,
        conversation_id: str,
        as_messages: bool = True
    ) -> List[BaseMessage]:
        """Get conversation history as LangChain messages."""

        memory = self.get_buffer_memory(conversation_id, return_messages=True)
        chat_history = memory.chat_memory

        return chat_history.messages

    def clear_memory(self, conversation_id: str, clear_summary: bool = True) -> None:
        """Clear memory for a conversation."""

        if conversation_id in self.memories:
            self.memories[conversation_id].clear()
            del self.memories[conversation_id]

        if clear_summary and conversation_id in self.summary_memories:
            self.summary_memories[conversation_id].clear()
            del self.summary_memories[conversation_id]

        logger.info(f"Cleared memory for conversation {conversation_id}")

    def get_memory_summary(self, conversation_id: str) -> Optional[str]:
        """Get a summary of the conversation using summary memory."""

        if conversation_id not in self.summary_memories:
            return None

        try:
            summary_memory = self.summary_memories[conversation_id]

            # Get the current summary from the memory
            memory_vars = summary_memory.load_memory_variables({})

            # Extract summary if available
            history = memory_vars.get("chat_history", "")
            if isinstance(history, str) and history.strip():
                return history
            elif isinstance(history, list) and history:
                # If it's a list of messages, create a simple summary
                return f"Conversation with {len(history)} exchanges covering clinical trial topics."

            return None

        except Exception as e:
            logger.error(f"Error getting memory summary for {conversation_id}: {e}")
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""

        total_buffer_memories = len(self.memories)
        total_summary_memories = len(self.summary_memories)

        # Calculate total messages across all memories
        total_messages = 0
        for memory in self.memories.values():
            total_messages += len(memory.chat_memory.messages)

        # Get memory sizes
        memory_sizes = {}
        for conv_id, memory in self.memories.items():
            memory_sizes[conv_id] = {
                "message_count": len(memory.chat_memory.messages),
                "type": "buffer"
            }

        for conv_id, memory in self.summary_memories.items():
            if conv_id not in memory_sizes:
                memory_sizes[conv_id] = {}
            memory_sizes[conv_id].update({
                "summary_message_count": len(memory.chat_memory.messages),
                "has_summary": True
            })

        return {
            "total_buffer_memories": total_buffer_memories,
            "total_summary_memories": total_summary_memories,
            "total_messages": total_messages,
            "memory_sizes": memory_sizes,
            "persistence_enabled": True
        }

    def migrate_from_legacy_conversation(
        self,
        conversation_id: str,
        legacy_messages: List[Dict[str, str]]
    ) -> None:
        """Migrate messages from legacy conversation format."""

        try:
            memory = self.get_buffer_memory(conversation_id)
            chat_history = memory.chat_memory

            # Clear existing messages
            chat_history.clear()

            # Convert and add legacy messages
            for msg in legacy_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    chat_history.add_message(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.add_message(AIMessage(content=content))
                elif role == "system":
                    chat_history.add_message(SystemMessage(content=content))

            logger.info(f"Migrated {len(legacy_messages)} messages for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Error migrating legacy conversation {conversation_id}: {e}")

# Global instance
langchain_memory_manager = LangChainMemoryManager()