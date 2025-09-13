"""Simple conversation state persistence for clinical trial chat system."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Data class for conversation state."""
    conversation_id: str
    conversation_type: str
    created_at: str
    last_activity: str
    message_count: int
    config: Dict[str, Any]
    messages: List[Dict[str, Any]]
    initial_context: Optional[Dict] = None

class ConversationStateManager:
    """Manages conversation state persistence using JSON files."""
    
    def __init__(self, state_dir: str = "conversations/state"):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory to store conversation state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ConversationStateManager initialized with state_dir: {self.state_dir}")
    
    def save_conversation(self, conversation) -> bool:
        """
        Save conversation state to JSON file.
        
        Args:
            conversation: Conversation object to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create state object
            state = ConversationState(
                conversation_id=conversation.conversation_id,
                conversation_type=conversation.conversation_type,
                created_at=conversation.created_at.isoformat(),
                last_activity=conversation.last_activity.isoformat(),
                message_count=len(conversation.messages),
                config=conversation.config,
                messages=conversation.messages,
                initial_context=getattr(conversation, 'initial_context', None)
            )
            
            # Save to file
            state_file = self.state_dir / f"{conversation.conversation_id}.json"
            with open(state_file, 'w') as f:
                json.dump(asdict(state), f, indent=2, default=str)
            
            logger.debug(f"Saved conversation state: {conversation.conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.conversation_id}: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation state from JSON file.
        
        Args:
            conversation_id: ID of conversation to load
            
        Returns:
            Conversation state dict or None if not found
        """
        try:
            state_file = self.state_dir / f"{conversation_id}.json"
            
            if not state_file.exists():
                return None
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            logger.debug(f"Loaded conversation state: {conversation_id}")
            return state_data
            
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def list_active_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all saved conversations with summary info.
        
        Args:
            limit: Optional limit on number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        conversations = []
        
        try:
            # Get all state files
            state_files = list(self.state_dir.glob("*.json"))
            
            # Sort by modification time (most recent first)
            state_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            if limit:
                state_files = state_files[:limit]
            
            for state_file in state_files:
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    # Create summary
                    summary = {
                        "conversation_id": state_data["conversation_id"],
                        "conversation_type": state_data["conversation_type"],
                        "created_at": state_data["created_at"],
                        "last_activity": state_data["last_activity"],
                        "message_count": state_data["message_count"],
                        "file_size": state_file.stat().st_size
                    }
                    
                    conversations.append(summary)
                    
                except Exception as e:
                    logger.warning(f"Error reading state file {state_file}: {e}")
                    continue
            
            logger.info(f"Listed {len(conversations)} conversations")
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation state file.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            state_file = self.state_dir / f"{conversation_id}.json"
            
            if state_file.exists():
                state_file.unlink()
                logger.info(f"Deleted conversation state: {conversation_id}")
                return True
            else:
                logger.warning(f"Conversation state file not found: {conversation_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """
        Clean up conversation state files older than specified days.
        
        Args:
            days_old: Delete conversations older than this many days
            
        Returns:
            Number of conversations deleted
        """
        deleted_count = 0
        
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for state_file in self.state_dir.glob("*.json"):
                try:
                    if state_file.stat().st_mtime < cutoff_time:
                        state_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old conversation state: {state_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Error deleting old state file {state_file}: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old conversations")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored conversations.
        
        Returns:
            Dictionary with state manager statistics
        """
        try:
            state_files = list(self.state_dir.glob("*.json"))
            
            total_conversations = len(state_files)
            total_size = sum(f.stat().st_size for f in state_files)
            
            # Get conversation types distribution
            type_distribution = {}
            message_counts = []
            
            for state_file in state_files:
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    conv_type = state_data.get("conversation_type", "unknown")
                    type_distribution[conv_type] = type_distribution.get(conv_type, 0) + 1
                    
                    message_counts.append(state_data.get("message_count", 0))
                    
                except Exception:
                    continue
            
            avg_messages = sum(message_counts) / len(message_counts) if message_counts else 0
            
            return {
                "total_conversations": total_conversations,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "conversation_types": type_distribution,
                "average_messages_per_conversation": round(avg_messages, 1),
                "state_directory": str(self.state_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_conversations": 0,
                "error": str(e)
            }