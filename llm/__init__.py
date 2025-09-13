"""LLM module for clinical trial conversational AI."""

from .gpt4_client import GPT4Client
from .conversation_manager import ConversationManager
from .medical_prompts import MedicalPrompts

__all__ = ["GPT4Client", "ConversationManager", "MedicalPrompts"]