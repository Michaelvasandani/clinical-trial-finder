"""LangChain-powered GPT-4 client for clinical trial conversations."""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

from config import (
    DEFAULT_MODEL, FALLBACK_MODEL, MAX_TOKENS, TEMPERATURE,
    LLM_REQUEST_TIMEOUT as REQUEST_TIMEOUT, LLM_MAX_RETRIES as MAX_RETRIES,
    MODEL_COSTS, MAX_DAILY_COST, TRACK_USAGE
)

load_dotenv()

logger = logging.getLogger(__name__)

class LangChainGPT4Client:
    """LangChain-powered OpenAI GPT-4 client with medical conversation optimizations."""

    def __init__(self):
        """Initialize the LangChain GPT-4 client."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")

        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=self.api_key,
            max_retries=MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT,
            streaming=True  # Enable streaming for better UX
        )

        # Fallback model for simpler tasks
        self.fallback_llm = ChatOpenAI(
            model=FALLBACK_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS // 2,
            openai_api_key=self.api_key,
            max_retries=MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT
        )

        # Usage tracking
        self.usage_tracker = UsageTracker() if TRACK_USAGE else None

        # Test connection
        self._verify_connection()

        logger.info(f"LangChain GPT-4 client initialized with model: {DEFAULT_MODEL}")

    def _verify_connection(self) -> None:
        """Verify OpenAI API connection using LangChain."""
        try:
            test_message = [HumanMessage(content="test")]
            response = self.fallback_llm.invoke(test_message)
            logger.info("LangChain OpenAI API connection verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify OpenAI API connection: {e}")
            raise ConnectionError(f"Cannot connect to OpenAI API: {e}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_fallback: bool = False,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion using LangChain.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            temperature: Override default temperature
            max_tokens: Override default max tokens
            use_fallback: Use fallback model for simpler tasks
            stream: Enable streaming response

        Returns:
            Dictionary with response content and metadata
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(messages, system_prompt)

            # Select model
            model = self.fallback_llm if use_fallback else self.llm

            # Override parameters if provided
            if temperature is not None or max_tokens is not None:
                model_kwargs = {}
                if temperature is not None:
                    model_kwargs['temperature'] = temperature
                if max_tokens is not None:
                    model_kwargs['max_tokens'] = max_tokens

                model = model.bind(**model_kwargs)

            # Track usage and costs
            with get_openai_callback() as callback:
                if stream:
                    # Streaming response
                    response_content = ""
                    async for chunk in model.astream(langchain_messages):
                        if hasattr(chunk, 'content'):
                            response_content += chunk.content

                    response = AIMessage(content=response_content)
                else:
                    # Regular response
                    response = await model.ainvoke(langchain_messages)

                # Track usage
                if self.usage_tracker:
                    self.usage_tracker.track_request(
                        model=model.model_name,
                        tokens_used=callback.total_tokens,
                        cost=callback.total_cost
                    )

                return {
                    "content": response.content,
                    "role": "assistant",
                    "model": model.model_name,
                    "usage": {
                        "prompt_tokens": callback.prompt_tokens,
                        "completion_tokens": callback.completion_tokens,
                        "total_tokens": callback.total_tokens
                    },
                    "cost": callback.total_cost,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error in LangChain chat completion: {e}")
            raise

    def _convert_messages_to_langchain(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """Convert standard message format to LangChain messages."""
        langchain_messages = []

        # Add system prompt if provided
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        # Convert messages
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}")

        return langchain_messages

    async def generate_with_memory(
        self,
        user_input: str,
        memory: ConversationBufferWindowMemory,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LangChain memory management.

        Args:
            user_input: User's input message
            memory: LangChain conversation memory
            system_prompt: Optional system prompt

        Returns:
            Response dictionary with content and metadata
        """
        try:
            # Get conversation history from memory
            history = memory.chat_memory.messages

            # Add system prompt if provided
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            # Add conversation history
            messages.extend(history)

            # Add current user input
            messages.append(HumanMessage(content=user_input))

            # Generate response
            with get_openai_callback() as callback:
                response = await self.llm.ainvoke(messages)

                # Save to memory
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(response.content)

                # Track usage
                if self.usage_tracker:
                    self.usage_tracker.track_request(
                        model=self.llm.model_name,
                        tokens_used=callback.total_tokens,
                        cost=callback.total_cost
                    )

                return {
                    "content": response.content,
                    "role": "assistant",
                    "model": self.llm.model_name,
                    "usage": {
                        "prompt_tokens": callback.prompt_tokens,
                        "completion_tokens": callback.completion_tokens,
                        "total_tokens": callback.total_tokens
                    },
                    "cost": callback.total_cost,
                    "timestamp": datetime.now().isoformat(),
                    "memory_length": len(memory.chat_memory.messages)
                }

        except Exception as e:
            logger.error(f"Error in memory-based generation: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models."""
        return {
            "primary_model": self.llm.model_name,
            "fallback_model": self.fallback_llm.model_name,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "supports_streaming": True,
            "supports_memory": True
        }

    def get_usage_stats(self) -> Optional[Dict[str, Any]]:
        """Get usage statistics if tracking is enabled."""
        if self.usage_tracker:
            return self.usage_tracker.get_stats()
        return None


class UsageTracker:
    """Track OpenAI API usage and costs."""

    def __init__(self):
        """Initialize usage tracker."""
        self.daily_usage = {}
        self.total_usage = {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0
        }
        self.start_time = datetime.now()

    def track_request(self, model: str, tokens_used: int, cost: float):
        """Track a single API request."""
        today = datetime.now().date().isoformat()

        # Initialize daily tracking
        if today not in self.daily_usage:
            self.daily_usage[today] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
                "models": {}
            }

        # Update daily stats
        self.daily_usage[today]["requests"] += 1
        self.daily_usage[today]["tokens"] += tokens_used
        self.daily_usage[today]["cost"] += cost

        # Track by model
        if model not in self.daily_usage[today]["models"]:
            self.daily_usage[today]["models"][model] = {"requests": 0, "tokens": 0, "cost": 0.0}

        self.daily_usage[today]["models"][model]["requests"] += 1
        self.daily_usage[today]["models"][model]["tokens"] += tokens_used
        self.daily_usage[today]["models"][model]["cost"] += cost

        # Update total stats
        self.total_usage["requests"] += 1
        self.total_usage["tokens"] += tokens_used
        self.total_usage["cost"] += cost

        # Check daily cost limit
        if self.daily_usage[today]["cost"] > MAX_DAILY_COST:
            logger.warning(f"Daily cost limit exceeded: ${self.daily_usage[today]['cost']:.2f}")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        today = datetime.now().date().isoformat()
        today_usage = self.daily_usage.get(today, {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "models": {}
        })

        return {
            "today": today_usage,
            "total": self.total_usage,
            "tracking_since": self.start_time.isoformat(),
            "daily_limit": MAX_DAILY_COST
        }