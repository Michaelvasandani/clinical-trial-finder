"""GPT-4 client wrapper for clinical trial conversations."""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import openai
from openai import OpenAI
from dotenv import load_dotenv

from config import (
    DEFAULT_MODEL, FALLBACK_MODEL, MAX_TOKENS, TEMPERATURE, 
    LLM_REQUEST_TIMEOUT as REQUEST_TIMEOUT, LLM_MAX_RETRIES as MAX_RETRIES, 
    LLM_RETRY_DELAY as RETRY_DELAY, MODEL_COSTS, MAX_DAILY_COST, TRACK_USAGE
)

load_dotenv()

logger = logging.getLogger(__name__)

class GPT4Client:
    """OpenAI GPT-4 client with medical conversation optimizations."""
    
    def __init__(self):
        """Initialize the GPT-4 client."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
            timeout=REQUEST_TIMEOUT
        )
        
        self.usage_tracker = UsageTracker() if TRACK_USAGE else None
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """Verify OpenAI API connection."""
        try:
            # Test with a minimal request
            response = self.client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("OpenAI API connection verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify OpenAI API connection: {e}")
            raise ConnectionError(f"Cannot connect to OpenAI API: {e}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion with error handling and usage tracking.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to DEFAULT_MODEL)
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            conversation_id: ID for usage tracking
            
        Returns:
            Dictionary with response content, usage info, and metadata
        """
        model = model or DEFAULT_MODEL
        temperature = temperature or TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS
        
        # Check daily cost limit
        if self.usage_tracker and not self.usage_tracker.can_make_request():
            raise Exception("Daily cost limit exceeded")
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                response_time = time.time() - start_time
                
                # Extract response data
                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": conversation_id
                }
                
                # Track usage and costs
                if self.usage_tracker:
                    cost = self._calculate_cost(result["usage"], model)
                    self.usage_tracker.record_usage(
                        model=model,
                        tokens=result["usage"],
                        cost=cost,
                        conversation_id=conversation_id
                    )
                    result["estimated_cost"] = cost
                
                logger.info(f"Chat completion successful: {result['usage']['total_tokens']} tokens in {response_time:.2f}s")
                return result
                
            except openai.RateLimitError as e:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"OpenAI API error after {MAX_RETRIES} attempts: {e}")
                    raise
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Unexpected error in chat completion: {e}")
                    raise
                time.sleep(RETRY_DELAY)
        
        raise Exception(f"Failed to get response after {MAX_RETRIES} attempts")
    
    def _calculate_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate estimated cost for API usage."""
        if model not in MODEL_COSTS:
            logger.warning(f"Unknown model cost: {model}")
            return 0.0
        
        costs = MODEL_COSTS[model]
        input_cost = (usage["prompt_tokens"] / 1000) * costs["input"]
        output_cost = (usage["completion_tokens"] / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def format_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Format messages for API request.
        
        Args:
            system_prompt: System prompt defining AI behavior
            user_message: Current user message
            conversation_history: Previous messages in conversation
            
        Returns:
            Formatted message list for API
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def get_usage_stats(self) -> Optional[Dict[str, Any]]:
        """Get current usage statistics."""
        if not self.usage_tracker:
            return None
        
        return self.usage_tracker.get_stats()


class UsageTracker:
    """Track API usage and costs."""
    
    def __init__(self):
        """Initialize usage tracker."""
        self.daily_usage = {}
        self.daily_costs = {}
        self.current_date = datetime.now().date()
    
    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if date changed."""
        today = datetime.now().date()
        if today != self.current_date:
            self.daily_usage = {}
            self.daily_costs = {}
            self.current_date = today
    
    def record_usage(
        self,
        model: str,
        tokens: Dict[str, int],
        cost: float,
        conversation_id: Optional[str] = None
    ) -> None:
        """Record API usage."""
        self._reset_daily_if_needed()
        
        date_key = self.current_date.isoformat()
        
        # Track tokens
        if date_key not in self.daily_usage:
            self.daily_usage[date_key] = {"models": {}, "total_tokens": 0}
        
        if model not in self.daily_usage[date_key]["models"]:
            self.daily_usage[date_key]["models"][model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "requests": 0
            }
        
        model_usage = self.daily_usage[date_key]["models"][model]
        model_usage["prompt_tokens"] += tokens["prompt_tokens"]
        model_usage["completion_tokens"] += tokens["completion_tokens"]
        model_usage["total_tokens"] += tokens["total_tokens"]
        model_usage["requests"] += 1
        
        self.daily_usage[date_key]["total_tokens"] += tokens["total_tokens"]
        
        # Track costs
        if date_key not in self.daily_costs:
            self.daily_costs[date_key] = 0.0
        
        self.daily_costs[date_key] += cost
        
        logger.debug(f"Usage recorded: {model} - {tokens['total_tokens']} tokens, ${cost:.4f}")
    
    def can_make_request(self) -> bool:
        """Check if request can be made within cost limits."""
        self._reset_daily_if_needed()
        
        today = self.current_date.isoformat()
        daily_cost = self.daily_costs.get(today, 0.0)
        
        return daily_cost < MAX_DAILY_COST
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        self._reset_daily_if_needed()
        
        today = self.current_date.isoformat()
        
        return {
            "date": today,
            "daily_cost": self.daily_costs.get(today, 0.0),
            "daily_limit": MAX_DAILY_COST,
            "remaining_budget": MAX_DAILY_COST - self.daily_costs.get(today, 0.0),
            "total_tokens": self.daily_usage.get(today, {}).get("total_tokens", 0),
            "models": self.daily_usage.get(today, {}).get("models", {}),
            "requests_made": sum(
                model.get("requests", 0) 
                for model in self.daily_usage.get(today, {}).get("models", {}).values()
            )
        }