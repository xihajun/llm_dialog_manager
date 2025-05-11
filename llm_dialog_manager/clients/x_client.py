"""
Client implementation for X.AI (Grok) models
"""
import os
import logging
import openai
from typing import List, Dict, Optional, Union

from ..formatters import OpenAIFormatter
from .base import BaseClient

logger = logging.getLogger(__name__)

class XClient(BaseClient):
    """Client for X.AI (Grok) API"""
    
    def _get_service_name(self) -> str:
        return "x"
    
    def completion(self, messages, max_tokens=1000, temperature=0.5, 
                   top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a completion using X.AI (Grok) API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            json_format: Whether to return JSON
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        try:
            # Get API credentials if not set
            self.get_credentials()
            
            # Format messages for X.AI API using OpenAI formatter
            # (X.AI uses the same message format as OpenAI)
            formatter = OpenAIFormatter()
            _, formatted_messages = formatter.format_messages(messages)
            
            # Set default base URL if not already set
            if not self.base_url:
                self.base_url = "https://api.x.ai/v1"
            
            # Initialize OpenAI client
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Process model name
            model = kwargs.get("model", "grok-3-beta")
            
            # Create base parameters
            params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            # Add optional parameters
            if json_format:
                params["response_format"] = {"type": "json_object"}
            
            # Generate completion using OpenAI client
            response = client.chat.completions.create(**params)
            
            # Release API credentials
            self.release_credentials()
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"X.AI API error: {e}")
            self.report_error()
            raise
