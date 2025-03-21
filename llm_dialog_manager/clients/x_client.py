"""
Client implementation for X.AI (Grok) models
"""
import os
import logging
import requests
from typing import List, Dict, Optional, Union

from ..formatters import XFormatter
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
            
            # Format messages for X.AI API
            formatter = XFormatter()
            system_message, formatted_messages = formatter.format_messages(messages)
            
            # Construct request payload
            payload = {
                "model": kwargs.get("model", "grok-1"),
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
            
            # Add system message if present
            if system_message:
                payload["system"] = system_message
                
            # Add JSON response format if requested
            if json_format:
                payload["response_format"] = {"type": "json_object"}
            
            # Make API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Release API credentials
            self.release_credentials()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"X.AI API error: {e}")
            self.report_error()
            raise
