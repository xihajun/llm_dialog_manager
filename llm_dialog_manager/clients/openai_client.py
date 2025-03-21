"""
Client implementation for OpenAI models
"""
import os
import logging
import httpx
from typing import List, Dict, Optional, Union

import openai

from ..formatters import OpenAIFormatter
from .base import BaseClient

logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):
    """Client for OpenAI API"""
    
    def _get_service_name(self) -> str:
        return "openai"
    
    def completion(self, messages, max_tokens=1000, temperature=0.5, 
                   top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (not used for OpenAI)
            json_format: Whether to return JSON
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        try:
            # Get API credentials if not set
            self.get_credentials()
            
            # Format messages for OpenAI API
            formatter = OpenAIFormatter()
            _, formatted_messages = formatter.format_messages(messages)
            
            # Get proxy configuration from environment or default to None
            http_proxy = os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("HTTPS_PROXY")
            
            # Create httpx client with proxy settings if needed
            http_options = {}
            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies["http://"] = http_proxy
                if https_proxy:
                    proxies["https://"] = https_proxy
                http_options["proxies"] = proxies
            
            # Create OpenAI client with proper configuration
            client = openai.OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                http_client=httpx.Client(**http_options) if http_options else None
            )
            
            # Generate completion
            response = client.chat.completions.create(
                model=kwargs.get("model", "gpt-4"),
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"} if json_format else None
            )
            
            # Release API credentials
            self.release_credentials()
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            self.report_error()
            raise
