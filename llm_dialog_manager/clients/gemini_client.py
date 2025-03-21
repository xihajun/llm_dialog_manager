"""
Client implementation for Google Gemini models
"""
import os
import logging
from typing import List, Dict, Optional, Union

import google.generativeai as genai

from ..formatters import GeminiFormatter
from .base import BaseClient

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):
    """Client for Google Gemini API"""
    
    def _get_service_name(self) -> str:
        return "gemini"
    
    def completion(self, messages, max_tokens=1000, temperature=0.5, 
                   top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a completion using Google Gemini.
        
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
            
            # Configure Google API
            genai.configure(api_key=self.api_key)
            
            # Format messages for Gemini API
            formatter = GeminiFormatter()
            system_message, formatted_messages = formatter.format_messages(messages)
            
            # Create model configuration
            model = genai.GenerativeModel(
                model_name=kwargs.get("model", "gemini-1.5-pro"),
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                },
                system_instruction=system_message
            )
            
            # Generate response
            if json_format:
                response = model.generate_content(
                    formatted_messages,
                    generation_config={"response_mime_type": "application/json"}
                )
            else:
                response = model.generate_content(formatted_messages)
            
            # Release API credentials
            self.release_credentials()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self.report_error()
            raise
