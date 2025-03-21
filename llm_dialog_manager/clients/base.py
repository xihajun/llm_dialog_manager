"""
Base client interface for LLM APIs
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import logging

from ..key_manager import key_manager

logger = logging.getLogger(__name__)

class BaseClient(ABC):
    """Base class for LLM API clients"""
    
    def __init__(self, api_key=None, base_url=None):
        """
        Initialize client with optional API key and base URL.
        
        Args:
            api_key: Optional API key to use 
            base_url: Optional base URL for API requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.service_name = self._get_service_name()
    
    @abstractmethod
    def _get_service_name(self) -> str:
        """Return the service name for this client (e.g., 'openai', 'anthropic')"""
        pass
    
    def get_credentials(self):
        """Get API credentials from key manager if not set"""
        if not self.api_key:
            self.api_key, self.base_url = key_manager.get_config(self.service_name)
    
    def release_credentials(self):
        """Release API credentials in key manager"""
        if self.api_key:
            key_manager.release_config(self.service_name, self.api_key)
    
    def report_error(self):
        """Report API error to key manager"""
        if self.api_key:
            key_manager.report_error(self.service_name, self.api_key)
    
    @abstractmethod
    def completion(self, messages, max_tokens=1000, temperature=0.5, 
                   top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a completion for the given messages.
        
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
        pass
