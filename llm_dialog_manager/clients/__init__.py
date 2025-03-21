"""
LLM API client modules for different services.
"""

from .base import BaseClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .x_client import XClient

def get_client(model_name, api_key=None, base_url=None):
    """Factory method to get the appropriate client for a given model.
    
    Args:
        model_name: Name of the model (e.g., "claude-3-opus", "gemini-1.5-pro")
        api_key: Optional API key to use
        base_url: Optional base URL to use
        
    Returns:
        An instance of the appropriate client class
    """
    if "claude" in model_name:
        return AnthropicClient(api_key=api_key, base_url=base_url)
    elif "gemini" in model_name:
        return GeminiClient(api_key=api_key, base_url=base_url)
    elif "grok" in model_name:
        return XClient(api_key=api_key, base_url=base_url)
    else:  # Default to OpenAI client
        return OpenAIClient(api_key=api_key, base_url=base_url)
