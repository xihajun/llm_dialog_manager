"""
Message formatters for different LLM services.
"""

from .base import BaseMessageFormatter
from .anthropic import AnthropicFormatter
from .gemini import GeminiFormatter
from .openai import OpenAIFormatter
from .x import XFormatter

def get_formatter(model_name):
    """Factory method to get the appropriate formatter for a given model.
    
    Args:
        model_name: Name of the model (e.g., "claude-3-opus", "gemini-1.5-pro")
        
    Returns:
        An instance of the appropriate formatter class
    """
    if "claude" in model_name:
        return AnthropicFormatter()
    elif "gemini" in model_name:
        return GeminiFormatter()
    elif "grok" in model_name:
        return XFormatter()
    else:  # Default to OpenAI formatter
        return OpenAIFormatter()
