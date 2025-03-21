"""
LLM Dialog Manager

A modular framework for building conversational AI applications with
support for multiple LLM providers.
"""

__version__ = "0.1.0"

from .agent import Agent
from .chat_history import ChatHistory
from .key_manager import key_manager

# Import factory functions for easy access
from .clients import get_client
from .formatters import get_formatter

# Setup environment by default
from .utils.environment import load_env_vars
load_env_vars()