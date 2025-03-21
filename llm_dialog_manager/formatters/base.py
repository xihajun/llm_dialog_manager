"""
Base message formatter interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from PIL import Image

class BaseMessageFormatter(ABC):
    """Base class for message formatters"""
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> tuple:
        """
        Format messages for the specific LLM API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            A tuple containing (system_message, formatted_messages)
            where system_message can be None if not used by the API
        """
        pass
