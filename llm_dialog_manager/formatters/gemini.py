"""
Message formatter for Google Gemini models
"""
from typing import List, Dict, Union, Optional
from PIL import Image

from .base import BaseMessageFormatter

class GeminiFormatter(BaseMessageFormatter):
    """Formatter for Google Gemini API messages"""
    
    def format_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> tuple:
        """
        Format messages for the Google Gemini API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            A tuple containing (system_message, formatted_messages)
            where system_message is extracted separately
        """
        system_msg = None
        formatted = []
        
        for msg in messages:
            # Extract system message if present
            if msg["role"] == "system":
                system_msg = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                continue
                
            content = msg["content"]
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "parts": [content]})
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, Image.Image):
                        parts.append(block)  # Gemini supports PIL.Image directly
                    elif isinstance(block, dict):
                        if block.get("type") == "image_url":
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": block["image_url"]["url"]
                                }
                            })
                        elif block.get("type") == "image_base64":
                            parts.append({
                                "inline_data": {
                                    "mime_type": block["image_base64"]["media_type"],
                                    "data": block["image_base64"]["data"]
                                }
                            })
                formatted.append({"role": msg["role"], "parts": parts})
                
        return system_msg, formatted
