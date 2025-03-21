"""
Message formatter for Anthropic Claude models
"""
import io
import base64
from typing import List, Dict, Union, Optional
from PIL import Image

from .base import BaseMessageFormatter

class AnthropicFormatter(BaseMessageFormatter):
    """Formatter for Anthropic Claude API messages"""
    
    def format_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> tuple:
        """
        Format messages for the Anthropic Claude API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            A tuple containing (system_message, formatted_messages)
            where system_message is extracted as a separate string
        """
        formatted = []
        system_msg = ""
        
        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]["content"]
            messages = messages[1:]
            
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # Combine content blocks into a single message
                combined_content = []
                for block in content:
                    if isinstance(block, str):
                        combined_content.append({"type": "text", "text": block})
                    elif isinstance(block, Image.Image):
                        # For Claude, convert PIL.Image to base64
                        buffered = io.BytesIO()
                        block.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        combined_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        })
                    elif isinstance(block, dict):
                        if block.get("type") == "image_url":
                            combined_content.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": block["image_url"]["url"]
                                }
                            })
                        elif block.get("type") == "image_base64":
                            combined_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block["image_base64"]["media_type"],
                                    "data": block["image_base64"]["data"]
                                }
                            })
                formatted.append({"role": msg["role"], "content": combined_content})
                
        return system_msg, formatted
