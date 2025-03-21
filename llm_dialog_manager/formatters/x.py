"""
Message formatter for X.AI (Grok) models
"""
from typing import List, Dict, Union, Optional
from PIL import Image
import io
import base64

from .base import BaseMessageFormatter

class XFormatter(BaseMessageFormatter):
    """Formatter for X.AI (Grok) API messages"""
    
    def format_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> tuple:
        """
        Format messages for the X.AI (Grok) API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            A tuple containing (system_message, formatted_messages)
        """
        system_msg = None
        formatted = []
        
        for msg in messages:
            # Extract system message if present, similar to many other APIs
            if msg["role"] == "system":
                system_msg = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                continue
                
            content = msg["content"]
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # Grok API format may need adjustments as it evolves
                combined_content = []
                
                for block in content:
                    if isinstance(block, str):
                        combined_content.append({"type": "text", "text": block})
                    elif isinstance(block, Image.Image):
                        # Convert PIL.Image to base64
                        buffered = io.BytesIO()
                        block.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        combined_content.append({
                            "type": "image",
                            "image": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        })
                    elif isinstance(block, dict):
                        if block.get("type") == "image_url":
                            combined_content.append({
                                "type": "image",
                                "image": {
                                    "type": "url",
                                    "url": block["image_url"]["url"]
                                }
                            })
                        elif block.get("type") == "image_base64":
                            combined_content.append({
                                "type": "image",
                                "image": {
                                    "type": "base64",
                                    "media_type": block["image_base64"]["media_type"],
                                    "data": block["image_base64"]["data"]
                                }
                            })
                
                formatted.append({"role": msg["role"], "content": combined_content})
                
        return system_msg, formatted
