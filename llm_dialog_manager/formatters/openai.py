"""
Message formatter for OpenAI models
"""
import io
import base64
from typing import List, Dict, Union, Optional
from PIL import Image

from .base import BaseMessageFormatter

class OpenAIFormatter(BaseMessageFormatter):
    """Formatter for OpenAI API messages"""
    
    def format_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> tuple:
        """
        Format messages for the OpenAI API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            A tuple containing (None, formatted_messages)
            since OpenAI handles system messages in the message list
        """
        formatted = []
        
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # For OpenAI with multimodal models like GPT-4V
                formatted_content = []
                
                for block in content:
                    if isinstance(block, str):
                        formatted_content.append({"type": "text", "text": block})
                    elif isinstance(block, Image.Image):
                        # Convert PIL.Image to base64
                        buffered = io.BytesIO()
                        block.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        })
                    elif isinstance(block, dict):
                        if block.get("type") == "image_url":
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": block["image_url"]["url"]
                                }
                            })
                        elif block.get("type") == "image_base64":
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block['image_base64']['media_type']};base64,{block['image_base64']['data']}"
                                }
                            })
                
                formatted.append({"role": msg["role"], "content": formatted_content})
                
        return None, formatted
