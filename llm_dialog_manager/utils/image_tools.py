"""
Image handling utilities for LLM dialog manager
"""
import base64
import io
from typing import Optional
import requests
from PIL import Image

def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load an image from a local file path.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        PIL.Image object
    """
    try:
        image = Image.open(image_path)
        # Store the filename for reference
        image.filename = image_path
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")

def load_image_from_url(image_url: str) -> Image.Image:
    """
    Load an image from a URL.
    
    Args:
        image_url: URL to the image
        
    Returns:
        PIL.Image object
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        # Store the URL for reference
        image.filename = image_url
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_url}: {e}")

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL.Image to base64 string.
    
    Args:
        image: PIL.Image object
        format: Image format (default: PNG)
        
    Returns:
        Base64-encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_image_content_block(image: Image.Image, media_type: str = "image/png") -> dict:
    """
    Create a standard image content block for messages.
    
    Args:
        image: PIL.Image object
        media_type: MIME type of the image
        
    Returns:
        Dictionary with image information
    """
    image_base64 = encode_image_to_base64(image)
    return {
        "type": "image_base64",
        "image_base64": {
            "media_type": media_type,
            "data": image_base64
        }
    }
