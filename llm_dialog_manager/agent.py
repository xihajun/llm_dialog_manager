"""
Agent class for managing LLM conversations
"""
# Standard library imports
import uuid
import logging
from typing import List, Dict, Optional, Union
from PIL import Image

# Local imports
from .chat_history import ChatHistory
from .clients import get_client
from .utils.environment import load_env_vars
from .utils.image_tools import load_image_from_path, load_image_from_url, create_image_content_block

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_env_vars()

class Agent:
    """
    Agent class for managing conversations with LLMs.
    
    This class provides a high-level interface for interacting with different
    LLM providers through a unified API.
    """
    
    def __init__(self, model_name: str, 
                 messages: Optional[Union[str, List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]]] = None, 
                 memory_enabled: bool = False, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        """
        Initialize an Agent instance.
        
        Args:
            model_name: Name of the LLM model to use
            messages: Optional initial messages or system prompt
            memory_enabled: Whether to enable conversation memory
            api_key: Optional API key to use
            base_url: Optional base URL for API requests
        """
        self.id = f"{model_name}-{uuid.uuid4().hex[:8]}"
        self.model_name = model_name
        self.history = ChatHistory(messages) if messages else ChatHistory()
        self.memory_enabled = memory_enabled
        self.client = get_client(model_name, api_key=api_key, base_url=base_url)
        self.repo_content = []
    
    def add_message(self, role: str, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """
        Add a message to the conversation.
        
        Args:
            role: Message role ('system', 'user', or 'assistant')
            content: Message content (text, image, or mixed content)
        """
        self.history.add_message(content, role)
    
    def add_user_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """
        Add a user message to the conversation.
        
        Args:
            content: Message content (text, image, or mixed content)
        """
        self.history.add_user_message(content)
    
    def add_assistant_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Message content (text, image, or mixed content)
        """
        self.history.add_assistant_message(content)
    
    def add_image(self, image_path: Optional[str] = None, 
                 image_url: Optional[str] = None, 
                 media_type: Optional[str] = "image/jpeg"):
        """
        Add an image to the conversation.
        
        Either image_path or image_url must be provided.
        
        Args:
            image_path: Path to a local image file
            image_url: URL of an image
            media_type: MIME type of the image
        
        Returns:
            The image content block that was added
        """
        if not (image_path or image_url):
            raise ValueError("Either image_path or image_url must be provided.")
        
        if image_path:
            image = load_image_from_path(image_path)
        else:
            image = load_image_from_url(image_url)
        
        return create_image_content_block(image, media_type)
    
    def generate_response(self, max_tokens=3585, temperature=0.7, 
                         top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a response from the agent.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            json_format: Whether to enable JSON output format
            **kwargs: Additional model-specific parameters
        
        Returns:
            The generated response text
        """
        response = self.client.completion(
            messages=self.history.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            json_format=json_format,
            model=self.model_name,
            **kwargs
        )
        
        # Add the response to history
        if not json_format:
            self.add_assistant_message(response)
        
        return response
    
    def save_conversation(self, filename=None):
        """
        Save the conversation history to a file.
        
        Args:
            filename: Optional filename to save to
        """
        if filename is None:
            filename = f"conversation_{self.id}.json"
        
        import json
        
        # Convert any PIL.Image objects to base64 for serialization
        serializable_history = []
        for msg in self.history.messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                serializable_history.append({"role": role, "content": content})
            elif isinstance(content, list):
                serializable_content = []
                for item in content:
                    if isinstance(item, str):
                        serializable_content.append(item)
                    elif isinstance(item, Image.Image):
                        serializable_content.append(create_image_content_block(item))
                    elif isinstance(item, dict):
                        serializable_content.append(item)
                serializable_history.append({"role": role, "content": serializable_content})
        
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        return filename
    
    def load_conversation(self, filename):
        """
        Load a conversation from a file.
        
        Args:
            filename: Path to the conversation file
        """
        import json
        
        with open(filename, 'r') as f:
            history = json.load(f)
        
        self.history = ChatHistory(history)
        
        return self.history
