# Standard library imports
import json
import os
import uuid
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import random
import requests
import zipfile
import io
import base64
from PIL import Image

# Third-party imports
import anthropic
from anthropic import AnthropicVertex
import google.generativeai as genai
import openai
from dotenv import load_dotenv

# Local imports
from .chat_history import ChatHistory
from .key_manager import key_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(".env file not found. Using system environment variables.")

load_env_vars()

def completion(model: str, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]], max_tokens: int = 1000, 
              temperature: float = 0.5, api_key: Optional[str] = None, 
              base_url: Optional[str] = None, json_format: bool = False) -> str:
    """
    Generate a completion using the specified model and messages.
    """
    try:
        service = ""
        if "claude" in model:
            service = "anthropic"
        elif "gemini" in model:
            service = "gemini"
        elif "grok" in model:
            service = "x"
        else:
            service = "openai"

        # Get API key and base URL from key manager if not provided
        if not api_key:
            # api_key, base_url = key_manager.get_config(service)
            # Placeholder for key_manager
            api_key = os.getenv(f"{service.upper()}_API_KEY")
            base_url = os.getenv(f"{service.upper()}_BASE_URL")

        def format_messages_for_api(model, messages):
            """Convert ChatHistory messages to the format required by the specific API."""
            if "claude" in model:
                formatted = []
                system_msg = ""
                if messages and messages[0]["role"] == "system":
                    system_msg = messages.pop(0)["content"]
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

            elif "gemini" in model or "gpt" in model or "grok" in model:
                formatted = []
                for msg in messages:
                    content = msg["content"]
                    if isinstance(content, str):
                        formatted.append({"role": msg["role"], "parts": [content]})
                    elif isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, str):
                                parts.append(block)
                            elif isinstance(block, Image.Image):
                                parts.append(block)
                            elif isinstance(block, dict):
                                if block.get("type") == "image_url":
                                    parts.append({"type": "image_url", "image_url": {"url": block["image_url"]["url"]}})
                                elif block.get("type") == "image_base64":
                                    parts.append({"type": "image_base64", "image_base64": {"data": block["image_base64"]["data"], "media_type": block["image_base64"]["media_type"]}})
                        formatted.append({"role": msg["role"], "parts": parts})
                return None, formatted

            else:  # OpenAI models
                formatted = []
                for msg in messages:
                    content = msg["content"]
                    if isinstance(content, str):
                        formatted.append({"role": msg["role"], "content": content})
                    elif isinstance(content, list):
                        # OpenAI expects 'content' as string; images are not directly supported
                        # You can convert images to URLs or descriptions if needed
                        combined_content = ""
                        for block in content:
                            if isinstance(block, str):
                                combined_content += block + "\n"
                            elif isinstance(block, Image.Image):
                                # Convert PIL.Image to base64 or upload and use URL
                                buffered = io.BytesIO()
                                block.save(buffered, format="PNG")
                                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                combined_content += f"[Image Base64: {image_base64[:30]}...]\n"
                            elif isinstance(block, dict):
                                if block.get("type") == "image_url":
                                    combined_content += f"[Image: {block['image_url']['url']}]\n"
                                elif block.get("type") == "image_base64":
                                    combined_content += f"[Image Base64: {block['image_base64']['data'][:30]}...]\n"
                        formatted.append({"role": msg["role"], "content": combined_content.strip()})
                return None, formatted

        system_msg, formatted_messages = format_messages_for_api(model, messages.copy())

        if "claude" in model:
            # Check for Vertex configuration
            vertex_project_id = os.getenv('VERTEX_PROJECT_ID')
            vertex_region = os.getenv('VERTEX_REGION')

            if vertex_project_id and vertex_region:
                client = AnthropicVertex(
                    region=vertex_region,
                    project_id=vertex_project_id
                )
            else:
                client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=formatted_messages,
                system=system_msg
            )

            while response.stop_reason == "max_tokens":
                if formatted_messages[-1]['role'] == "user":
                    formatted_messages.append({"role": "assistant", "content": response.completion})
                else:
                    formatted_messages[-1]['content'] += response.completion

                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=formatted_messages,
                    system=system_msg
                )

            if formatted_messages[-1]['role'] == "assistant" and response.stop_reason == "end_turn":
                formatted_messages[-1]['content'] += response.completion
                return formatted_messages[-1]['content']

            return response.completion

        elif "gemini" in model:
            try:
                # First try OpenAI-style API
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/"
                )
                # Set response_format based on json_format
                response_format = {"type": "json_object"} if json_format else {"type": "plain_text"}

                response = client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    temperature=temperature,
                    response_format=response_format  # Added response_format
                )
                return response.choices[0].message.content

            except Exception as e:
                # If OpenAI-style API fails, fall back to Google's genai library
                logger.info("Falling back to Google's genai library")
                genai.configure(api_key=api_key)

                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        # Prepend system message to first user message if exists
                        if gemini_messages:
                            first_msg = gemini_messages[0]
                            if "parts" in first_msg and len(first_msg["parts"]) > 0:
                                first_msg["parts"][0] = f"{msg['content']}\n\n{first_msg['parts'][0]}"
                    else:
                        gemini_messages.append({"role": msg["role"], "parts": msg["content"]})

                # Set response_mime_type based on json_format
                mime_type = "application/json" if json_format else "text/plain"

                # Create Gemini model and generate response
                model_instance = genai.GenerativeModel(model_name=model)
                response = model_instance.generate_content(
                    gemini_messages,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        response_mime_type=mime_type,  # Modified based on json_format
                        max_output_tokens=max_tokens
                    )
                )

                return response.text

        elif "grok" in model:
            # Randomly choose between OpenAI and Anthropic SDK
            use_anthropic = random.choice([True, False])

            if use_anthropic:
                logger.info("Using Anthropic for Grok model")
                client = anthropic.Anthropic(
                    api_key=api_key,
                    base_url="https://api.x.ai"
                )

                system_msg = ""
                if messages and messages[0]["role"] == "system":
                    system_msg = messages.pop(0)["content"]

                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=formatted_messages,
                    system=system_msg
                )
                return response.completion
            else:
                logger.info("Using OpenAI for Grok model")
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1"
                )
                # Set response_format based on json_format
                response_format = {"type": "json_object"} if json_format else {"type": "plain_text"}

                response = client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format  # Added response_format
                )
                return response.choices[0].message.content

        else:  # OpenAI models
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            # Set response_format based on json_format
            response_format = {"type": "json_object"} if json_format else {"type": "plain_text"}

            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format  # Added response_format
            )
            return response.choices[0].message.content

        # Release the API key after successful use
        if not api_key:
            # key_manager.release_config(service, api_key)
            pass

        return response

    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise

class Agent:
    def __init__(self, model_name: str, messages: Optional[Union[str, List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]]] = None, 
                 memory_enabled: bool = False, api_key: Optional[str] = None) -> None:
        """Initialize an Agent instance."""
        self.id = f"{model_name}-{uuid.uuid4().hex[:8]}"
        self.model_name = model_name
        self.history = ChatHistory(messages) if messages else ChatHistory()
        self.memory_enabled = memory_enabled
        self.api_key = api_key
        self.repo_content = []

    def add_message(self, role: str, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """Add a message to the conversation."""
        self.history.add_message(content, role)

    def add_user_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """Add a user message."""
        self.history.add_user_message(content)

    def add_assistant_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        """Add an assistant message."""
        self.history.add_assistant_message(content)

    def add_image(self, image_path: Optional[str] = None, image_url: Optional[str] = None, media_type: Optional[str] = "image/jpeg"):
        """
        Add an image to the conversation.
        Either image_path or image_url must be provided.
        """
        if not image_path and not image_url:
            raise ValueError("Either image_path or image_url must be provided.")

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file {image_path} does not exist.")
            if "gemini" in self.model_name:
                # For Gemini, load as PIL.Image
                image_pil = Image.open(image_path)
                image_block = image_pil
            else:
                # For Claude and others, use base64 encoding
                with open(image_path, "rb") as img_file:
                    image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
                image_block = {
                    "type": "image_base64",
                    "image_base64": {
                        "media_type": media_type,
                        "data": image_data
                    }
                }
        else:
            # If image_url is provided
            if "gemini" in self.model_name:
                # For Gemini, you can pass image URLs directly
                image_block = {"type": "image_url", "image_url": {"url": image_url}}
            else:
                # For Claude and others, use image URLs
                image_block = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }

        # Add the image block to the last user message or as a new user message
        if self.history.last_role == "user":
            current_content = self.history.messages[-1]["content"]
            if isinstance(current_content, list):
                current_content.append(image_block)
            else:
                self.history.messages[-1]["content"] = [current_content, image_block]
        else:
            # Start a new user message with the image
            self.history.add_message([image_block], "user")

    def generate_response(self, max_tokens=3585, temperature=0.7, json_format: bool = False) -> str:
        """Generate a response from the agent.

        Args:
            max_tokens (int, optional): Maximum number of tokens. Defaults to 3585.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            json_format (bool, optional): Whether to enable JSON output format. Defaults to False.

        Returns:
            str: The generated response.
        """
        if not self.history.messages:
            raise ValueError("No messages in history to generate response from")
        
        messages = self.history.messages

        response_text = completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key,
            json_format=json_format  # Pass json_format to completion
        )
        if self.model_name.startswith("openai"):
            # OpenAI does not support images, so responses are simple strings
            if self.history.messages[-1]["role"] == "assistant":
                self.history.messages[-1]["content"] = response_text
            elif self.memory_enabled:
                self.add_message("assistant", response_text)
        elif "claude" in self.model_name:
            if self.history.messages[-1]["role"] == "assistant":
                self.history.messages[-1]["content"] = response_text
            elif self.memory_enabled:
                self.add_message("assistant", response_text)
        elif "gemini" in self.model_name or "grok" in self.model_name:
            if self.history.messages[-1]["role"] == "assistant":
                if isinstance(self.history.messages[-1]["content"], list):
                    self.history.messages[-1]["content"].append(response_text)
                else:
                    self.history.messages[-1]["content"] = [self.history.messages[-1]["content"], response_text]
            elif self.memory_enabled:
                self.add_message("assistant", response_text)
        else:
            # Handle other models similarly
            if self.history.messages[-1]["role"] == "assistant":
                self.history.messages[-1]["content"] = response_text
            elif self.memory_enabled:
                self.add_message("assistant", response_text)
        
        return response_text
    
    def save_conversation(self):
        filename = f"{self.id}.json"
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.history.messages, file, ensure_ascii=False, indent=4)

    def load_conversation(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"{self.id}.json"
        with open(filename, 'r', encoding='utf-8') as file:
            messages = json.load(file)
            # Handle deserialization of images if necessary
            self.history = ChatHistory(messages)

    def add_repo(self, repo_url: Optional[str] = None, username: Optional[str] = None, repo_name: Optional[str] = None, commit_hash: Optional[str] = None):
        if username and repo_name:
            if commit_hash:
                repo_url = f"https://github.com/{username}/{repo_name}/archive/{commit_hash}.zip"
            else:
                repo_url = f"https://github.com/{username}/{repo_name}/archive/refs/heads/main.zip"
        
        if not repo_url:
            raise ValueError("Either repo_url or both username and repo_name must be provided")
        
        response = requests.get(repo_url)
        if response.status_code == 200:
            repo_content = ""
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_info in z.infolist():
                    if not file_info.is_dir() and file_info.filename.endswith(('.py', '.txt')):
                        with z.open(file_info) as f:
                            content = f.read().decode('utf-8')
                            repo_content += f"{file_info.filename}\n```\n{content}\n```\n"
            self.repo_content.append(repo_content)
        else:
            raise ValueError(f"Failed to download repository from {repo_url}")

if __name__ == "__main__":
    # Example Usage
    # Create an Agent instance (Gemini model)
    agent = Agent("gemini-1.5-flash", "you are an assistant", memory_enabled=True)
    
    # Add an image
    agent.add_image(image_path="/Users/junfan/Projects/Personal/oneapi/dialog_manager/example.png")
    
    # Add a user message
    agent.add_message("user", "What's in this image?")
    
    # Generate response with JSON format enabled
    try:
        response = agent.generate_response(json_format=True)  # json_format set to True
        print("Response:", response)
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")

    # Print the entire conversation history
    print("Conversation History:")
    print(agent.history)
    
    # Pop the last message
    last_message = agent.history.pop()
    print("Last Message:", last_message)

    # Generate another response without JSON format
    response = agent.generate_response()
    print("Response:", response)
