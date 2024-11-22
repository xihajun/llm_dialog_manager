# Standard library imports
import json
import os
import uuid
from typing import List, Dict, Optional
import logging
from pathlib import Path
import random

# Third-party imports
import anthropic
from anthropic import AnthropicVertex
import google.generativeai as genai
import openai
from dotenv import load_dotenv

# Local imports
from llm_dialog_manager.chat_history import ChatHistory
from llm_dialog_manager.key_manager import key_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(".env file not found. Using system environment variables.")

load_env_vars()

def create_and_send_message(client, model, max_tokens, temperature, messages, system_msg):
    """Function to send a message to the Anthropic API and handle the response."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            system=system_msg
        )
        return response
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise

def completion(model: str, messages: List[Dict[str, str]], max_tokens: int = 1000, 
              temperature: float = 0.5, api_key: Optional[str] = None, 
              base_url: Optional[str] = None) -> str:
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
            api_key, base_url = key_manager.get_config(service)

        try:
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

                system_msg = messages.pop(0)["content"] if messages and messages[0]["role"] == "system" else ""
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    system=system_msg
                )
                
                while response.stop_reason == "max_tokens":
                    if messages[-1]['role'] == "user":
                        messages.append({"role": "assistant", "content": response.content[0].text})
                    else:
                        messages[-1]['content'] += response.content[0].text

                    response = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        system=system_msg
                    )

                if messages[-1]['role'] == "assistant" and response.stop_reason == "end_turn":
                    messages[-1]['content'] += response.content[0].text
                    return messages[-1]['content']
                
                return response.content[0].text

            elif "gemini" in model:
                try:
                    # First try OpenAI-style API
                    client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/"
                    )
                    # Remove any system message from the beginning if present
                    if messages and messages[0]["role"] == "system":
                        system_msg = messages.pop(0)
                        # Prepend system message to first user message if exists
                        if messages:
                            messages[0]["content"] = f"{system_msg['content']}\n\n{messages[0]['content']}"
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
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
                                gemini_messages[0].parts[0].text = f"{msg['content']}\n\n{gemini_messages[0].parts[0].text}"
                        else:
                            gemini_messages.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
                    
                    # Create Gemini model and generate response
                    model = genai.GenerativeModel(model_name=model)
                    response = model.generate_content(
                        gemini_messages,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                    )
                    
                    return response.text

            elif "grok" in model:
                # Randomly choose between OpenAI and Anthropic SDK
                use_anthropic = random.choice([True, False])
                
                if use_anthropic:
                    print("using anthropic")
                    client = anthropic.Anthropic(
                        api_key=api_key,
                        base_url="https://api.x.ai"
                    )
                    
                    system_msg = messages.pop(0)["content"] if messages and messages[0]["role"] == "system" else ""
                    response = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        system=system_msg
                    )
                    return response.content[0].text
                else:
                    print("using openai")
                    client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.x.ai/v1"
                    )
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content

            else:  # OpenAI models
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content

            # Release the API key after successful use
            if not api_key:
                key_manager.release_config(service, api_key)

            return response

        except Exception as e:
            # Report error to key manager
            if not api_key:
                key_manager.report_error(service, api_key)
            raise

    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise

class Agent:
    def __init__(self, model_name: str, messages: Optional[str] = None, 
                 memory_enabled: bool = False, api_key: Optional[str] = None) -> None:
        """Initialize an Agent instance."""
        # valid_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-2.1', 'gemini-1.5-pro', 'gemini-1.5-flash', 'grok-beta', 'claude-3-5-sonnet-20241022']
        # if model_name not in valid_models:
        #     raise ValueError(f"Model {model_name} not supported. Supported models: {valid_models}")
        
        self.id = f"{model_name}-{uuid.uuid4().hex[:8]}"
        self.model_name = model_name
        self.history = ChatHistory(messages)
        self.memory_enabled = memory_enabled
        self.api_key = api_key

    def add_message(self, role, content):
        self.history.add_message(content, role)

    def generate_response(self, max_tokens=3585, temperature=0.7):
        if not self.history.messages:
            raise ValueError("No messages in history to generate response from")
        
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.history.messages]
        
        response_text = completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key
        )
        if messages[-1]["role"] == "assistant":
            self.history.messages[-1]["content"] = response_text

        elif self.memory_enabled:
            self.add_message("assistant", response_text)
        
        return response_text
    
    def save_conversation(self):
        filename = f"{self.id}.json"
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.history.messages, file, ensure_ascii=False, indent=4)

    def load_conversation(self):
        filename = f"{self.id}.json"
        with open(filename, 'r', encoding='utf-8') as file:
            messages = json.load(file)
            self.history = ChatHistory(messages)

if __name__ == "__main__":

    # write a test for detect finding agent
    text = "I think the answer is 42"

    # from agent.messageloader import information_detector_messages
    
    # # Now you can print or use information_detector_messages as needed
    # information_detector_agent = Agent("gemini-1.5-pro", information_detector_messages)
    # information_detector_agent.add_message("user", text)
    # response = information_detector_agent.generate_response()
    # print(response)
    agent = Agent("gemini-1.5-pro-002", "you are an assistant", memory_enabled=True)
    
    # Format the prompt to check if the section is the last one in the outline
    prompt = f"Say: {text}\n"
    
    # Add the prompt as a message from the user
    agent.add_message("user", prompt)
    agent.add_message("assistant", "the answer")

    print(agent.generate_response())
    print(agent.history[:])
    last_message = agent.history.pop()
    print(last_message)
    print(agent.history[:])
