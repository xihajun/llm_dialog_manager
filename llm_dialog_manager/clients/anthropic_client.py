"""
Client implementation for Anthropic Claude models
"""
import os
import logging
import httpx
from typing import List, Dict, Optional, Union

import anthropic
from anthropic import AnthropicVertex, AnthropicBedrock

from ..formatters import AnthropicFormatter
from .base import BaseClient

logger = logging.getLogger(__name__)

class AnthropicClient(BaseClient):
    """Client for Anthropic Claude API"""
    
    def _get_service_name(self) -> str:
        return "anthropic"
    
    def completion(self, messages, max_tokens=1000, temperature=0.5, 
                   top_p=1.0, top_k=40, json_format=False, **kwargs):
        """
        Generate a completion using Anthropic Claude.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            json_format: Whether to return JSON
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        try:
            # Get API credentials if not set
            self.get_credentials()
            
            # Format messages for Anthropic API
            formatter = AnthropicFormatter()
            system_message, formatted_messages = formatter.format_messages(messages)
            
            # Get the model name from kwargs or use default
            model_name = kwargs.get("model", "claude-3-opus")
            
            # Check for Vertex configuration
            vertex_project_id = os.getenv('VERTEX_PROJECT_ID')
            vertex_region = os.getenv('VERTEX_REGION')
            
            # Check for AWS Bedrock configuration
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')
            
            # Get proxy configuration from environment or default to None
            http_proxy = os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("HTTPS_PROXY")
            
            # Determine if we should use Bedrock based on model name prefix
            use_bedrock = "anthropic." in model_name
            
            if use_bedrock:
                logger.info(f"Using AWS Bedrock for model: {model_name}")
                # Use AWS Bedrock for Claude
                bedrock_kwargs = {
                    "aws_region": aws_region
                }
                
                # Only add credentials if explicitly provided
                if aws_access_key and aws_secret_key:
                    bedrock_kwargs["aws_access_key"] = aws_access_key
                    bedrock_kwargs["aws_secret_key"] = aws_secret_key
                    
                client = AnthropicBedrock(**bedrock_kwargs)
                
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=formatted_messages,
                    top_p=top_p,
                    top_k=top_k
                )
            elif vertex_project_id and vertex_region:
                # Use Vertex AI for Claude
                client = AnthropicVertex(
                    region=vertex_region,
                    project_id=vertex_project_id
                )
                
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=formatted_messages,
                    top_p=top_p,
                    top_k=top_k
                )
            else:
                # Create httpx client with proxy settings if needed
                http_options = {}
                if http_proxy or https_proxy:
                    proxies = {}
                    if http_proxy:
                        proxies["http://"] = http_proxy
                    if https_proxy:
                        proxies["https://"] = https_proxy
                    http_options["proxies"] = proxies
                
                # Use direct Anthropic API with proper http client
                client = anthropic.Anthropic(
                    api_key=self.api_key, 
                    base_url=self.base_url,
                    http_client=httpx.Client(**http_options) if http_options else None
                )
                
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=formatted_messages,
                    top_p=top_p,
                    top_k=top_k
                )
            
            # Release API credentials
            self.release_credentials()
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            self.report_error()
            raise
