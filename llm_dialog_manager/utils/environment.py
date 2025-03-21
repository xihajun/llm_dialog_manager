"""
Environment utilities for LLM Dialog Manager
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env_vars(env_path=None):
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Optional path to .env file
    
    Returns:
        True if env vars were loaded, False otherwise
    """
    try:
        # Default to .env in the current directory or parent directory
        if not env_path:
            if os.path.exists(".env"):
                env_path = ".env"
            elif os.path.exists("../.env"):
                env_path = "../.env"
            else:
                # Try to find .env in parent directories
                current_dir = Path.cwd()
                for parent in current_dir.parents:
                    potential_path = parent / ".env"
                    if potential_path.exists():
                        env_path = str(potential_path)
                        break
        
        if env_path and os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
            
            # Log detected providers without showing sensitive data
            providers = []
            if os.getenv("OPENAI_API_KEY"):
                providers.append("OpenAI")
            if os.getenv("ANTHROPIC_API_KEY"):
                providers.append("Anthropic")
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                providers.append("AWS Bedrock (for anthropic.* models)")
            if os.getenv("VERTEX_PROJECT_ID") and os.getenv("VERTEX_REGION"):
                providers.append("Anthropic Vertex")
            if os.getenv("GEMINI_API_KEY"):
                providers.append("Google Gemini")
            if os.getenv("XAI_API_KEY"):
                providers.append("X.AI Grok")
            
            if providers:
                logger.info(f"Detected LLM providers: {', '.join(providers)}")
            
            return True
        else:
            logger.warning(f"Environment file not found: {env_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False
