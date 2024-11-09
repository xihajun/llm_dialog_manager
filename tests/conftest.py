import pytest
import os
from pathlib import Path
import sys

@pytest.fixture
def env_setup(tmp_path):
    """Create a temporary .env file for testing"""
    # Save original env vars if they exist
    original_vars = {}
    env_vars = ['OPENAI_API_KEY_1', 'OPENAI_API_BASE_1', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY']
    for var in env_vars:
        if var in os.environ:
            original_vars[var] = os.environ[var]
            del os.environ[var]
    
    # Create temporary .env file
    env_content = """
OPENAI_API_KEY_1=test_openai_key_1
OPENAI_API_BASE_1=https://test1.com
OPENAI_API_KEY_2=test_openai_key_2
OPENAI_API_BASE_2=https://test2.com
ANTHROPIC_API_KEY=test_anthropic_key
GEMINI_API_KEY=test_gemini_key
"""
    env_path = tmp_path / '.env'
    env_path.write_text(env_content)
    
    # Point the application to use this temporary .env file
    os.environ['ENV_FILE'] = str(env_path)
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    yield env_path
    
    # Cleanup
    if env_path.exists():
        env_path.unlink()
    
    # Restore original env vars
    for var, value in original_vars.items():
        os.environ[var] = value 