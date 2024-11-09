import pytest
from llm_dialog_manager.key_manager import KeyManager

def test_key_manager_initialization(env_setup):
    manager = KeyManager()
    assert len(manager._configs["openai"]) == 2
    assert manager._configs["openai"][0].api_key == "test_openai_key_1"
    assert manager._configs["openai"][0].base_url == "https://test1.com"

def test_load_balancing(env_setup):
    manager = KeyManager()
    
    # Get first key
    key1, url1 = manager.get_config("openai")
    assert key1 == "test_openai_key_1"
    
    # Get second key (should be different due to load balancing)
    key2, url2 = manager.get_config("openai")
    assert key2 == "test_openai_key_2"
    
    # Release first key
    manager.release_config("openai", key1)
    
    # Should get first key again
    key3, url3 = manager.get_config("openai")
    assert key3 == "test_openai_key_1"

def test_error_handling(env_setup):
    manager = KeyManager()
    
    key, _ = manager.get_config("openai")
    manager.report_error("openai", key)
    
    stats = manager.get_stats()
    assert stats["openai"][0]["errors"] == 1