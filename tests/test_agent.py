import pytest
from llm_dialog_manager.agent import Agent

def test_agent_initialization(env_setup):
    # Test valid model initialization
    agent = Agent("claude-2.1")
    assert agent.model_name == "claude-2.1"
    
    # Test invalid model
    with pytest.raises(ValueError):
        Agent("invalid-model")

def test_agent_message_handling(env_setup):
    agent = Agent("claude-2.1", memory_enabled=True)
    
    # Test adding messages
    agent.add_message("system", "You are a helpful assistant")
    agent.add_message("user", "Hello!")
    
    assert len(agent.history) == 2
    assert agent.history.messages[-1]["role"] == "user"

def test_conversation_save_load(env_setup, tmp_path):
    agent = Agent("claude-2.1")
    agent.add_message("system", "Test message")
    
    # Save conversation
    agent.save_conversation()
    
    # Create new agent and load conversation
    new_agent = Agent("claude-2.1")
    new_agent.load_conversation()
    
    assert len(new_agent.history) == 1
    assert new_agent.history.messages[0]["content"] == "Test message" 