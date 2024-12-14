import pytest
from llm_dialog_manager.agent import Agent

def test_agent_initialization(env_setup):
    # Test valid model initialization
    agent = Agent("claude-2.1")
    assert agent.model_name == "claude-2.1"
    
    # Test invalid model
    with pytest.raises(ValueError):
        Agent("invalid-model")

    # Test add_repo method
    agent.add_repo("https://github.com/some/repo/archive/refs/heads/main.zip")
    assert "some content from the repo" in agent.history.messages[-1]["content"]

    # Test add_repo method with username and repo name
    agent.add_repo(repo_url="", username="someuser", repo_name="somerepo")
    assert "some content from the repo" in agent.history.messages[-1]["content"]

def test_agent_message_handling(env_setup):
    agent = Agent("claude-2.1", memory_enabled=True)
    
    # Test adding messages
    agent.add_message("system", "You are a helpful assistant")
    agent.add_message("user", "Hello!")
    
    assert len(agent.history) == 2
    assert agent.history.messages[-1]["role"] == "user"

    # Test add_repo method
    agent.add_repo("https://github.com/some/repo/archive/refs/heads/main.zip")
    assert "some content from the repo" in agent.history.messages[-1]["content"]

    # Test add_repo method with username and repo name
    agent.add_repo(repo_url="", username="someuser", repo_name="somerepo")
    assert "some content from the repo" in agent.history.messages[-1]["content"]

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

def test_add_repo_with_github_repo(env_setup):
    agent = Agent("claude-2.1", memory_enabled=True)
    
    # Test add_repo method with a real GitHub repository
    agent.add_repo(repo_url="https://github.com/xihajun/llm_dialog_manager/archive/refs/heads/main.zip")
    
    # Check if .env.example file content is in the repo_content
    assert any(".env.example" in content for content in agent.repo_content)
