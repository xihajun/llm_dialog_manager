import pytest
from llm_dialog_manager import ChatHistory

def test_chat_history_initialization():
    # Test empty initialization
    history = ChatHistory()
    assert len(history) == 0
    
    # Test initialization with system message
    history = ChatHistory("system message")
    assert len(history) == 1
    assert history.messages[0]["role"] == "system"
    
    # Test initialization with message list
    messages = [
        {"role": "system", "content": "system message"},
        {"role": "user", "content": "user message"}
    ]
    history = ChatHistory(messages)
    assert len(history) == 2

def test_add_message():
    history = ChatHistory()
    
    # Test adding messages in correct order
    history.add_message("Hello system", "system")
    history.add_message("Hello user", "user")
    history.add_message("Hello assistant", "assistant")
    
    assert len(history) == 3
    assert history.messages[-1]["role"] == "assistant"

def test_invalid_message_sequence():
    history = ChatHistory()
    
    # Test invalid sequence (user after user)
    history.add_message("Hello user", "user")
    with pytest.raises(ValueError):
        history.add_user_message("Invalid user message")

def test_pop_message():
    history = ChatHistory()
    history.add_message("test message", "user")
    
    popped = history.pop()
    assert popped == "test message"
    assert len(history) == 0 