# LLM Dialog Manager

A Python package for managing AI chat conversation history with support for multiple LLM providers (OpenAI, Anthropic, Google, X.AI) and convenient conversation management features.

## Features

- Support for multiple AI providers:
  - OpenAI (GPT-3.5, GPT-4)
  - Anthropic (Claude)
  - Google (Gemini)
  - X.AI (Grok)
- Intelligent message role management (system, user, assistant)
- Conversation history tracking and validation
- Load balancing across multiple API keys
- Error handling and retry mechanisms
- Conversation saving and loading
- Memory management options
- Conversation search and indexing
- Rich conversation display options

## Installation

```bash
pip install llm-dialog-manager
```

## Quick Start

### Basic Usage

```python
from llm_dialog_manager import ChatHistory

# Initialize with a system message
history = ChatHistory("You are a helpful assistant")

# Add messages
history.add_user_message("Hello!")
history.add_assistant_message("Hi there! How can I help you today?")

# Print conversation
print(history)
```

### Using the AI Agent

```python
from llm_dialog_manager import Agent

# Initialize an agent with a specific model
agent = Agent("claude-2.1", memory_enabled=True)

# Add messages and generate responses
agent.add_message("system", "You are a helpful assistant")
agent.add_message("user", "What is the capital of France?")
response = agent.generate_response()

# Save conversation
agent.save_conversation()
```

## Advanced Features

### Managing Multiple API Keys

```python
from llm_dialog_manager import Agent

# Use specific API key
agent = Agent("gpt-4", api_key="your-api-key")

# Or use environment variables
# OPENAI_API_KEY_1=key1
# OPENAI_API_KEY_2=key2
# The system will automatically handle load balancing
```

### Conversation Management

```python
from llm_dialog_manager import ChatHistory

history = ChatHistory()

# Add messages with role validation
history.add_message("Hello system", "system")
history.add_message("Hello user", "user")
history.add_message("Hello assistant", "assistant")

# Search conversations
results = history.search_for_keyword("hello")

# Get conversation status
status = history.conversation_status()
history.display_conversation_status()

# Get conversation snippets
snippet = history.get_conversation_snippet(1)
history.display_snippet(1)
```

## Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY_1=your-key-1
OPENAI_API_BASE_1=https://api.openai.com/v1

# Anthropic
ANTHROPIC_API_KEY_1=your-anthropic-key
ANTHROPIC_API_BASE_1=https://api.anthropic.com

# Google
GEMINI_API_KEY=your-gemini-key

# X.AI
XAI_API_KEY=your-x-key
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
