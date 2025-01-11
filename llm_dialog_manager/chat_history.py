from typing import List, Dict, Optional, Union
from PIL import Image

class ChatHistory:
    def __init__(self, input_data: Union[str, List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]] = "") -> None:
        self.messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]] = []
        if isinstance(input_data, str) and input_data:
            self.add_message(input_data, "system")
        elif isinstance(input_data, list):
            self.load_messages(input_data)
        self.last_role: str = "system" if not self.messages else self.get_last_role()

    def load_messages(self, messages: List[Dict[str, Union[str, List[Union[str, Image.Image, Dict]]]]]) -> None:
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("Each message must have a 'role' and 'content'.")
            if message["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {message['role']}")
            self.messages.append(message)
        self.last_role = self.get_last_role()

    def get_last_role(self):
        return self.messages[-1]["role"] if self.messages else "system"

    def pop(self):
        if not self.messages:
            return None

        popped_message = self.messages.pop()

        if self.messages:
            self.last_role = self.get_last_role()
        else:
            self.last_role = "system"

        return popped_message["content"]

    def __len__(self):
        return len(self.messages)

    def __str__(self):
        formatted_messages = []
        for i, msg in enumerate(self.messages):
            role = msg['role']
            content = msg['content']
            if isinstance(content, str):
                formatted_content = content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, Image.Image):
                        parts.append(f"[Image Object: {block.filename}]")
                    elif isinstance(block, dict):
                        if block.get("type") == "image_url":
                            parts.append(f"[Image URL: {block.get('image_url', {}).get('url', '')}]")
                        elif block.get("type") == "image_base64":
                            parts.append(f"[Image Base64: {block.get('image_base64', {}).get('data', '')[:30]}...]")
                formatted_content = "\n".join(parts)
            else:
                formatted_content = str(content)
            formatted_messages.append(f"Message {i} ({role}): {formatted_content}")
        return '\n'.join(formatted_messages)

    def __getitem__(self, key):
        if isinstance(key, slice):
            sliced_messages = self.messages[key]
            formatted = []
            for msg in sliced_messages:
                role = msg['role']
                content = msg['content']
                if isinstance(content, str):
                    formatted_content = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, Image.Image):
                            parts.append(f"[Image Object: {block.filename}]")
                        elif isinstance(block, dict):
                            if block.get("type") == "image_url":
                                parts.append(f"[Image URL: {block.get('image_url', {}).get('url', '')}]")
                            elif block.get("type") == "image_base64":
                                parts.append(f"[Image Base64: {block.get('image_base64', {}).get('data', '')[:30]}...]")
                    formatted_content = "\n".join(parts)
                else:
                    formatted_content = str(content)
                formatted.append(f"({role}): {formatted_content}")
            print('\n'.join(formatted))
            return sliced_messages
        elif isinstance(key, int):
            # Adjust for negative indices
            if key < 0:
                key += len(self.messages)
            if 0 <= key < len(self.messages):
                msg = self.messages[key]
                role = msg['role']
                content = msg['content']
                if isinstance(content, str):
                    formatted_content = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, Image.Image):
                            parts.append(f"[Image Object: {block.filename}]")
                        elif isinstance(block, dict):
                            if block.get("type") == "image_url":
                                parts.append(f"[Image URL: {block.get('image_url', {}).get('url', '')}]")
                            elif block.get("type") == "image_base64":
                                parts.append(f"[Image Base64: {block.get('image_base64', {}).get('data', '')[:30]}...]")
                    formatted_content = "\n".join(parts)
                else:
                    formatted_content = str(content)
                snippet = self.get_conversation_snippet(key)
                print('\n'.join([f"({v['role']}): {v['content']}" for k, v in snippet.items() if v]))
                return self.messages[key]
            else:
                raise IndexError("Message index out of range.")
        else:
            raise TypeError("Invalid argument type.")

    def __setitem__(self, index, value):
        if not isinstance(value, (str, list)):
            raise ValueError("Message content must be a string or a list of content blocks.")
        role = "system" if index % 2 == 0 else "user"
        self.messages[index] = {"role": role, "content": value}

    def __add__(self, message):
        if self.last_role == "system":
            self.add_user_message(message)
        else:
            next_role = "assistant" if self.last_role == "user" else "user"
            self.add_message(message, next_role)

    def __contains__(self, item):
        for message in self.messages:
            content = message['content']
            if isinstance(content, str) and item in content:
                return True
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str) and item in block:
                        return True
        return False

    def add_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]], role: str):
        self.messages.append({"role": role, "content": content})
        self.last_role = role

    def add_user_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        if self.last_role in ["system", "assistant"]:
            self.add_message(content, "user")
        else:
            raise ValueError("A user message must follow a system or assistant message.")

    def add_assistant_message(self, content: Union[str, List[Union[str, Image.Image, Dict]]]):
        if self.last_role == "user":
            self.add_message(content, "assistant")
        else:
            raise ValueError("An assistant message must follow a user message.")

    def add_marker(self, marker, index=None):
        if not isinstance(marker, str):
            raise ValueError("Marker must be a string.")
        if index is None:
            index = len(self.messages) - 1
        if 0 <= index < len(self.messages):
            self.messages[index]["marker"] = marker
        else:
            raise IndexError("Invalid index for marker.")

    def conversation_status(self):
        return {
            "last_message_role": self.last_role,
            "total_messages": len(self.messages),
            "last_message_content": self.messages[-1]["content"] if self.messages else "No messages",
        }

    def display_conversation_status(self):
        status = self.conversation_status()
        print(f"Role of the last message: {status['last_message_role']}")
        print(f"Total number of messages: {status['total_messages']}")
        print(f"Content of the last message: {status['last_message_content']}")

    def search_for_keyword(self, keyword):
        results = []
        for msg in self.messages:
            content = msg['content']
            if isinstance(content, str) and keyword.lower() in content.lower():
                results.append(msg)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str) and keyword.lower() in block.lower():
                        results.append(msg)
                        break
        return results

    def has_user_or_assistant_spoken_since_last_system(self):
        for msg in reversed(self.messages):
            if msg["role"] == "system":
                return False
            if msg["role"] in ["user", "assistant"]:
                return True
        return False

    def get_conversation_snippet(self, index):
        snippet = {"previous": None, "current": None, "next": None}
        if 0 <= index < len(self.messages):
            snippet['current'] = self.messages[index]
            if index > 0:
                snippet['previous'] = self.messages[index - 1]
            if index + 1 < len(self.messages):
                snippet['next'] = self.messages[index + 1]
        else:
            raise IndexError("Invalid index.")
        return snippet

    def display_snippet(self, index):
        snippet = self.get_conversation_snippet(index)
        for key, value in snippet.items():
            if value:
                print(f"{key.capitalize()} Message ({value['role']}): {value['content']}")
            else:
                print(f"{key.capitalize()}: None")

    @staticmethod
    def color_text(text, color):
        colors = {"green": "\033[92m", "red": "\033[91m", "end": "\033[0m"}
        return f"{colors.get(color, '')}{text}{colors.get('end', '')}"
