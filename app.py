from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from llm_dialog_manager import Agent
import time
from pathlib import Path

from dotenv import load_dotenv

# Local imports
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
from datetime import datetime
import os


app = FastAPI()

def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(".env file not found. Using system environment variables.")

load_env_vars()


class ChatManager:
    def __init__(self):
        self.agent = None
        self.current_model = None 

    async def handle_message(self, data):
        if isinstance(data, dict):
            # Get model from request data
            model = data.get('model', 'claude-3-5-sonnet-20241022')  # Default to Claude if not specified
            
            # Initialize or update agent with selected model
            self.agent = Agent(model_name=model)

            # Handle system prompt
            system_prompt = data.get('system', '')
            if system_prompt:
                self.agent.system_prompt = system_prompt
            else:
                self.agent.system_prompt = "You are an assistant"

            # Clear previous messages
            self.agent.messages = []

            # Add system message
            self.agent.add_message("system", self.agent.system_prompt)

            # Process message pairs
            for pair in data.get('messages', []):
                user_message = pair.get('user')
                assistant_message = pair.get('assistant')
                if user_message:
                    self.agent.add_message("user", user_message)
                if assistant_message:
                    self.agent.add_message("assistant", assistant_message)

            print(self.agent.messages)
            #TODO: not working with multiple messages
            print("start generate response")
            start_time = time.time()
            try:
                response = self.agent.generate_response()  # Removed await
                print("response", response)
                if isinstance(response, dict):
                    content = response.get('content', '')  # Handle dict response
                else:
                    content = str(response)  # Convert response to string if it's not a dict
            except Exception as e:
                print(f"Generation error: {e}")
                content = f"Error generating response: {str(e)}"
            end_time = time.time()
            
            # Calculate metrics
            elapsed_time = end_time - start_time
            formatted_time = f"{elapsed_time:.1f}s"
            token_count = len(content) // 4  # Using content instead of response

            return {
                "content": content,  # Using processed content
                "metrics": {
                    "confidence": 92,
                    "time": formatted_time,
                    "tokens": token_count
                },
                "actions": {
                    "add_to_chat": True
                }
            }
        else:
            # Handle other types of data if necessary
            pass

chat_manager = ChatManager()

@app.get("/")
async def get():
    with open("llm_dialog_manager/interface/static/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            response = await chat_manager.handle_message(data)
            await websocket.send_json(response)
        except Exception as e:
            print(f"Error: {e}")
            break

@app.get("/api/history")
async def get_history():
    """Get list of saved chat histories"""
    history_dir = Path("chat_history")
    if not history_dir.exists():
        history_dir.mkdir(exist_ok=True)
        return []
    
    histories = []
    for file in history_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            histories.append({
                "id": file.stem,
                "title": data.get("title", "Untitled"),
                "timestamp": data.get("timestamp"),
            })
    
    # Sort by timestamp descending
    histories.sort(key=lambda x: x["timestamp"], reverse=True)
    return histories

@app.get("/api/history/{history_id}")
async def get_history_by_id(history_id: str):
    """Get specific chat history by ID"""
    file_path = Path(f"chat_history/{history_id}.json")
    if not file_path.exists():
        return {"error": "History not found"}
    
    with open(file_path) as f:
        return json.load(f)

@app.post("/api/history")
async def save_history(data: dict):
    """Save chat history"""
    history_dir = Path("chat_history")
    history_dir.mkdir(exist_ok=True)
    
    # Generate unique ID using timestamp
    history_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Add timestamp to data
    data["timestamp"] = datetime.now().isoformat()
    
    with open(history_dir / f"{history_id}.json", "w") as f:
        json.dump(data, f)
    
    return {"id": history_id}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8099)

if __name__ == "__main__":
    main()
