#!/usr/bin/env python3

import argparse
import yaml
import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from language_models import load_attack_model, load_policy_model
from tool_judge import ToolJudge
from tool_discovery import ToolDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Vapi Attacker LLM Server")

# Global state for conversation tracking
conversation_states = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class VapiAttacker:
    def __init__(self, attack_model: str, helper_model: str, judge_model: str):
        """Initialize the Vapi attacker with the specified models."""
        self.attack_model = load_attack_model(attack_model)
        self.helper_model = load_attack_model(helper_model)
        self.policy_model = load_policy_model(judge_model)
        self.tool_judge = None
        self.tool_discovery = None
        self.conversation_history = []
        self.discovered_tools = {}
        self.current_phase = "discovery"  # or "usage"
        self.current_tool = None

    def set_tool_config(self, tools: List[Dict[str, Any]]):
        """Set the tool configuration for the target agent."""
        self.tool_judge = ToolJudge(None, tools)
        self.tool_discovery = ToolDiscovery(
            tool_judge=self.tool_judge,
            attack_model=self.attack_model,
            policy_model=self.policy_model
        )

    def generate_response(self, messages: List[ChatMessage]) -> str:
        """Generate a response based on the conversation state and phase."""
        # Extract the latest user message
        latest_message = messages[-1].content if messages else ""
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": latest_message})
        
        if self.current_phase == "discovery":
            response, done = self.tool_discovery.handle_discovery_phase(latest_message)
            if done:
                # Log discovery results
                self.tool_discovery.log_discovery_results(
                    output_dir=f'outputs/voice_tool_enumeration_{self.attack_model.model_name}',
                    agent_name=self.tool_judge.tools[0]['name']  # Use first tool's name as agent name
                )
                self.current_phase = "usage"
            return response
        else:
            return self.tool_discovery.handle_usage_phase(latest_message)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """Handle chat completion requests from Vapi."""
    try:
        # Get or create attacker instance for this conversation
        conversation_id = request.model
        if conversation_id not in conversation_states:
            # Initialize with default models - these should be configurable
            conversation_states[conversation_id] = VapiAttacker(
                attack_model="grok",
                helper_model="grok",
                judge_model="gpt-4"
            )
        
        attacker = conversation_states[conversation_id]
        
        # Generate response
        response_text = attacker.generate_response(request.messages)
        
        # Create response in OpenAI format
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # We don't actually count tokens
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def load_tool_config(config_path: str) -> Dict[str, Any]:
    """Load tool configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load tool config from {config_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run Vapi attacker LLM server')
    parser.add_argument(
        "--tool-config",
        type=str,
        required=True,
        help="Path to YAML file containing tool configurations"
    )
    parser.add_argument(
        "--attack-model",
        type=str,
        default="grok",
        help="Model to use for generating attacks"
    )
    parser.add_argument(
        "--helper-model",
        type=str,
        default="grok",
        help="Model to use for generating follow-up questions"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        help="Model to use for evaluating tool discovery"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    
    args = parser.parse_args()
    
    try:
        # Load tool configuration
        config = load_tool_config(args.tool_config)
        
        # Initialize global attacker with tool config
        for agent in config['agents']:
            attacker = VapiAttacker(
                attack_model=args.attack_model,
                helper_model=args.helper_model,
                judge_model=args.judge_model
            )
            attacker.set_tool_config(agent['tools'])
            conversation_states[agent['name']] = attacker
        
        # Start the server
        uvicorn.run(app, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)

if __name__ == "__main__":
    main() 