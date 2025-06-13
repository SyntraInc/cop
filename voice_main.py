#!/usr/bin/env python3

import argparse
import yaml
import logging
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from attacker import load_attack_model
from LM_util import load_policy_model
from tool_judge import ToolJudge
from tool_discovery import ToolDiscovery
from urllib.parse import urlparse
import subprocess
import re
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if we're in development mode
IS_DEV = os.getenv('ENV') != 'production'

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
    def __init__(self, attack_model: str, helper_model: str, judge_model: str, tool_config: Dict[str, Any]):
        """Initialize the Vapi attacker with the specified models."""
        # Create args object for model loading
        class Args:
            def __init__(self, model_name, max_tokens=4000):
                self.attack_model = model_name
                self.helper_model = model_name  # This is used by load_policy_model
                self.judge_model = model_name
                self.attack_max_n_tokens = max_tokens
                self.helper_max_n_tokens = max_tokens  # This is used by load_policy_model
                self.judge_max_n_tokens = max_tokens
                self.max_n_attack_attempts = 10
                self.target_model = model_name
                self.target_max_n_tokens = max_tokens
                self.system_prompt_file = None
                self.behaviors_csv = None
                self.mode = "behavior"
                self.tool_config = tool_config
                self.keep_n = 3

        # Initialize models with proper args
        attack_args = Args(attack_model)
        helper_args = Args(helper_model)
        judge_args = Args(judge_model)

        self.attack_model = load_attack_model(attack_args)
        self.helper_model = load_attack_model(helper_args)
        self.policy_model = load_policy_model(helper_args)  # Use helper_args for policy model
        self.tool_judge = None
        self.tool_discovery = None
        self.conversation_history = []
        self.discovered_tools = {}
        self.current_phase = "discovery"  # or "usage"
        self.current_tool = None

    def set_tool_config(self, tools: List[Dict[str, Any]]):
        """Set the tool configuration for the target agent."""
        # Create args object for ToolJudge
        class Args:
            def __init__(self):
                self.judge_max_n_tokens = 10
                self.judge_temperature = 0
                self.judge_model = "gpt-4o"

        args = Args()
        self.tool_judge = ToolJudge(args, tools)
        self.tool_discovery = ToolDiscovery(
            tool_judge=self.tool_judge,
            attack_model=self.attack_model,
            policy_model=self.policy_model,
            target_model=self.helper_model
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
                judge_model="gpt-4",
                tool_config={}
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

def check_ngrok_server(port: int = 4332) -> Optional[str]:
    """
    Checks if ngrok is running on the specified port and returns the public URL if it is.
    """
    try:
        response = requests.get(f"http://localhost:4040/api/tunnels")
        if response.status_code == 200:
            tunnels = response.json()["tunnels"]
            for tunnel in tunnels:
                if tunnel["config"]["addr"] == f"http://localhost:{port}":
                    return tunnel["public_url"]
        logger.error(f"No ngrok tunnel found for port {port}. Please start ngrok with: ngrok http {port}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to ngrok API on port {port}. Is ngrok running? Start it with: ngrok http {port}")
        return None
    except Exception as e:
        logger.error(f"Error checking ngrok: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-config", required=True, help="Path to tool configuration YAML file")
    parser.add_argument("--attack-model", required=True, help="Name of the attack model to use")
    parser.add_argument("--helper-model", required=True, help="Name of the helper model to use")
    parser.add_argument("--judge-model", required=True, help="Name of the judge model to use")
    parser.add_argument("--port", type=int, default=4332, help="Port to run the server on")
    args = parser.parse_args()

    # Check if we're in development mode
    IS_DEV = os.getenv('ENV') != 'production'
    
    if IS_DEV:
        logger.info("Running in development mode")
        # Check for ngrok
        ngrok_url = check_ngrok_server(args.port)
        if not ngrok_url:
            logger.error(f"Failed to start server: ngrok not running on port {args.port}")
            sys.exit(1)
        logger.info(f"Found ngrok URL: {ngrok_url}")
        # Use local port for binding
        host = "0.0.0.0"
        port = args.port
    else:
        logger.info("Running in production mode")
        host = "0.0.0.0"
        port = args.port

    # Load tool configuration
    with open(args.tool_config, 'r') as f:
        tool_config = yaml.safe_load(f)

    # Create VapiAttacker instance
    attacker = VapiAttacker(
        attack_model=args.attack_model,
        helper_model=args.helper_model,
        judge_model=args.judge_model,
        tool_config=tool_config
    )

    # Start the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 