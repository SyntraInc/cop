#!/usr/bin/env python3

import yaml
import requests
import argparse
import logging
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VapiAgentSyncer:
    def __init__(self):
        """Initialize the Vapi agent syncer with API credentials."""
        self.api_key = os.getenv('VAPI_API_KEY')
        if not self.api_key:
            raise ValueError("VAPI_API_KEY must be set in environment variables")
        
        self.base_url = "https://api.vapi.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def load_agent_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load agent config from {config_path}: {e}")
            raise

    def list_assistants(self) -> List[Dict[str, Any]]:
        """List all assistants from Vapi."""
        try:
            response = requests.get(
                f"{self.base_url}/assistant",
                headers=self.headers,
                params={"limit": 1000}
            )
            response.raise_for_status()
            data = response.json()
            return data
            # If the response is a list, use it directly
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list assistants: {e}")
            raise

    def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Get assistant by ID from Vapi."""
        try:
            response = requests.get(
                f"{self.base_url}/assistant/{assistant_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Failed to get assistant {assistant_id}: {e}")
            raise

    def get_assistant_by_name(self, name: str) -> Dict[str, Any]:
        """Get assistant by name from Vapi."""
        assistants = self.list_assistants()
        for assistant in assistants:
            if assistant.get('name') == name:
                return assistant
        return None

    def delete_assistant(self, vapi_id: str):
        """Delete an assistant by Vapi-generated ID."""
        try:
            response = requests.delete(
                f"{self.base_url}/assistant/{vapi_id}",
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Deleted assistant with Vapi ID: {vapi_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete assistant {vapi_id}: {e}")
            raise

    def associate_phone_number(self, phone_number: str, assistant_id: str):
        """Associate a phone number with an assistant in Vapi."""
        # Normalize phone number for comparison
        def normalize(num):
            return num.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
        phone_number_norm = normalize(phone_number)
        phone_numbers = self.list_phone_numbers()
        for phone in phone_numbers:
            if normalize(phone.get('number', '')) == phone_number_norm:
                phone_id = phone['id']
                # PATCH the phone number to set assistantId
                try:
                    response = requests.patch(
                        f"{self.base_url}/phone-number/{phone_id}",
                        headers=self.headers,
                        json={"assistantId": assistant_id}
                    )
                    response.raise_for_status()
                    logger.info(f"Associated phone number {phone_number} with assistant {assistant_id}")
                    return
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to associate phone number: {e}")
                    raise
        raise ValueError(f"Phone number {phone_number} not found in Vapi.")

    def create_voice_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Delete existing assistant (if any) and create a new voice agent in Vapi."""
        agent = agent_config['agents'][0]  # Assuming single agent for now
        
        if not agent.get('vapi', {}).get('enabled', False):
            logger.info(f"Vapi integration not enabled for agent {agent['name']}")
            return None

        vapi_config = agent['vapi']
        
        # Format tools information for system prompt
        tools_info = "\n\nAvailable tools:\n"
        if 'tools' in agent:
            for tool in agent['tools']:
                tools_info += f"\n{tool['name']}: {tool['description']}\n"
                tools_info += "Parameters:\n"
                for arg in tool['args']:
                    required = "" if arg.get('optional', False) else " (required)"
                    tools_info += f"- {arg['name']}: {arg['description']}{required}\n"
        
        # Prepare the assistant data for Vapi
        assistant_data = {
            "name": agent['name'],
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": agent['system_prompt'] + tools_info
                    }
                ]
            },
            "transcriber": {
                "provider": "deepgram",
                "language": "en"
            }
        }
        # Add voice if present
        voice_id = vapi_config.get('voice_id')
        if not voice_id:
            raise ValueError("voice_id must be specified in the YAML under vapi.voice_id.")
        assistant_data["voice"] = {"provider": "vapi", "voiceId": voice_id}

        # Add firstMessage if present
        first_message = vapi_config.get('first_message')
        if first_message:
            assistant_data["firstMessage"] = first_message

        try:
            # Check if assistant exists by name
            existing_assistant = self.get_assistant_by_name(agent['name'])
            
            if existing_assistant:
                vapi_id = existing_assistant['id']
                logger.info(f"Deleting existing assistant {agent['name']} (Vapi ID: {vapi_id}) before creating a new one.")
                self.delete_assistant(vapi_id)

            # Create new assistant
            logger.info(f"Creating new assistant {agent['name']}")
            response = requests.post(
                f"{self.base_url}/assistant",
                headers=self.headers,
                json=assistant_data
            )
            response.raise_for_status()
            assistant = response.json()
            # Associate phone number if present
            phone_number = vapi_config.get('phone_number')
            if phone_number:
                self.associate_phone_number(phone_number, assistant['id'])
            return assistant

        except requests.exceptions.RequestException as e:
            if e.response is not None:
                logger.error(f"Vapi API error: {e.response.text}")
            logger.error(f"Failed to sync assistant {agent['name']}: {e}")
            raise

    def list_phone_numbers(self) -> List[Dict[str, Any]]:
        """List all phone numbers in Vapi."""
        try:
            response = requests.get(
                f"{self.base_url}/phone-number",
                headers=self.headers
            )
            response.raise_for_status()
            phone_numbers = response.json()
            # Remove debug logging of phone numbers
            return phone_numbers
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list phone numbers: {e}")
            raise

    def verify_phone_number_setup(self, agent_name: str) -> bool:
        """Verify that the agent has a phone number set up in Vapi."""
        try:
            phone_numbers = self.list_phone_numbers()
            for phone in phone_numbers:
                if phone.get('assistantId') and phone.get('status') == 'active':
                    # Get the assistant details to check the name
                    assistant = self.get_assistant_by_name(agent_name)
                    if assistant and phone['assistantId'] == assistant['id']:
                        logger.info(f"Found active phone number for assistant {agent_name}")
                        return True
            logger.error(f"No active phone number found for assistant {agent_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to verify phone number setup: {e}")
            return False

    def sync_agents(self, config_file: str):
        """Sync agents from YAML config with Vapi."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            for agent_config in config['agents']:
                # Verify phone number setup before proceeding
                if not self.verify_phone_number_setup(agent_config['name']):
                    raise ValueError(f"Agent {agent_config['name']} does not have an active phone number set up in Vapi")
                
                self.create_voice_agent(agent_config)

        except Exception as e:
            logger.error(f"Failed to sync assistants: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Sync voice agents with Vapi')
    parser.add_argument('--config', required=True, help='Path to agent configuration YAML file')
    args = parser.parse_args()

    try:
        syncer = VapiAgentSyncer()
        config = syncer.load_agent_config(args.config)
        result = syncer.create_voice_agent(config)
        
        if result:
            logger.info(f"Successfully synced assistant: {result}")
        else:
            logger.info("No assistants were synced (Vapi integration not enabled)")

    except Exception as e:
        logger.error(f"Failed to sync assistants: {e}")
        exit(1)

if __name__ == "__main__":
    main() 