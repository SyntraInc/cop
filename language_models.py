import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import vertexai
import random
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
# from vllm import LLM
# from vllm.sampling_params import SamplingParams
from config import MINISTRAL_8B_PATH
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from fastchat.model import get_conversation_template
import google.generativeai.types as genai_types
from config import VICUNA_PATH, VICUNA_13B_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
    
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.temperature = 1.0
        self.top_p = 0.9
        self.max_n_tokens = 4000
        self.max_n_attack_attempts = 10
        self.keep_n = 3
        self.conv_template = get_conversation_template(model_name)
        self.conv_template.sep = "\n"
        self.conv_template.sep2 = None
        self.conv_template.roles = ("user", "assistant")
        self.conv_template.stop_str = None
        self.conv_template.stop_token_ids = None
        self.model = openai.OpenAI()  # Initialize the OpenAI client
        
        # Define the function schema for structured responses
        self.functions = [{
            "name": "generate_prompt",
            "description": "Generate a new prompt based on the given instructions",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_prompt": {
                        "type": "string",
                        "description": "The generated prompt that follows the instructions"
                    }
                },
                "required": ["new_prompt"]
            }
        }]

        # Define the function schema for judge responses
        self.judge_functions = [{
            "name": "judge_response",
            "description": "Evaluate a response and provide a rating between 1 and 10",
            "parameters": {
                "type": "object",
                "properties": {
                    "rating": {
                        "type": "integer",
                        "description": "A rating between 1 and 10",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["rating"]
            }
        }]

        # Define the function schema for policy responses
        self.policy_functions = [{
            "name": "generate_policy",
            "description": "Generate a conversational strategy",
            "parameters": {
                "type": "object",
                "properties": {
                    "unsafe_objective": {
                        "type": "string",
                        "description": "The current goal to achieve"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name of the option"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of how this option works"
                                },
                                "primitive_actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of primitive actions this option uses"
                                }
                            },
                            "required": ["name", "description", "primitive_actions"]
                        }
                    },
                    "high_level_policy": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Description of the high-level policy"
                            },
                            "rules": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "condition": {
                                            "type": "string",
                                            "description": "Condition for selecting an option"
                                        },
                                        "option": {
                                            "type": "string",
                                            "description": "Name of the option to select"
                                        }
                                    },
                                    "required": ["condition", "option"]
                                }
                            }
                        },
                        "required": ["description", "rules"]
                    }
                },
                "required": ["unsafe_objective", "options", "high_level_policy"]
            }
        }]

    def batched_generate(self, 
                        prompts_list,
                        max_n_tokens = None,
                        temperature = None,
                        top_p = None,
                        is_judge = False,
                        is_policy = False,
                        is_attack = False,
                        has_tools = False):
        """
        Generate responses for a batch of prompts using the GPT model.
        
        Args:
            prompts_list: List of prompts to generate responses for
            max_n_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top p for generation
            is_judge: Whether this is a judge call
            is_policy: Whether this is a policy call
            is_attack: Whether this is an attack call
            has_tools: Whether the model has tools available
            
        Returns:
            List of generated responses
        """
        if max_n_tokens is None:
            max_n_tokens = self.max_n_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p

        # Determine which function schema to use
        if is_judge:
            functions = self.judge_functions
            function_call = {"name": "judge_response"}
            expected_function = "judge_response"
        elif is_policy:
            functions = self.policy_functions
            function_call = {"name": "generate_policy"}
            expected_function = "generate_policy"
        elif is_attack:
            functions = self.functions
            function_call = {"name": "generate_attack"}
            expected_function = "generate_attack"
        else:
            functions = None
            function_call = None
            expected_function = None

        # Generate responses
        outputs = []
        for i, prompt in enumerate(prompts_list):
            try:
                # Handle prompts with tool definitions
                if has_tools and isinstance(prompt, dict) and "functions" in prompt:
                    messages = prompt["messages"]
                    functions = prompt["functions"]
                    response = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_n_tokens,
                        functions=functions,
                        function_call="auto"  # Let the model decide whether to call functions
                    )
                else:
                    # Regular prompt handling
                    response = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=prompt if not has_tools else prompt["messages"],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_n_tokens,
                        functions=functions,
                        function_call=function_call
                    )
                
                # Extract the response
                message = response.choices[0].message
                if message.function_call:
                    args = json.loads(message.function_call.arguments)
                    if is_judge:
                        # Format judge output as [[number]]
                        outputs.append(f"[[{args['rating']}]]")
                    elif is_policy or is_attack:
                        outputs.append(args)
                    else:
                        # For tool calls, return both function call and content
                        outputs.append({
                            "function_call": {
                                "name": message.function_call.name,
                                "arguments": args
                            },
                            "content": message.content
                        })
                else:
                    # For regular responses, just return the content
                    outputs.append(message.content)
                    
            except Exception as e:
                print(f"Error generating response for prompt {i}: {str(e)}")
                raise  # Re-raise the exception to fail hard
                
        return outputs

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = "<YOUR_API_KEY>"
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.messages.create(
                    model="claude-2.1",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": conv}
                    ]
                )
                output = completion.content[0].text
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Gemini():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:
        PROJECT_ID = "<your-project-id>"  # @param {type: "string", placeholder: "[your-project-id]" isTemplate: true}
        if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
            PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        
        LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        MODEL_ID = "gemini-1.5-pro-002"  # @param {type:"string"}

        self.model = GenerativeModel(MODEL_ID)

        self.generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )
        
        # Set safety settings
    
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.OFF,
        }
    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                completion = self.model.generate_content(
                    conv,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                output = completion.text
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Ministral():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        model_name = MINISTRAL_8B_PATH

        self.sampling_params = SamplingParams(max_tokens=8192)

        self.model = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": conv
                    },
                ]
                completion = self.model.chat(messages, sampling_params=self.sampling_params, tensor_parallel_size=2)
                output = completion[0].outputs[0].text
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Claude3():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = "<YOUR_API_KEY>"
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                print(conv)
                completion = self.model.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": conv}
                    ]
                )
                output = completion.content[0].text
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class YI():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.model_name = "yi-lightning"

        self.API_BASE = "https://api.lingyiwanwu.com/v1"
        self.API_KEY = "<YOUR_API_KEY>"

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                template = [{"role": "user", "content": conv}]
                client = openai.OpenAI(
                    api_key=self.API_KEY,
                    base_url=self.API_BASE
                )
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=template
                )
                output = completion.choices[0].message.content
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class GROK():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:
        self.model_name = "grok-2-1212"
        self.url = "https://api.x.ai/v1/chat/completions"
        self.api_key = os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROK_API_KEY environment variable is not set")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float,
                is_judge: bool = False,
                is_attack: bool = False,
                is_policy: bool = False,
                is_tool_discovery: bool = False):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
            is_judge: bool, whether this is a judge call
            is_attack: bool, whether this is an attack call
            is_policy: bool, whether this is a policy call
            is_tool_discovery: bool, whether this is a tool discovery call
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": conv
                        }
                    ],
                    "model": self.model_name,
                    "stream": False,
                    "temperature": temperature
                }
                
                # Define the standard prompt function once
                prompt_function = {
                    "name": "generate_prompt",
                    "description": "Generate a prompt to achieve the given goal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_prompt": {
                                "type": "string",
                                "description": "The generated prompt that attempts to achieve the goal"
                            }
                        },
                        "required": ["new_prompt"]
                    }
                }

                if is_policy or is_attack or is_judge or is_tool_discovery:
                    if is_policy:
                        func_name = "generate_policy"
                        functions = policy_functions
                    elif is_judge:
                        func_name = "judge_response"
                        functions = judge_functions
                    elif is_tool_discovery:
                        func_name = "generate_prompt"
                        functions = [prompt_function]
                    else:  # is_attack
                        func_name = "generate_attack"
                        functions = attack_functions
                    
                    # Add type field to each tool
                    tools = []
                    for func in functions:
                        tool = {
                            "type": "function",  # Required type field
                            "function": func  # Original function definition
                        }
                        tools.append(tool)
                    data["tools"] = tools
                    
                    data["tool_choice"] = {
                        "type": "function",
                        "function": {
                            "name": func_name
                        }
                    }
                
                response = requests.post(self.url, headers=self.headers, json=data)
                response_json = response.json()
                
                if (is_policy or is_attack or is_judge or is_tool_discovery):
                    try:
                        # First try standard OpenAI format
                        message = response_json.get('choices', [{}])[0].get('message', {})
                        if message.get('tool_calls'):
                            output = message['tool_calls'][0]['function']['arguments']
                        else:
                            # Try alternate format where function call is direct
                            function_call = response_json.get('choices', [{}])[0].get('function_call', {})
                            if function_call:
                                output = function_call.get('arguments', '{}')
                            else:
                                raise ValueError(f"No tool calls found in response: {response_json}")
                    except (KeyError, IndexError) as e:
                        raise ValueError(f"Failed to parse tool call from response: {response_json}")
                else:
                    output = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not output:
                        output = response_json.get('choices', [{}])[0].get('content', '')
                break
            except Exception as e:
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,
                        is_judge: bool = False,
                        is_attack: bool = False,
                        is_policy: bool = False,
                        is_tool_discovery: bool = False):
        return [self.generate(conv, max_n_tokens, temperature, top_p, is_judge, is_attack, is_policy, is_tool_discovery) for conv in convs_list]

class DEEPSEEK_CHAT():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.client = openai.OpenAI(api_key="<YOUR_API_KEY>", base_url="https://api.deepseek.com")

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "user", "content": conv}
                    ],
                )
                output = response.choices[0].message.content
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class GPT_o1():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.model_name = "o1"
        self.API_KEY = "<YOUR_API_KEY>"

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                client = openai.OpenAI(
                    api_key=self.API_KEY
                )
                print(conv[1]['content'])
                completion = client.chat.completions.create(
                    model = self.model_name,
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": conv[1]['content']
                                },
                            ],
                        }
                    ]
                )
                output = completion.choices[0].message.content
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class GEMMA():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained("/workspace/gemma-7b-it")
        self.model = AutoModelForCausalLM.from_pretrained(
            "/workspace/gemma-7b-it",
            device_map="auto"
        )
    def extract_model_response(self, output: str):
        # Find the part starting with <start_of_turn>model
        start_marker = "<start_of_turn>model"
        start_index = output.find(start_marker)
        
        if start_index == -1:
            return None  # Return None if <start_of_turn>model is not found
        
        # Extract everything after <start_of_turn>model
        model_text = output[start_index + len(start_marker):].strip()
        
        # Optionally remove any <end_of_turn> marker or other tags
        end_marker = "<end_of_turn>"
        if end_marker in model_text:
            model_text = model_text.split(end_marker)[0].strip()
        
        return model_text

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                chat = [
                    { "role": "user", "content": conv },
                ]
                prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=500)
                text = self.tokenizer.decode(outputs[0])
                output = self.extract_model_response(output=text)
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
        
class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last
                
                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(1)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
# Function schemas for different types of responses
prompt_functions = [
    {
        "name": "generate_prompt",
        "description": "Generate a prompt to achieve the given goal",
        "parameters": {
            "type": "object",
            "properties": {
                "new_prompt": {
                    "type": "string",
                    "description": "The generated prompt that attempts to achieve the goal"
                }
            },
            "required": ["new_prompt"]
        }
    }
]

judge_functions = [
    {
        "name": "judge_response",
        "description": "Judge whether a response achieves the goal",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Score from 1-10 indicating how well the response achieves the goal",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["score"]
        }
    }
]

policy_functions = [
    {
        "name": "generate_policy",
        "description": "Generate a conversational strategy",
        "parameters": {
            "type": "object",
            "properties": {
                "unsafe_objective": {
                    "type": "string",
                    "description": "The current goal to achieve"
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the option"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of how this option works"
                            },
                            "primitive_actions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of primitive actions this option uses"
                            }
                        },
                        "required": ["name", "description", "primitive_actions"]
                    }
                },
                "high_level_policy": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the high-level policy"
                        },
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "condition": {
                                        "type": "string",
                                        "description": "Condition for selecting an option"
                                    },
                                    "option": {
                                        "type": "string",
                                        "description": "Name of the option to select"
                                    }
                                },
                                "required": ["condition", "option"]
                            }
                        }
                    },
                    "required": ["description", "rules"]
                }
            },
            "required": ["unsafe_objective", "options", "high_level_policy"]
        }
    }
]

attack_functions = [
    {
        "name": "generate_attack",
        "description": "Generate an attack prompt to achieve the given goal",
        "parameters": {
            "type": "object",
            "properties": {
                "new_prompt": {
                    "type": "string",
                    "description": "The generated attack prompt that attempts to achieve the goal"
                }
            },
            "required": ["new_prompt"]
        }
    }
]

tool_discovery_functions = [
    {
        "name": "generate_prompt",
        "description": "Generate a prompt to achieve the given goal",
        "parameters": {
            "type": "object",
            "properties": {
                "new_prompt": {
                    "type": "string",
                    "description": "The generated prompt that attempts to achieve the goal"
                }
            },
            "required": ["new_prompt"]
        }
    }
]

