import ast
import logging
from fastchat.model import get_conversation_template
from language_models import GPT, Claude, Gemini, PaLM, HuggingFace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, VICUNA_13B_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P 
from LM_util import load_indiv_model, get_model_path_and_template
import json
import regex as re
def extract_json_attack_backup(s):
    print(s)
    try:
        json_match = re.search(r'{.*}', s, re.DOTALL)
        if json_match:
            json_like_content = json_match.group(0)
            clean_content = json_like_content.replace("```python", "").replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_content)
            keys = list(parsed.keys())
            if not all(x in parsed for x in keys):
                logging.error("Error in extracted structure. Missing keys.")
                logging.error(f"Extracted:\n {parsed}")
                return None, None, None
            new_jb_prompt = parsed[keys[0]]
            return parsed, s, new_jb_prompt
        else:
            print("No JSON-like content found.")
            return None, None, None
        
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        # logging.error(f"Extracted:\n {s}")
        return None, None, None

def extract_json_attack(s):
    # print(f"Raw input:\n{s}")
    
    # First try to find JSON-like content using regex
    json_match = re.search(r'{.*}', s, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        print(f"\nFound JSON-like content:\n{json_str}")
        
        # Clean up the JSON string
        json_str = json_str.replace("\n", "").strip()
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        try:
            # Try parsing as JSON first
            parsed = json.loads(json_str)
            print(f"\nSuccessfully parsed JSON:\n{parsed}")
        except json.JSONDecodeError:
            try:
                # Fallback to eval if JSON parsing fails
                parsed = eval(json_str)
                print(f"\nSuccessfully parsed using eval:\n{parsed}")
            except (SyntaxError, ValueError) as e:
                logging.error(f"Error parsing JSON structure: {str(e)}")
                # logging.error(f"Failed JSON string:\n{json_str}")
                return None, None, None
        
        # Validate required keys
        if not isinstance(parsed, dict) or "new_prompt" not in parsed:
            logging.error("Error: Missing required 'new_prompt' key in parsed structure")
            logging.error(f"Parsed structure:\n{parsed}")
            return None, None, None
            
        return parsed, json_str, parsed['new_prompt']
    else:
        logging.error("No JSON-like content found in response")
        logging.error(f"Input string:\n{s}")
        return None, None, None

def load_attack_model(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    return attackLM

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list, is_tool_discovery=False):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Uses function calls to get structured responses.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        - is_tool_discovery: Whether this is a tool discovery attempt.
        
        Returns:
        - List of generated attack prompts or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initialize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = ""
        else:
            init_message = ""

        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.system_message = ""
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "o1" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "claude-3" in self.model_name:
                full_prompts.append(prompt)
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            elif "claude-2" in self.model_name:
                full_prompts.append(prompt)
            elif "gemini" in self.model_name:
                full_prompts.append(prompt)
            elif "gemma" in self.model_name:
                full_prompts.append(prompt)
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt())

        for attempt in range(self.max_n_attack_attempts):
            print(f"\n=== Attack Attempt {attempt + 1} ===")
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs using function calls
            outputs_list = self.model.batched_generate(
                full_prompts_subset,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                is_attack=not is_tool_discovery,
                is_tool_discovery=is_tool_discovery,
            )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if output is None:
                    print(f"Warning: Got None output for prompt {i}")
                    new_indices_to_regenerate.append(orig_index)
                    continue

                try:
                    # Parse the function call arguments
                    parsed = json.loads(output)
                    if "new_prompt" in parsed:
                        valid_outputs[orig_index] = parsed["new_prompt"]
                    else:
                        print(f"Warning: Missing 'new_prompt' in parsed output: {parsed}")
                        new_indices_to_regenerate.append(orig_index)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing attack output: {str(e)}")
                    # print(f"Raw output: {output}")
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs
