import json
import random
from typing import List, Dict, Any
from .config import ToolDefinition, GoalConfig
from .providers.base import BaseProvider


def _strip_markdown_fences(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()

def build_datagen_prompt(tool: ToolDefinition, count: int) -> str:
    """Builds a prompt asking the LLM to generate tool-use examples based on the tool definition."""
    prompt = f"""
You are tasked with generating {count} diverse user queries that should trigger the following tool:

Tool Name: {tool.name}
Description: {tool.description}

Parameters:
"""
    for p_name, p_def in tool.parameters.items():
        req = "required" if p_def.required else "optional"
        prompt += f"- {p_name} ({p_def.type}, {req}): {p_def.description}\n"
        
    prompt += """
Please generate the output as a valid JSON array of objects. 
Each object MUST have:
1. "user_query": A realistic user message (include varied phrasing, typos, partial info if optional parameters exist).
2. "expected_tool_call": A dictionary having the "name" of the tool, and "arguments" containing the exact extracted parameters from the user's query.

Make the examples diverse:
- Standard clear requests.
- Requests where some optional parameters are missing.
- Requests with extra conversational boilerplate.

Output strictly valid JSON (an array of objects) and nothing else.
"""
    return prompt

def generate_examples(tools: List[ToolDefinition], goal: GoalConfig, provider: BaseProvider, count_per_tool: int = 50) -> List[Dict[str, Any]]:
    all_examples = []
    
    # We serialize the full tools schema to include it in every example's "tools" field
    tools_schema = []
    for t in tools:
        props = {}
        required = []
        for p_name, p_def in t.parameters.items():
            props[p_name] = {"type": p_def.type, "description": p_def.description}
            if p_def.required:
                required.append(p_name)
        
        tools_schema.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        })
    
    for tool in tools:
        prompt = build_datagen_prompt(tool, count_per_tool)
        system_instruction = "You are a synthetic training data generator. Generate JSON responses only, without formatting blocks."
        
        raw_response = provider.generate(prompt=prompt, system=system_instruction)
        
        raw_response = _strip_markdown_fences(raw_response)
            
        try:
            generated_items = json.loads(raw_response.strip())
            if not isinstance(generated_items, list):
                print(f"Provider output for tool {tool.name} was not a JSON array. Skipping tool output.")
                continue
            for item in generated_items:
                if "user_query" not in item or "expected_tool_call" not in item:
                    continue
                # Format to ChatML with tool calls
                formatted_example = {
                    "tools": tools_schema,
                    "messages": [
                        {"role": "user", "content": item["user_query"]}
                    ],
                    "expected_tool_call": item["expected_tool_call"]
                }
                all_examples.append(formatted_example)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for tool {tool.name}: {e}")
            print(f"Raw response was: {raw_response[:200]}...")
            
    return all_examples

def split_and_save(examples: List[Dict[str, Any]], train_path: str, eval_path: str, split_ratio: float = 0.8):
    random.shuffle(examples)
    split_idx = int(len(examples) * split_ratio)
    
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]
    
    # Ensure dirs exist
    import os
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    
    with open(train_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(train_data):
            ex["id"] = f"train_{idx}"
            f.write(json.dumps(ex) + "\n")
            
    with open(eval_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(eval_data):
            ex["id"] = f"eval_{idx}"
            f.write(json.dumps(ex) + "\n")
