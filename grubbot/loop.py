import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger

from .config import GrubbotConfig, ToolDefinition
from .cluster import FailureCluster
from .providers.base import BaseProvider
from .finetune import load_model, prepare_dataset, train, save_checkpoint
from .eval import evaluate
from .cluster import embed_failures, cluster_failures


def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        cleaned = cleaned.split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1]
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()

class LoopResult(BaseModel):
    iterations: int
    final_accuracy: float
    per_tool_accuracy: Dict[str, float]
    clusters_resolved: List[str]

def generate_targeted_data(cluster: FailureCluster, tools: List[ToolDefinition], provider: BaseProvider, target_count: int = 15) -> List[Dict[str, Any]]:
    # Format a prompt specifically highlighting the failure
    examples_str = "\n".join([f"Query: {e.user_query} | Expected: {json.dumps(e.expected)} | Model did: {e.predicted}" for e in cluster.examples[:3]])
    
    prompt = f"""
We have a local text generation model failing on specific tool use cases. 
The failure pattern is categorized as: {cluster.label}

Here are some examples of what it got wrong:
{examples_str}

Please generate {target_count} diverse NEW user queries and their correct expected tool_calls (JSON) that address this exact failure pattern. Focus on corner cases where the model might get confused.

Output ONLY a JSON array of objects, with each object structured exactly like:
{{
  "user_query": "...",
  "expected_tool_call": {{"name": "...", "arguments": {{...}}}}
}}
"""
    system_instruction = "You are an expert synthetic data generator fixing AI failure cases. Output raw JSON arrays."
    
    raw_response = provider.generate(prompt, system=system_instruction)
    raw_response = _extract_json_block(raw_response)
        
    all_examples = []
        
    # Tools schema again
    tools_schema = []
    for t in tools:
        props = {}
        for p_name, p_def in t.parameters.items():
            props[p_name] = {"type": p_def.type, "description": p_def.description}
        required = [k for k, v in t.parameters.items() if v.required]
        tools_schema.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {"type": "object", "properties": props, "required": required}
            }
        })
        
    try:
        data = json.loads(raw_response)
        if not isinstance(data, list):
            return []
        for item in data:
             if "user_query" not in item or "expected_tool_call" not in item:
                continue
             all_examples.append({
                "tools": tools_schema,
                "messages": [{"role": "user", "content": item["user_query"]}],
                "expected_tool_call": item["expected_tool_call"]
            })
    except Exception as e:
        logger.error(f"Failed to parse targeted examples for cluster {cluster.label}: {e}")
        
    return all_examples

def run_loop(
    config: GrubbotConfig,
    provider: BaseProvider,
    start_model_path: str | None = None,
    train_path: str = "data/train.jsonl",
    eval_path: str = "data/eval.jsonl",
) -> LoopResult:
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/failures", exist_ok=True)

    current_model_path = start_model_path or config.model_name
    best_accuracy = 0.0
    best_per_tool = {t.name: 0.0 for t in config.tools}
    clusters_resolved: List[str] = []
    completed_iterations = 0

    for iteration in range(1, config.goal.max_iterations + 1):
        logger.info(f"--- Loop Iteration {iteration}/{config.goal.max_iterations} ---")
        output_dir = f"models/grubbot-{config.model_name.replace('/', '-')}-v{iteration}"

        logger.info(f"Stage 2 - Finetuning from {current_model_path}")
        model, tokenizer = load_model(current_model_path)
        dataset = prepare_dataset(train_path, tokenizer)
        train(model, tokenizer, dataset, output_dir)
        save_checkpoint(model, tokenizer, output_dir)
        current_model_path = output_dir

        logger.info("Stage 3 - Evaluation")
        result = evaluate(model, tokenizer, eval_path, config.tools)
        completed_iterations = iteration

        logger.info(f"Iteration accuracy: {result.overall_accuracy * 100:.2f}%")
        if result.overall_accuracy >= best_accuracy:
            best_accuracy = result.overall_accuracy
            best_per_tool = result.per_tool_accuracy

        if result.overall_accuracy >= config.goal.target_accuracy:
            logger.info("Target met. Ending loop.")
            break

        if iteration >= config.goal.max_iterations:
            logger.info("Max iterations reached. Ending loop.")
            break

        if not result.failures:
            logger.info("No failures found. Ending loop.")
            break

        logger.info("Stage 4 - Cluster failures and generate targeted data")
        failures_path = f"data/failures/iter_{iteration}.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump([fx.model_dump() for fx in result.failures], f, indent=2)

        embeddings = embed_failures(result.failures)
        clusters = cluster_failures(result.failures, embeddings)

        appended = 0
        with open(train_path, "a", encoding="utf-8") as f:
            for cluster in clusters:
                clusters_resolved.append(cluster.label)
                targeted = generate_targeted_data(cluster, config.tools, provider, target_count=min(20, max(6, cluster.size * 3)))
                for idx, ex in enumerate(targeted):
                    ex["id"] = f"train_iter{iteration}_{cluster.cluster_id}_{idx}"
                    f.write(json.dumps(ex) + "\n")
                    appended += 1

        logger.info(f"Appended {appended} targeted examples.")

    return LoopResult(
        iterations=completed_iterations,
        final_accuracy=best_accuracy,
        per_tool_accuracy=best_per_tool,
        clusters_resolved=clusters_resolved,
    )
