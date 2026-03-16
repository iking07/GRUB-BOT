import os
import json
from loguru import logger

from .config import load_tools, load_goal_from_markdown, GrubbotConfig
from .providers import get_provider
from .datagen import generate_examples, split_and_save
from .finetune import load_model
from .eval import evaluate, EvalResult
from .loop import run_loop

def run_full_pipeline(tools_path: str, goal_path: str, model_name: str, provider_name: str = "gemini"):
    # Stage 0: Config Load
    logger.info("Loading tools and goal configuration...")
    tools = load_tools(tools_path)
    goal = load_goal_from_markdown(goal_path)
    config = GrubbotConfig(tools=tools, goal=goal, model_name=model_name, provider=provider_name)
    provider = get_provider(provider_name)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/failures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"
    
    # Stage 1: Data Generation (if either split is missing)
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        logger.info("Stage 1 - Generating base synthetic data...")
        examples = generate_examples(tools, goal, provider, count_per_tool=30)
        logger.info(f"Generated {len(examples)} examples.")
        split_and_save(examples, train_path, eval_path)
    else:
        logger.info("Base synthetic data exists, skipping Stage 1 initial generation.")

    loop_result = run_loop(
        config=config,
        provider=provider,
        start_model_path=model_name,
        train_path=train_path,
        eval_path=eval_path,
    )

    logger.info("=== Pipeline Complete ===")
    logger.info(f"Final Best Accuracy: {loop_result.final_accuracy * 100:.1f}%")
    logger.info(f"Total Iterations: {loop_result.iterations}")

    run_log = {
        "iterations": loop_result.iterations,
        "best_accuracy": loop_result.final_accuracy,
        "per_tool_accuracy": loop_result.per_tool_accuracy,
        "clusters_resolved": loop_result.clusters_resolved,
        "provider": provider_name,
        "model_name": model_name,
    }

    run_files = [f for f in os.listdir("runs") if f.startswith("run_") and f.endswith(".json")]
    next_index = len(run_files) + 1
    with open(f"runs/run_{next_index}.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)


def run_datagen_only(tools_path: str, goal_path: str, provider_name: str = "gemini"):
    """Run only Stage 1 to generate synthetic training data."""
    logger.info("Loading tools and goal configuration for datagen...")
    tools = load_tools(tools_path)
    goal = load_goal_from_markdown(goal_path)
    provider = get_provider(provider_name)
    
    os.makedirs("data", exist_ok=True)
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"
    
    logger.info(f"Stage 1 - Generating base synthetic data using {provider_name}...")
    examples = generate_examples(tools, goal, provider, count_per_tool=30)
    logger.info(f"Generated {len(examples)} examples.")
    split_and_save(examples, train_path, eval_path)
    logger.info(f"Data saved to {train_path} and {eval_path}.")


def run_eval_only(model_path: str, eval_path: str, tools_path: str) -> EvalResult:
    tools = load_tools(tools_path)
    model, tokenizer = load_model(model_path)
    return evaluate(model, tokenizer, eval_path, tools)
