import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@click.group()
def cli():
    """Grubbot: Autonomous loop for tool-use finetuning."""
    pass

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--model', required=True, help="HuggingFace model to fine-tune")
@click.option('--provider', default='gemini', show_default=True, help="Provider for synthetic data generation (gemini, groq, ollama, mock)")
def run(tools, goal, model, provider):
    """Run the complete 4-stage pipeline."""
    click.echo(f"Starting Grubbot pipeline with model {model} (provider={provider})...")
    from grubbot.pipeline import run_full_pipeline
    run_full_pipeline(tools, goal, model, provider_name=provider)

@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help="Path to finetuned model")
@click.option('--data', type=click.Path(exists=True), required=True, help="Path to eval.jsonl")
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
def eval(model, data, tools):
    """Run evaluation only."""
    click.echo(f"Evaluating model {model} on {data}...")
    from grubbot.pipeline import run_eval_only

    result = run_eval_only(model, data, tools)
    click.echo(f"Overall Accuracy: {result.overall_accuracy * 100:.2f}%")
    for tool_name, acc in result.per_tool_accuracy.items():
        click.echo(f"  {tool_name}: {acc * 100:.2f}%")
    click.echo(f"Failures: {len(result.failures)}")

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--model', type=click.Path(exists=True), required=True, help="Path to current model checkpoint")
@click.option('--provider', default='gemini', show_default=True, help="Provider for targeted data generation (gemini, groq, ollama, mock)")
def loop(tools, goal, model, provider):
    """Resume the retrain loop from a checkpoint."""
    click.echo(f"Resuming loop with model {model} (provider={provider})...")
    from grubbot.config import load_tools, load_goal_from_markdown, GrubbotConfig
    from grubbot.providers import get_provider
    from grubbot.loop import run_loop

    tool_defs = load_tools(tools)
    goal_config = load_goal_from_markdown(goal)
    config = GrubbotConfig(tools=tool_defs, goal=goal_config, model_name=model, provider=provider)
    provider_impl = get_provider(provider)

    result = run_loop(config=config, provider=provider_impl, start_model_path=model)
    click.echo(f"Loop complete after {result.iterations} iteration(s).")
    click.echo(f"Final Accuracy: {result.final_accuracy * 100:.2f}%")
    for tool_name, acc in result.per_tool_accuracy.items():
        click.echo(f"  {tool_name}: {acc * 100:.2f}%")

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--provider', default="gemini", show_default=True, help="LLM Provider to use (gemini, groq, ollama, mock)")
@click.option('--count', default=50, help="Number of examples per tool")
def datagen(tools, goal, provider, count):
    """Run data generation (Stage 1) only."""
    from grubbot.config import load_tools, load_goal_from_markdown
    from grubbot.providers import get_provider
    from grubbot.datagen import generate_examples, split_and_save
    import os
    
    click.echo(f"Starting Data Generation with provider {provider}...")
    tool_defs = load_tools(tools)
    goal_config = load_goal_from_markdown(goal)
    llm = get_provider(provider)
    
    click.echo(f"Loaded {len(tool_defs)} tools.")
    examples = generate_examples(tool_defs, goal_config, llm, count_per_tool=count)
    
    click.echo(f"Generated {len(examples)} total examples. Splitting into train and eval...")
    
    os.makedirs("data", exist_ok=True)
    split_and_save(examples, "data/train.jsonl", "data/eval.jsonl")
    click.echo("Done! Wrote realistic data to data/train.jsonl and data/eval.jsonl")

if __name__ == '__main__':
    cli()
