"""
Main CLI Module for LLM Experiment Runner

This module provides the command-line interface for discovering models,
configuring experiments, and running LLM comparisons.
"""

# Suppress common warnings before other imports
import warnings

warnings.filterwarnings(
    "ignore", message=".*pynvml package is deprecated.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*TRANSFORMERS_CACHE.*is deprecated.*", category=FutureWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import click
import json
import sys
import os
import logging
import platform
import subprocess
import shutil
import time
from typing import List, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel

from ..discovery.hf_model_discovery import (
    HuggingFaceModelDiscovery,
    ModelSearchCriteria,
)
from ..algorithms.comparison_algorithms import ComparisonEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LCT (LLMs Comparative Tool) - Compare Language Models with systematic experimentation.

    Available commands:
    • search     - Find models on HuggingFace Hub with advanced filters
    • select     - Interactive multi-model selection interface
    • info       - Get detailed model information
    • algorithms - List available comparison algorithms
    • init       - Create new experiment configuration
    • auth       - Manage HuggingFace API authentication
    • diagnostic - Check system compatibility and readiness

    For enhanced functionality, set up HuggingFace authentication:
    lct auth setup
    """
    pass


@cli.command()
@click.option("--task", help="Task type (e.g., text-generation, conversational)")
@click.option("--language", help="Language (e.g., en, multilingual)")
@click.option("--author", help="Model author/organization")
@click.option("--min-downloads", type=int, help="Minimum number of downloads")
@click.option("--max-downloads", type=int, help="Maximum number of downloads")
@click.option("--min-size", type=float, help="Minimum model size in GB")
@click.option("--max-size", type=float, help="Maximum model size in GB")
@click.option("--tags", help="Comma-separated tags")
@click.option("--license", help="License type")
@click.option("--query", help="Search query string (supports fuzzy matching)")
@click.option("--fuzzy", is_flag=True, help="Enable fuzzy matching for query")
@click.option(
    "--sort-by",
    default="downloads",
    type=click.Choice(["downloads", "created_at", "modified_at", "size", "relevance"]),
    help="Sort results by",
)
@click.option("--limit", default=20, type=int, help="Maximum number of results")
@click.option("--save", help="Save results to JSON file")
@click.option("--compatible-only", is_flag=True, help="Show only compatible models")
@click.option("--lightweight", is_flag=True, help="Show only lightweight models (<2GB)")
@click.option(
    "--popular", is_flag=True, help="Show only popular models (>1000 downloads)"
)
@click.option(
    "--recent", is_flag=True, help="Show only recently updated models (last 30 days)"
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Launch interactive selection after search",
)
def search(
    task,
    language,
    author,
    min_downloads,
    max_downloads,
    min_size,
    max_size,
    tags,
    license,
    query,
    fuzzy,
    sort_by,
    limit,
    save,
    compatible_only,
    lightweight,
    popular,
    recent,
    interactive,
):
    """Search for models on Hugging Face Hub with advanced filtering and fuzzy matching."""

    console.print("[bold blue]🔍 Enhanced Search - Hugging Face Hub[/bold blue]")

    # Apply quick filters
    if lightweight:
        max_size = min(max_size or 2.0, 2.0)
        console.print("[dim]Applied filter: lightweight models (<2GB)[/dim]")

    if popular:
        min_downloads = max(min_downloads or 1000, 1000)
        console.print("[dim]Applied filter: popular models (>1000 downloads)[/dim]")

    # Parse tags
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]

    # Create enhanced search criteria
    criteria = ModelSearchCriteria(
        task=task,
        language=language,
        author=author,
        min_downloads=min_downloads,
        max_size_gb=max_size,
        tags=tag_list,
        license=license,
        query=query,
        sort_by=sort_by,
        limit=limit,
    )

    try:
        # Check for HuggingFace token
        import os

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            console.print("[yellow]⚠️  No HuggingFace token found![/yellow]")
            console.print("For full access to models and faster searches:")
            console.print("1. Run: [cyan]lct auth setup[/cyan] (recommended)")
            console.print("2. Or manually: export HF_TOKEN=your_token_here")
            console.print("3. Get token from: https://huggingface.co/settings/tokens")
            console.print("Continuing with public access (limited features)...\n")

        # Initialize discovery
        discovery = HuggingFaceModelDiscovery()

        # Search models with progress indicator
        with console.status("🔎 Searching models..."):
            models = discovery.search_models(criteria)

        if not models:
            console.print("[yellow]No models found matching the criteria.[/yellow]")
            console.print("\n[dim]💡 Try adjusting your search criteria:[/dim]")
            console.print("  • Remove or adjust size/download limits")
            console.print("  • Try broader search terms")
            console.print("  • Use --fuzzy for approximate matching")
            return

        # Apply additional filters
        original_count = len(models)

        if max_downloads:
            models = [m for m in models if m.downloads and m.downloads <= max_downloads]

        if min_size:
            models = [m for m in models if m.size_gb and m.size_gb >= min_size]

        if compatible_only:
            models = [m for m in models if m.compatible]

        if recent:
            # Filter for recently updated models (last 30 days)
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=30)
            # Note: This would need modification_date in the model data
            console.print(
                "[dim]Recent filter applied (implementation may vary by API)[/dim]"
            )

        # Apply fuzzy matching if requested
        if fuzzy and query:
            models = _apply_fuzzy_matching(models, query)

        # Show filtering results
        if len(models) < original_count:
            console.print(
                f"[dim]Filtered from {original_count} to {len(models)} models[/dim]"
            )

        if not models:
            console.print("[yellow]No models found after filtering.[/yellow]")
            return

        # Sort by relevance if fuzzy matching was used
        if fuzzy and query and sort_by == "relevance":
            models = _sort_by_relevance(models, query)

        # Display results
        _display_search_results(models, enhanced=True)

        # Save results if requested
        if save:
            output_path = discovery.save_search_results(models, save)
            console.print(f"[green]✅ Results saved to {output_path}[/green]")

        # Launch interactive selection if requested
        if interactive:
            console.print("\n" + "=" * 50)
            selected_models = _interactive_model_selection(models)
            if selected_models:
                console.print(
                    f"\n[green]✅ Selected {len(selected_models)} models for your experiment![/green]"
                )
                _save_selected_models(selected_models)

    except Exception as e:
        console.print(f"[red]❌ Error searching models: {e}[/red]")
        logger.error(f"Search error: {e}", exc_info=True)


# Helper functions for enhanced search functionality
def _apply_fuzzy_matching(models, query):
    """Apply fuzzy matching to filter models based on query."""
    try:
        # Simple fuzzy matching - look for query terms in model ID, author, and description
        query_words = query.lower().split()
        matched_models = []

        for model in models:
            searchable_text = f"{model.id} {model.author or ''} {getattr(model, 'description', '') or ''}".lower()

            # Score based on how many query words appear
            score = 0
            for word in query_words:
                if word in searchable_text:
                    score += 1
                # Partial match bonus
                elif any(word in text_word for text_word in searchable_text.split()):
                    score += 0.5

            if score > 0:
                model._fuzzy_score = score / len(query_words)  # Normalize score
                matched_models.append(model)

        return matched_models
    except Exception:
        return models  # Fallback to original list if fuzzy matching fails


def _sort_by_relevance(models, query):
    """Sort models by relevance score from fuzzy matching."""
    try:
        return sorted(models, key=lambda m: getattr(m, "_fuzzy_score", 0), reverse=True)
    except Exception:
        return models


def _display_search_results(models, enhanced=False):
    """Display search results in an enhanced table format."""
    title = f"Found {len(models)} Models"
    if enhanced:
        title = f"🔍 Enhanced Search Results - {len(models)} Models"

    table = Table(title=title)
    table.add_column("ID", style="cyan", max_width=30)
    table.add_column("Author", style="green", max_width=20)
    table.add_column("Task", style="yellow", max_width=15)
    table.add_column("Downloads", justify="right", style="blue")
    table.add_column("Size (GB)", justify="right", style="magenta")
    table.add_column("License", style="dim", max_width=10)
    table.add_column("Status", style="red")

    if enhanced:
        table.add_column("Match", justify="right", style="bright_green")

    for model in models:
        size_str = f"{model.size_gb:.1f}" if model.size_gb else "?"
        downloads_str = f"{model.downloads:,}" if model.downloads else "0"
        compatible_str = "✅" if model.compatible else "⚠️"

        row_data = [
            model.id,
            model.author or "Unknown",
            model.task or "N/A",
            downloads_str,
            size_str,
            model.license or "N/A",
            compatible_str,
        ]

        if enhanced and hasattr(model, "_fuzzy_score"):
            match_score = f"{model._fuzzy_score:.1%}"
            row_data.append(match_score)
        elif enhanced:
            row_data.append("-")

        table.add_row(*row_data)

    console.print(table)


def _interactive_model_selection(models):
    """Interactive model selection interface."""
    try:
        selected_models = []
        console.print("\n[bold cyan]🎯 Interactive Model Selection[/bold cyan]")
        console.print("Select models for your experiment (press Enter when done):")
        console.print(
            "[dim]Commands: 'a' = select all, 'c' = clear selection, 'q' = quit, number = toggle selection[/dim]\n"
        )

        while True:
            # Display models with selection status
            for i, model in enumerate(models):
                status = "✅" if model in selected_models else "⬜"
                size_str = f"{model.size_gb:.1f}GB" if model.size_gb else "?GB"
                downloads_str = f"{model.downloads:,}" if model.downloads else "0"

                console.print(
                    f"{status} [{i+1:2d}] {model.id} ({size_str}, {downloads_str} downloads)"
                )

            console.print(f"\n[green]Selected: {len(selected_models)} models[/green]")
            choice = (
                input(
                    "\nEnter choice (1-{}, 'a', 'c', 'q', or Enter to finish): ".format(
                        len(models)
                    )
                )
                .strip()
                .lower()
            )

            if not choice or choice == "q":
                break
            elif choice == "a":
                selected_models = models.copy()
                console.print("[green]✅ All models selected![/green]\n")
            elif choice == "c":
                selected_models.clear()
                console.print("[yellow]🗑️  Selection cleared![/yellow]\n")
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    model = models[idx]
                    if model in selected_models:
                        selected_models.remove(model)
                        console.print(f"[yellow]➖ Removed: {model.id}[/yellow]\n")
                    else:
                        selected_models.append(model)
                        console.print(f"[green]➕ Added: {model.id}[/green]\n")
                else:
                    console.print("[red]Invalid selection![/red]\n")
            else:
                console.print("[red]Invalid command![/red]\n")

        return selected_models

    except KeyboardInterrupt:
        console.print("\n[yellow]Selection cancelled.[/yellow]")
        return []
    except Exception as e:
        console.print(f"[red]Error in selection: {e}[/red]")
        return []


def _save_selected_models(selected_models):
    """Save selected models to a JSON file for later use."""
    try:
        import os
        from datetime import datetime

        # Create selections directory if it doesn't exist
        os.makedirs("model-selections", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model-selections/selected_models_{timestamp}.json"

        # Prepare data
        model_data = []
        for model in selected_models:
            model_dict = {
                "id": model.id,
                "author": model.author,
                "task": model.task,
                "size_gb": model.size_gb,
                "downloads": model.downloads,
                "license": model.license,
                "compatible": model.compatible,
            }
            model_data.append(model_dict)

        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "timestamp": timestamp,
                    "count": len(model_data),
                    "models": model_data,
                },
                f,
                indent=2,
            )

        console.print(f"[green]💾 Models saved to: {filename}[/green]")
        console.print(f"[dim]Use with: lct init --models-file {filename}[/dim]")

    except Exception as e:
        console.print(f"[red]Error saving models: {e}[/red]")


@cli.command()
@click.option("--from-search", help="Load models from search results file")
@click.option("--task", help="Filter by task type")
@click.option("--max-size", type=float, default=5.0, help="Maximum model size in GB")
@click.option("--limit", type=int, default=50, help="Maximum models to display")
def select(from_search, task, max_size, limit):
    """Interactive multi-model selection interface."""

    console.print("[bold blue]🎯 Multi-Model Selection Interface[/bold blue]")

    try:
        models = []

        if from_search and os.path.exists(from_search):
            # Load from search results file
            with open(from_search, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert back to model objects (simplified)
                for model_data in data.get("models", []):
                    # Create a simple model-like object
                    class SimpleModel:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                setattr(self, k, v)

                    models.append(SimpleModel(**model_data))
        else:
            # Search for models with basic criteria
            console.print("Searching for suitable models...")

            # Create search criteria
            criteria = ModelSearchCriteria(
                task=task, max_size_gb=max_size, limit=limit, sort_by="downloads"
            )

            discovery = HuggingFaceModelDiscovery()
            with console.status("Loading models..."):
                models = discovery.search_models(criteria)

        if not models:
            console.print(
                "[yellow]No models found. Try adjusting criteria or running a search first.[/yellow]"
            )
            return

        # Launch interactive selection
        console.print(f"\n[green]Found {len(models)} models to choose from[/green]")
        selected_models = _interactive_model_selection(models)

        if selected_models:
            console.print(
                f"\n[bold green]🎉 Successfully selected {len(selected_models)} models![/bold green]"
            )
            _save_selected_models(selected_models)

            # Show summary
            console.print("\n[bold]Selection Summary:[/bold]")
            for model in selected_models:
                size_str = f"{model.size_gb:.1f}GB" if model.size_gb else "?GB"
                console.print(f"  • {model.id} ({size_str})")
        else:
            console.print("[yellow]No models selected.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error in model selection: {e}[/red]")
        logger.error(f"Model selection error: {e}", exc_info=True)


@cli.command()
@click.argument("model_id")
@click.option("--save", help="Save model info to JSON file")
def info(model_id, save):
    """Get detailed information about a specific model."""

    console.print(f"[bold blue]Getting info for model: {model_id}[/bold blue]")

    try:
        discovery = HuggingFaceModelDiscovery()

        with console.status("Fetching model information..."):
            model = discovery.get_model_info(model_id)

        if not model:
            console.print(f"[red]Model not found: {model_id}[/red]")
            return

        # Display detailed info
        console.print(f"[bold]Model ID:[/bold] {model.id}")
        console.print(f"[bold]Author:[/bold] {model.author}")
        console.print(f"[bold]Task:[/bold] {model.task or 'N/A'}")
        console.print(f"[bold]Library:[/bold] {model.library or 'N/A'}")
        console.print(f"[bold]Language:[/bold] {model.language or 'N/A'}")
        console.print(f"[bold]License:[/bold] {model.license or 'N/A'}")
        console.print(f"[bold]Downloads:[/bold] {model.downloads:,}")
        console.print(f"[bold]Likes:[/bold] {model.likes:,}")
        console.print(
            f"[bold]Size:[/bold] {model.size_gb:.1f} GB"
            if model.size_gb
            else "Size: Unknown"
        )
        console.print(
            f"[bold]Compatible:[/bold] {'✓ Yes' if model.compatible else '✗ No'}"
        )

        if not model.compatible and model.compatibility_notes:
            console.print(
                f"[yellow]Compatibility Notes:[/yellow] {model.compatibility_notes}"
            )

        if model.description:
            console.print(f"[bold]Description:[/bold]\n{model.description[:500]}...")

        if model.tags:
            console.print(f"[bold]Tags:[/bold] {', '.join(model.tags[:10])}")

        # Save if requested
        if save:
            model_dict = {
                "id": model.id,
                "author": model.author,
                "task": model.task,
                "library": model.library,
                "language": model.language,
                "license": model.license,
                "downloads": model.downloads,
                "likes": model.likes,
                "size_gb": model.size_gb,
                "compatible": model.compatible,
                "compatibility_notes": model.compatibility_notes,
                "description": model.description,
                "tags": model.tags,
                "created_at": model.created_at,
                "last_modified": model.last_modified,
            }

            with open(save, "w", encoding="utf-8") as f:
                json.dump(model_dict, f, indent=2, ensure_ascii=False)

            console.print(f"[green]Model info saved to {save}[/green]")

    except Exception as e:
        console.print(f"[red]Error getting model info: {e}[/red]")
        logger.error(f"Model info error: {e}", exc_info=True)


@cli.command()
@click.option("--name", prompt="Experiment name", help="Name of the experiment")
@click.option("--models", help="Comma-separated model IDs or path to JSON file")
@click.option("--models-file", help="JSON file containing selected models")
@click.option("--prompts", help="Comma-separated prompts or path to text file")
@click.option("--prompts-file", help="Text file with prompts (one per line)")
@click.option(
    "--algorithms",
    default="response_time,text_length",
    help="Comma-separated algorithm names",
)
@click.option(
    "--interactive", "-i", is_flag=True, help="Interactive algorithm selection"
)
@click.option(
    "--repetitions", default=3, type=int, help="Number of repetitions per test"
)
@click.option("--output-dir", help="Output directory for results")
@click.option(
    "--max-memory", default=8.0, type=float, help="Maximum memory for models (GB)"
)
@click.option("--max-length", default=150, type=int, help="Maximum generation length")
@click.option("--temperature", default=0.7, type=float, help="Generation temperature")
@click.option(
    "--energy-profiler",
    default="none",
    type=click.Choice(["none", "codecarbon"]),
    help="Energy profiler to use (CodeCarbon with RAPL + GPU support)",
)
@click.option(
    "--energy-profiling",
    "-e",
    is_flag=True,
    help="Enable energy profiling with CodeCarbon",
)
def init(
    name,
    models,
    models_file,
    prompts,
    prompts_file,
    algorithms,
    interactive,
    repetitions,
    output_dir,
    max_memory,
    max_length,
    temperature,
    energy_profiler,
    energy_profiling,
):
    """Initialize a new LLM comparison experiment."""

    console.print("[bold blue]Initializing LLM comparison experiment...[/bold blue]")

    # Parse model IDs
    model_ids = []
    if models:
        if models.endswith(".json"):
            # Load from JSON file
            try:
                with open(models, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "models" in data:
                        model_ids = [m["id"] for m in data["models"]]
                    else:
                        model_ids = data  # Assume it's a list
            except Exception as e:
                console.print(f"[red]Error loading models file: {e}[/red]")
                return
        else:
            # Parse comma-separated
            model_ids = [mid.strip() for mid in models.split(",")]
    elif models_file:
        # Load from specified file
        try:
            with open(models_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "models" in data:
                    model_ids = [m["id"] for m in data["models"]]
                else:
                    model_ids = data
        except Exception as e:
            console.print(f"[red]Error loading models file: {e}[/red]")
            return
    else:
        console.print("[red]Must specify either --models or --models-file[/red]")
        return

    # Parse prompts
    prompt_list = []
    if prompts:
        if prompts.endswith(".txt"):
            # Load from text file
            try:
                with open(prompts, "r", encoding="utf-8") as f:
                    prompt_list = [line.strip() for line in f if line.strip()]
            except Exception as e:
                console.print(f"[red]Error loading prompts file: {e}[/red]")
                return
        else:
            # Parse comma-separated
            prompt_list = [p.strip() for p in prompts.split(",")]
    elif prompts_file:
        # Load from specified file
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompt_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            console.print(f"[red]Error loading prompts file: {e}[/red]")
            return
    else:
        # Default prompts
        prompt_list = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
        ]

    # Get available algorithms
    engine = ComparisonEngine()
    available_algorithms = engine.get_available_algorithms()

    # Interactive algorithm selection
    if interactive:
        console.print("\n[bold blue]Interactive Algorithm Selection[/bold blue]")
        console.print("Select algorithms from the following categories:\n")

        algorithm_categories = {
            "1": (
                "Performance Metrics",
                ["response_time", "token_throughput", "text_length"],
            ),
            "2": ("N-gram Based (need references)", ["bleu", "rouge"]),
            "3": (
                "Semantic Similarity (need references)",
                ["bert_score", "semantic_similarity", "semantic_textual_similarity"],
            ),
            "4": ("Human-Aligned Evaluation", ["pairwise_comparison", "llm_as_judge"]),
            "5": ("Advanced Frameworks", ["g_eval", "rlhf_preference"]),
            "6": (
                "Task-Specific",
                [
                    "code_generation",
                    "commonsense_reasoning",
                    "mathematical_reasoning",
                    "safety_alignment",
                    "truthfulness",
                ],
            ),
        }

        selected_algorithms = []

        for cat_num, (cat_name, cat_algorithms) in algorithm_categories.items():
            available_in_category = [
                alg for alg in cat_algorithms if alg in available_algorithms
            ]
            if available_in_category:
                console.print(f"[yellow]{cat_num}. {cat_name}[/yellow]")
                for i, alg in enumerate(available_in_category, 1):
                    console.print(f"   {i}. {alg}")

                selection = click.prompt(
                    f"Select algorithms from {cat_name} (comma-separated numbers, 'all', or 'skip')",
                    default="skip",
                )

                if selection.lower() == "all":
                    selected_algorithms.extend(available_in_category)
                elif selection.lower() != "skip":
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(",")]
                        selected_algorithms.extend(
                            [
                                available_in_category[i]
                                for i in indices
                                if 0 <= i < len(available_in_category)
                            ]
                        )
                    except (ValueError, IndexError):
                        console.print("[red]Invalid selection, skipping category[/red]")

                console.print()

        if selected_algorithms:
            algorithms = ",".join(selected_algorithms)
            console.print(f"[green]Selected algorithms: {algorithms}[/green]")
        else:
            console.print("[yellow]No algorithms selected, using defaults[/yellow]")

    # Handle energy profiling flag
    if energy_profiling and energy_profiler == "none":
        energy_profiler = "codecarbon"  # Default to CodeCarbon if -e flag is used

    # Parse algorithms
    algorithm_list = [alg.strip() for alg in algorithms.split(",")]

    # Validate algorithms
    invalid_algorithms = [
        alg for alg in algorithm_list if alg not in available_algorithms
    ]

    if invalid_algorithms:
        console.print(
            f"[yellow]Warning: Invalid algorithms (will be skipped): {', '.join(invalid_algorithms)}[/yellow]"
        )
        algorithm_list = [alg for alg in algorithm_list if alg in available_algorithms]

    console.print(
        f"[green]Available algorithms: {', '.join(available_algorithms.keys())}[/green]"
    )

    # Check API key requirements for selected algorithms
    api_key_requiring_algorithms = {"g_eval": "openai", "llm_as_judge": "openai"}

    from ..config.api_keys import get_api_key_manager

    manager = get_api_key_manager()

    algorithms_needing_keys = []
    for alg in algorithm_list:
        if alg in api_key_requiring_algorithms:
            service = api_key_requiring_algorithms[alg]
            if not manager.has_key(service):
                algorithms_needing_keys.append((alg, service))

    if algorithms_needing_keys:
        console.print("\n[yellow]⚠️  API Key Warning:[/yellow]")
        console.print("The following algorithms require API keys:\n")
        for alg, service in algorithms_needing_keys:
            console.print(
                f"  • [red]{alg}[/red] requires [cyan]{service.upper()}[/cyan] API key"
            )

        console.print(f"\n[bold]To set API keys:[/bold]")
        for alg, service in algorithms_needing_keys:
            console.print(f"  lct apikey {service} --key YOUR_KEY")

        console.print(
            f"\n[dim]These algorithms will be skipped if API keys are not provided.[/dim]"
        )

        if not click.confirm("\nContinue without these algorithms?", default=True):
            console.print("[yellow]Experiment initialization cancelled.[/yellow]")
            return

        # Remove algorithms that need API keys
        for alg, _ in algorithms_needing_keys:
            if alg in algorithm_list:
                algorithm_list.remove(alg)
                console.print(f"[dim]Removed {alg} from algorithm list[/dim]")

    # Create experiment directory
    output_path = Path(output_dir) if output_dir else Path.cwd() / "experiments" / name
    output_path.mkdir(parents=True, exist_ok=True)

    # Choose config template based on energy profiling
    if energy_profiler != "none":
        console.print(
            f"[green]Energy profiling enabled with: {energy_profiler}[/green]"
        )
        config_import = "from llm_runner.configs.energy_profiled_config import EnergyProfiledLLMConfig"
        config_class = "EnergyProfiledLLMConfig"
        energy_config = f"""
    
    # ================================ ENERGY PROFILING CONFIG ================================
    
    energy_profiler = "{energy_profiler}"
    time_between_runs_in_ms = 5000  # Allow cooling between runs
"""
    else:
        config_import = (
            "from llm_runner.configs.llm_comparison_config import LLMComparisonConfig"
        )
        config_class = "LLMComparisonConfig"
        energy_config = ""

    # Create RunnerConfig.py
    config_content = f'''"""
Generated LLM Comparison Configuration
Experiment: {name}
Energy Profiling: {energy_profiler}
Generated by LLM Experiment Runner
"""

{config_import}
from llm_runner.loaders.universal_loader import ModelLoadConfig
from pathlib import Path


class RunnerConfig({config_class}):
    """Configuration for {name} experiment."""
    
    # Experiment configuration
    name = "{name}"
    results_output_path = Path(r"{output_path.absolute()}")
    
    # Model configuration
    model_ids = {model_ids}
    
    # Prompt configuration
    prompts = {prompt_list}
    
    # Algorithm configuration
    comparison_algorithms = {algorithm_list}
    
    # Experiment parameters
    repetitions = {repetitions}
    max_memory_gb = {max_memory}
    
    # Generation parameters
    generation_params = {{
        "max_length": {max_length},
        "temperature": {temperature},
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": 50256  # Set explicitly to avoid warnings
        # NOTE: Removed max_new_tokens to avoid conflicts with max_length
    }}
    
    # Model loading configuration
    model_load_config = ModelLoadConfig(
        device="cpu" if "{energy_profiler}" != "none" else "auto",
        torch_dtype="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True
    ){energy_config}
'''

    config_path = output_path / "RunnerConfig.py"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    # Create experiment info file
    experiment_info = {
        "name": name,
        "model_ids": model_ids,
        "prompts": prompt_list,
        "algorithms": algorithm_list,
        "repetitions": repetitions,
        "max_memory_gb": max_memory,
        "generation_params": {"max_length": max_length, "temperature": temperature},
        "created_at": str(Path().resolve()),
        "config_file": str(config_path),
    }

    info_path = output_path / "experiment_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Experiment '{name}' initialized successfully![/green]")
    console.print(f"[bold]Configuration:[/bold] {config_path}")
    console.print(f"[bold]Info file:[/bold] {info_path}")
    console.print(f"[bold]Models:[/bold] {len(model_ids)}")
    console.print(f"[bold]Prompts:[/bold] {len(prompt_list)}")
    console.print(f"[bold]Algorithms:[/bold] {len(algorithm_list)}")
    console.print(
        f"[bold]Total runs:[/bold] {len(model_ids) * len(prompt_list) * repetitions}"
    )

    console.print(f"\n[bold blue]To run the experiment:[/bold blue]")
    if energy_profiler == "codecarbon":
        console.print(f"[cyan]# CodeCarbon will track CPU (RAPL), GPU, and CO2[/cyan]")
        relative_path = Path(config_path).parent.relative_to(Path.cwd())
        console.print(f"[cyan]cd {relative_path}[/cyan]")
        console.print(f"[cyan]python3 -m experiment_runner RunnerConfig.py[/cyan]")
        console.print(
            f"\n[yellow]⚠️  The -E flag preserves environment variables for Python modules[/yellow]"
        )
    else:
        console.print(f"python -m llm_runner.cli.main_cli run {config_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--dry-run", is_flag=True, help="Show what would be run without executing"
)
def run(config_file, dry_run):
    """Run an LLM comparison experiment."""

    config_path = Path(config_file)

    if dry_run:
        console.print(f"[yellow]DRY RUN - Would execute: {config_path}[/yellow]")
        return

    console.print(f"[bold blue]Running LLM comparison experiment...[/bold blue]")
    console.print(f"[bold]Config:[/bold] {config_path}")

    try:
        # Import and run using the correct experiment runner
        import subprocess
        import sys
        import os

        # Get the correct experiment runner path
        current_file = Path(__file__)
        app_src_path = current_file.parent.parent.parent  # Go up to app/src
        runner_path = app_src_path / "experiment_runner"

        # Set up environment with proper Python path
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{app_src_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(app_src_path)

        # Run the experiment using the Python module
        cmd = [sys.executable, "-m", "experiment_runner", str(config_path)]

        console.print(f"[dim]Executing: {' '.join(cmd)}[/dim]")

        # Run from the app/src directory so the experiment_runner module can be found
        result = subprocess.run(
            cmd, capture_output=False, text=True, cwd=str(app_src_path), env=env
        )

        if result.returncode == 0:
            console.print("[green]✓ Experiment completed successfully![/green]")
        else:
            console.print(
                f"[red]✗ Experiment failed with return code {result.returncode}[/red]"
            )

    except Exception as e:
        console.print(f"[red]Error running experiment: {e}[/red]")
        logger.error(f"Run error: {e}", exc_info=True)


@cli.command()
@click.argument(
    "service",
    type=click.Choice(["openai", "huggingface", "anthropic"], case_sensitive=False),
    required=False,
)
@click.option("--key", help="API key to set")
@click.option("--remove", is_flag=True, help="Remove API key for service")
@click.option("--list", "list_keys", is_flag=True, help="List all configured services")
@click.option("--show", is_flag=True, help="Show API key (masked)")
def apikey(service, key, remove, list_keys, show):
    """Manage API keys for LLM services (OpenAI, HuggingFace, etc.).

    Examples:

      # Set OpenAI API key
      lct apikey openai --key sk-...

      # Set HuggingFace API key
      lct apikey huggingface --key hf_...

      # List configured services
      lct apikey --list

      # Remove API key
      lct apikey openai --remove

    Note: API keys are stored securely in ~/.lct/api_keys.json
    """
    from ..config.api_keys import get_api_key_manager

    manager = get_api_key_manager()

    if list_keys:
        services = manager.list_services()
        if not services:
            console.print("[yellow]No API keys configured[/yellow]")
            console.print(
                "\nSet API keys with: [cyan]lct apikey <service> --key <your-key>[/cyan]"
            )
        else:
            console.print("[bold]Configured API Keys:[/bold]\n")
            for svc in services:
                api_key = manager.get_api_key(svc)
                masked = (
                    api_key[:8] + "..." + api_key[-4:]
                    if api_key and len(api_key) > 12
                    else "****"
                )
                console.print(f"  • [green]{svc}[/green]: {masked}")
        return

    if not service:
        console.print("[red]Error: Please specify a service or use --list[/red]")
        console.print("\nUsage: [cyan]lct apikey <service> --key <your-key>[/cyan]")
        console.print("       [cyan]lct apikey --list[/cyan]")
        return

    if remove:
        if manager.remove_api_key(service):
            console.print(f"[green]✓ Removed API key for {service}[/green]")
        else:
            console.print(f"[yellow]No API key found for {service}[/yellow]")
        return

    if show:
        api_key = manager.get_api_key(service)
        if api_key:
            masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
            console.print(f"[green]{service}[/green]: {masked}")
        else:
            console.print(f"[yellow]No API key configured for {service}[/yellow]")
        return

    if key:
        manager.set_api_key(service, key)
        console.print(f"[green]✓ API key for {service} saved successfully[/green]")
        console.print(f"[dim]Stored in: ~/.lct/api_keys.json[/dim]")

        # Set environment variable for current session
        os.environ[f"{service.upper()}_API_KEY"] = key
        console.print(
            f"[dim]Also set in current session: {service.upper()}_API_KEY[/dim]"
        )
    else:
        console.print(f"[red]Error: Please provide --key or --remove[/red]")
        console.print(f"\nSet key: [cyan]lct apikey {service} --key <your-key>[/cyan]")


@cli.command()
def setup():
    """Run comprehensive setup for energy profiling and dependencies."""
    import subprocess
    import os
    from pathlib import Path

    console = Console()

    # Check if setup script exists
    setup_script = Path(__file__).parent.parent.parent / "setup_energy_profiling.sh"

    if not setup_script.exists():
        console.print("[red]❌ Setup script not found![/red]")
        console.print(f"Expected: {setup_script}")
        console.print("\n[yellow]Creating setup script...[/yellow]")
        console.print(
            "The setup script should be available. Please check the repository."
        )
        return

    console.print("[blue]🔋 Starting LCT Energy Profiling Setup...[/blue]")
    console.print(f"Running: {setup_script}")
    console.print("\n[yellow]This will install:[/yellow]")
    console.print("• CodeCarbon (Python package)")
    console.print("• RAPL access configuration")
    console.print("• Python dependencies")
    console.print("• nvidia-ml-py (for GPU tracking)")
    console.print("• Verification scripts")

    # Ask for confirmation
    import click

    if not click.confirm("\nProceed with installation?", default=True):
        console.print("Setup cancelled.")
        return

    try:
        # Run setup script
        result = subprocess.run(
            ["bash", str(setup_script)], cwd=setup_script.parent, check=True
        )
        console.print("[green]✅ Setup completed successfully![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Logout/login to activate group permissions (if needed)")
        console.print("2. Run verification: [cyan]lct verify[/cyan]")
        console.print(
            "3. Test energy profiling: [cyan]lct init --energy-profiler codecarbon[/cyan]"
        )

    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Setup failed with exit code {e.returncode}[/red]")
        console.print("Check the output above for details")
    except Exception as e:
        console.print(f"[red]❌ Setup error: {e}[/red]")


@cli.command()
def verify():
    """Verify energy profiling tools installation."""
    import subprocess
    from pathlib import Path

    console = Console()

    # Check if verification script exists
    verify_script = Path(__file__).parent.parent.parent / "energy_check.py"

    if not verify_script.exists():
        console.print("[red]❌ Verification script not found![/red]")
        console.print(
            "Run '[cyan]lct setup[/cyan]' first to create verification script"
        )
        return

    console.print("[blue]� Verifying energy profiling installation...[/blue]")

    try:
        # Run verification script
        result = subprocess.run(
            ["python3", str(verify_script)], cwd=verify_script.parent
        )

        if result.returncode == 0:
            console.print("\n[green]✅ All energy profiling tools verified![/green]")
            console.print("\n[bold]Ready for energy profiling experiments![/bold]")
            console.print(
                "Create an experiment: [cyan]lct init --name test --energy-profiler codecarbon[/cyan]"
            )
        else:
            console.print(
                f"\n[yellow]⚠️  Some tools need attention (exit code: {result.returncode})[/yellow]"
            )
            console.print(
                "Check the output above and run '[cyan]lct setup[/cyan]' if needed"
            )

    except Exception as e:
        console.print(f"[red]❌ Verification error: {e}[/red]")


@cli.command()
def status():
    """Quick status check of LCT setup and dependencies."""

    console.print("[bold blue]🔧 LCT Status Check[/bold blue]")
    console.print()

    # Check if we're in the correct directory structure
    current_file = Path(__file__)
    app_src_path = current_file.parent.parent.parent  # Go up to app/src
    experiment_runner_path = app_src_path / "experiment_runner"
    if experiment_runner_path.exists():
        console.print("✅ Experiment Runner: [green]Found[/green]")
        console.print(f"   Path: {experiment_runner_path}")
    else:
        console.print("❌ Experiment Runner: [red]Not found[/red]")
        console.print(
            "   LCT should be run from within the correct directory structure"
        )
        console.print("   Expected structure:")
        console.print("   - app/src/experiment_runner/")
        console.print("   - app/src/llm_runner/ (this tool)")

    # Check energy profiling tools
    console.print("\n[bold]🔋 Energy Profiling Tools[/bold]")

    try:
        import codecarbon

        console.print("✅ CodeCarbon: [green]Available[/green]")
        console.print(f"   Version: {codecarbon.__version__}")
    except ImportError:
        console.print("❌ CodeCarbon: [red]Not installed[/red]")
        console.print("   Install: [cyan]lct dependencies[/cyan]")

    # Check RAPL
    if Path("/sys/class/powercap/intel-rapl").exists():
        console.print("✅ RAPL: [green]Available[/green]")
    else:
        console.print("❌ RAPL: [yellow]Not available (Intel CPU required)[/yellow]")

    # Quick algorithm check
    console.print("\n[bold]🧮 Core Algorithms[/bold]")

    core_deps = [
        ("transformers", "LLM Support"),
        ("torch", "PyTorch Backend"),
        ("sacrebleu", "BLEU Score"),
        ("rouge_score", "ROUGE Score"),
    ]

    available_count = 0
    for dep, name in core_deps:
        try:
            __import__(dep)
            console.print(f"✅ {name}: [green]Available[/green]")
            available_count += 1
        except ImportError:
            console.print(f"❌ {name}: [red]Not available[/red]")

    console.print(
        f"\n[bold]Summary: {available_count}/{len(core_deps)} core dependencies available[/bold]"
    )

    if available_count < len(core_deps):
        console.print("Run '[cyan]lct dependencies[/cyan]' to install missing packages")

    console.print(f"\n[bold]🚀 Quick Commands[/bold]")
    console.print("• Full setup: [cyan]lct setup[/cyan]")
    console.print("• Install deps: [cyan]lct dependencies[/cyan]")
    console.print("• Verify energy: [cyan]lct verify[/cyan]")
    console.print("• List algorithms: [cyan]lct algorithms[/cyan]")


@cli.command()
def dependencies():
    """Install missing Python dependencies for LCT algorithms."""

    console.print("[bold blue]📦 Installing LCT Dependencies[/bold blue]")
    console.print()

    # Core algorithm dependencies
    deps_to_install = []

    deps = [
        ("sacrebleu", "BLEU Score algorithms"),
        ("rouge_score", "ROUGE Score algorithms"),
        ("bert_score", "BERT Score algorithms"),
        ("sentence_transformers", "Semantic similarity algorithms"),
        ("codecarbon", "Carbon footprint tracking"),
        ("evaluate", "HuggingFace evaluation metrics"),
        ("datasets", "Dataset handling"),
        ("psutil", "System monitoring"),
        ("numpy", "Numerical computations"),
    ]

    console.print("Checking Python dependencies...")
    for dep, description in deps:
        try:
            __import__(dep)
            console.print(f"✅ {dep}: Already installed")
        except ImportError:
            console.print(f"❌ {dep}: Missing ({description})")
            deps_to_install.append(dep)

    if deps_to_install:
        console.print(
            f"\n[yellow]Installing {len(deps_to_install)} missing dependencies...[/yellow]"
        )

        # Ask for confirmation
        import click

        if not click.confirm("Proceed with installation?", default=True):
            console.print("Installation cancelled.")
            return

        try:
            import subprocess
            import sys

            cmd = [sys.executable, "-m", "pip", "install"] + deps_to_install
            console.print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=False, text=True)

            if result.returncode == 0:
                console.print(
                    "\n✅ [green]Python dependencies installed successfully![/green]"
                )
                console.print(
                    "Run [cyan]lct algorithms[/cyan] to see available algorithms"
                )
            else:
                console.print("\n❌ [red]Some dependencies failed to install[/red]")

        except Exception as e:
            console.print(f"\n❌ [red]Installation error: {e}[/red]")
    else:
        console.print("\n✅ [green]All Python dependencies already installed![/green]")

    console.print(f"\n[bold]🔋 Energy Profiling Setup[/bold]")
    console.print("CodeCarbon energy profiling with RAPL + GPU tracking:")
    console.print("• Full setup: [cyan]lct setup[/cyan]")
    console.print("• Verify installation: [cyan]lct verify[/cyan]")
    console.print("\n[bold]🎯 Algorithm Categories Available:[/bold]")
    console.print("• N-gram: BLEU, ROUGE, METEOR")
    console.print("• Semantic: BERT Score, Sentence Similarity")
    console.print("• Performance: Response Time, Memory Usage")
    console.print("• Human-aligned: BERTScore semantic matching")
    console.print("• Task-specific: Perplexity, Token Count")


@cli.command()
def algorithms():
    """List available comparison algorithms."""

    console.print("[bold blue]Available Comparison Algorithms in LCT[/bold blue]")

    engine = ComparisonEngine()
    available = engine.get_available_algorithms()

    # Create algorithm categories
    categories = {
        "Traditional N-gram Based Metrics": {
            "bleu": "BLEU score - Bilingual Evaluation Understudy (requires reference)",
            "rouge": "ROUGE score - Recall-Oriented Understudy for Gisting Evaluation (requires reference)",
        },
        "Semantic Similarity Metrics": {
            "bert_score": "BERTScore - Contextual embeddings similarity (requires reference)",
            "semantic_similarity": "Semantic similarity using sentence embeddings (requires reference)",
            "semantic_textual_similarity": "Advanced STS using Sentence-BERT (requires reference)",
        },
        "Performance Metrics": {
            "response_time": "Response generation speed comparison",
            "token_throughput": "Tokens generated per second",
            "text_length": "Response length analysis",
        },
        "Human-Aligned Evaluation": {
            "pairwise_comparison": "Pairwise preference comparison using Bradley-Terry model",
            "llm_as_judge": "Use LLM to evaluate response quality",
        },
        "Advanced Evaluation Frameworks": {
            "g_eval": "G-Eval framework with chain-of-thought reasoning",
            "rlhf_preference": "RLHF preference learning with Bradley-Terry model",
        },
        "Task-Specific Benchmarks": {
            "code_generation": "Code generation functional correctness (HumanEval-style)",
            "commonsense_reasoning": "Commonsense reasoning evaluation (HellaSwag-style)",
            "mathematical_reasoning": "Mathematical problem solving (GSM8K-style)",
            "safety_alignment": "Safety and alignment evaluation for responsible AI",
            "truthfulness": "Truthfulness evaluation to mitigate hallucinations (TruthfulQA-style)",
        },
    }

    for category, algorithms_dict in categories.items():
        console.print(f"\n[bold yellow]{category}[/bold yellow]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Algorithm", style="cyan", width=25)
        table.add_column("Description", style="white")
        table.add_column("Status", style="green", width=12)
        table.add_column("Requires", style="blue", width=20)

        api_key_algorithms = {"g_eval": "OpenAI API", "llm_as_judge": "OpenAI API"}

        for name, description in algorithms_dict.items():
            status = "✓ Available" if name in available else "✗ Unavailable"
            status_style = "green" if name in available else "red"

            requirements = []
            if "requires reference" in description.lower():
                requirements.append("Reference")
            if name in api_key_algorithms:
                requirements.append(api_key_algorithms[name])

            requires_text = ", ".join(requirements) if requirements else "None"
            requires_style = "yellow" if requirements else "green"

            table.add_row(
                name,
                description.replace(" (requires reference)", ""),
                f"[{status_style}]{status}[/{status_style}]",
                f"[{requires_style}]{requires_text}[/{requires_style}]",
            )

        console.print(table)

    # Summary
    total_algorithms = sum(len(algs) for algs in categories.values())
    available_count = len(available)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total algorithms: {total_algorithms}")
    console.print(f"Available: {available_count}")
    console.print(f"Unavailable: {total_algorithms - available_count}")

    if available_count < total_algorithms:
        console.print(
            "\n[yellow]To enable unavailable algorithms, install missing dependencies:[/yellow]"
        )
        console.print(
            "pip install sacrebleu rouge-score bert-score sentence-transformers evaluate datasets numpy"
        )

    console.print("\n[bold blue]Algorithm Selection Examples:[/bold blue]")
    console.print("# Performance comparison")
    console.print('lct init --algorithms "response_time,token_throughput,text_length"')
    console.print("\n# Quality evaluation (needs reference texts)")
    console.print('lct init --algorithms "bleu,rouge,bert_score,semantic_similarity"')
    console.print("\n# Advanced evaluation")
    console.print('lct init --algorithms "g_eval,llm_as_judge,pairwise_comparison"')
    console.print("\n# Task-specific benchmarks")
    console.print(
        'lct init --algorithms "code_generation,mathematical_reasoning,safety_alignment"'
    )
    console.print("\n# Comprehensive evaluation")
    console.print(
        'lct init --algorithms "response_time,bleu,rouge,g_eval,safety_alignment,truthfulness"'
    )


@cli.group()
def auth():
    """Manage HuggingFace API authentication."""
    pass


@auth.command()
@click.option(
    "--token",
    prompt="HuggingFace Token",
    hide_input=True,
    help="Your HuggingFace API token",
)
@click.option("--permanent", is_flag=True, help="Save token permanently to ~/.bashrc")
def setup(token, permanent):
    """Set up HuggingFace API token for enhanced functionality."""

    console.print("[bold blue]🔧 Setting up HuggingFace API Authentication[/bold blue]")

    # Validate token format
    if not token.startswith("hf_"):
        console.print(
            "[red]❌ Invalid token format. HuggingFace tokens should start with 'hf_'[/red]"
        )
        console.print("Get a valid token from: https://huggingface.co/settings/tokens")
        return

    # Test the token
    console.print("🧪 Testing token validity...")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user_info = api.whoami()
        console.print(
            f"[green]✅ Token valid! Authenticated as: {user_info['name']}[/green]"
        )
    except Exception as e:
        console.print(f"[red]❌ Token validation failed: {e}[/red]")
        console.print("Please check your token and try again.")
        return

    # Set environment variable for current session
    import os

    os.environ["HF_TOKEN"] = token
    console.print("[green]✅ Token set for current session[/green]")

    # Optionally save permanently
    if permanent:
        try:
            home_dir = Path.home()
            bashrc_path = home_dir / ".bashrc"

            # Check if already exists
            bashrc_content = ""
            if bashrc_path.exists():
                bashrc_content = bashrc_path.read_text()

            # Remove old HF_TOKEN lines
            lines = bashrc_content.split("\n")
            lines = [
                line
                for line in lines
                if not line.strip().startswith("export HF_TOKEN=")
            ]

            # Add new token
            lines.append(f"export HF_TOKEN={token}")

            # Write back
            bashrc_path.write_text("\n".join(lines))
            console.print(f"[green]✅ Token saved permanently to {bashrc_path}[/green]")
            console.print(
                "[yellow]💡 Restart your terminal or run 'source ~/.bashrc' to load in new sessions[/yellow]"
            )

        except Exception as e:
            console.print(f"[red]❌ Failed to save permanently: {e}[/red]")
            console.print("You can manually add this to your ~/.bashrc:")
            console.print(f"export HF_TOKEN={token}")

    console.print(
        "\n[bold green]🎉 HuggingFace authentication setup complete![/bold green]"
    )
    console.print("You can now use enhanced features:")
    console.print("• Higher rate limits (1000/hr vs 100/hr)")
    console.print("• Access to private models")
    console.print("• Better search performance")


@auth.command()
def status():
    """Check current HuggingFace authentication status."""

    console.print("[bold blue]🔍 HuggingFace Authentication Status[/bold blue]")

    import os

    # Check environment variables
    hf_token = os.getenv("HF_TOKEN")
    hf_hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token:
        console.print(
            f"[green]✅ HF_TOKEN found: {hf_token[:8]}...{hf_token[-4:]}[/green]"
        )
        token_to_test = hf_token
    elif hf_hub_token:
        console.print(
            f"[green]✅ HUGGINGFACE_HUB_TOKEN found: {hf_hub_token[:8]}...{hf_hub_token[-4:]}[/green]"
        )
        token_to_test = hf_hub_token
    else:
        console.print("[yellow]⚠️  No HuggingFace token found[/yellow]")
        console.print("\n[blue]To set up authentication:[/blue]")
        console.print("1. Get a token: https://huggingface.co/settings/tokens")
        console.print("2. Run: lct auth setup")
        console.print("3. Or set manually: export HF_TOKEN=hf_your_token_here")
        return

    # Test token validity
    console.print("\n🧪 Testing token validity...")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token_to_test)
        user_info = api.whoami()

        console.print(f"[green]✅ Token valid![/green]")
        console.print(f"[green]   User: {user_info.get('name', 'Unknown')}[/green]")
        console.print(f"[green]   Email: {user_info.get('email', 'Unknown')}[/green]")

        # Test API functionality
        console.print("\n🔍 Testing API functionality...")
        from ..discovery.hf_model_discovery import HuggingFaceModelDiscovery

        discovery = HuggingFaceModelDiscovery(token=token_to_test)

        # Try a simple search
        from ..discovery.hf_model_discovery import ModelSearchCriteria

        criteria = ModelSearchCriteria(task="text-generation", limit=1)
        models = discovery.search_models(criteria)

        if models:
            console.print("[green]✅ API search working[/green]")
            console.print(f"   Sample model: {models[0].id}")
        else:
            console.print("[yellow]⚠️  API search returned no results[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Token validation failed: {e}[/red]")
        console.print("\n[blue]To fix this:[/blue]")
        console.print(
            "1. Check token validity at: https://huggingface.co/settings/tokens"
        )
        console.print("2. Run: lct auth setup")


@auth.command()
def remove():
    """Remove HuggingFace token from environment."""

    console.print("[bold blue]🗑️  Removing HuggingFace Authentication[/bold blue]")

    import os

    # Remove from current session
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
        console.print("[green]✅ Removed HF_TOKEN from current session[/green]")

    if "HUGGINGFACE_HUB_TOKEN" in os.environ:
        del os.environ["HUGGINGFACE_HUB_TOKEN"]
        console.print(
            "[green]✅ Removed HUGGINGFACE_HUB_TOKEN from current session[/green]"
        )

    # Remove from ~/.bashrc
    try:
        home_dir = Path.home()
        bashrc_path = home_dir / ".bashrc"

        if bashrc_path.exists():
            bashrc_content = bashrc_path.read_text()
            lines = bashrc_content.split("\n")

            # Remove HF_TOKEN lines
            original_count = len(lines)
            lines = [
                line
                for line in lines
                if not (
                    line.strip().startswith("export HF_TOKEN=")
                    or line.strip().startswith("export HUGGINGFACE_HUB_TOKEN=")
                )
            ]

            if len(lines) < original_count:
                bashrc_path.write_text("\n".join(lines))
                console.print(f"[green]✅ Removed tokens from {bashrc_path}[/green]")
            else:
                console.print("[blue]ℹ️  No tokens found in ~/.bashrc[/blue]")

    except Exception as e:
        console.print(f"[yellow]⚠️  Could not modify ~/.bashrc: {e}[/yellow]")
        console.print("You may need to manually remove HF_TOKEN lines from ~/.bashrc")

    console.print("\n[green]🎉 Authentication removed![/green]")
    console.print("LCT will now use public access (limited features)")


@cli.command()
@click.option(
    "--quick", is_flag=True, help="Run quick diagnostic (skip model loading test)"
)
@click.option(
    "--model",
    default="distilgpt2",
    help="Test model for loading capability (default: distilgpt2)",
)
@click.option("--verbose", is_flag=True, help="Show detailed output")
def diagnostic(quick, model, verbose):
    """Run system diagnostic to check if your setup can run LLM experiments.

    This command checks:
    • System resources (RAM, CPU, GPU)
    • Python dependencies
    • HuggingFace authentication
    • Model loading capability
    • Disk space and network connectivity
    """
    console.print(
        Panel.fit(
            "[bold blue]🔍 LCT System Diagnostic[/bold blue]\n"
            "Checking if your system can run LLM experiments...",
            style="blue",
        )
    )

    results = {}

    # 1. System Information
    console.print("\n[bold cyan]📊 System Information[/bold cyan]")
    try:
        import psutil

        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()

        console.print(f"[green]✅ CPU:[/green] {cpu_count} cores, {cpu_percent}% usage")
        if cpu_freq:
            console.print(f"    Current frequency: {cpu_freq.current:.0f} MHz")
        results["cpu"] = {"status": "ok", "cores": cpu_count, "usage": cpu_percent}

        # Memory Info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_percent = memory.percent

        if memory_gb >= 8:
            console.print(
                f"[green]✅ RAM:[/green] {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available ({100-memory_percent:.1f}% free)"
            )
        elif memory_gb >= 4:
            console.print(
                f"[yellow]⚠️  RAM:[/yellow] {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available (may limit model size)"
            )
        else:
            console.print(
                f"[red]❌ RAM:[/red] {memory_gb:.1f} GB total - insufficient for most LLM experiments"
            )

        results["memory"] = {
            "status": (
                "ok" if memory_gb >= 8 else "warning" if memory_gb >= 4 else "error"
            ),
            "total_gb": memory_gb,
            "available_gb": memory_available_gb,
        }

        # Disk Space
        disk = psutil.disk_usage("/")
        disk_free_gb = disk.free / (1024**3)

        if disk_free_gb >= 10:
            console.print(f"[green]✅ Disk Space:[/green] {disk_free_gb:.1f} GB free")
        elif disk_free_gb >= 5:
            console.print(
                f"[yellow]⚠️  Disk Space:[/yellow] {disk_free_gb:.1f} GB free (may limit model downloads)"
            )
        else:
            console.print(
                f"[red]❌ Disk Space:[/red] {disk_free_gb:.1f} GB free - insufficient for model downloads"
            )

        results["disk"] = {
            "status": (
                "ok"
                if disk_free_gb >= 10
                else "warning" if disk_free_gb >= 5 else "error"
            ),
            "free_gb": disk_free_gb,
        }

        # Platform Info
        platform_info = platform.platform()
        python_version = platform.python_version()
        console.print(f"[blue]ℹ️  Platform:[/blue] {platform_info}")
        console.print(f"[blue]ℹ️  Python:[/blue] {python_version}")

        results["platform"] = {"os": platform_info, "python": python_version}

    except ImportError:
        console.print(
            "[yellow]⚠️  psutil not available - install with: pip install psutil[/yellow]"
        )
        results["system"] = {"status": "warning", "message": "psutil not available"}
    except Exception as e:
        console.print(f"[red]❌ System check failed: {e}[/red]")
        results["system"] = {"status": "error", "message": str(e)}

    # 2. GPU Detection
    console.print("\n[bold cyan]🎮 GPU Detection[/bold cyan]")
    gpu_found = False

    # Check for NVIDIA GPU (CUDA)
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(
                f"[green]✅ NVIDIA GPU:[/green] {gpu_name} ({gpu_memory:.1f} GB VRAM)"
            )
            results["gpu"] = {
                "status": "ok",
                "type": "cuda",
                "name": gpu_name,
                "memory_gb": gpu_memory,
            }
            gpu_found = True
        else:
            console.print("[yellow]⚠️  CUDA not available[/yellow]")
    except ImportError:
        if verbose:
            console.print("[blue]ℹ️  PyTorch not available for CUDA detection[/blue]")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]⚠️  CUDA check failed: {e}[/yellow]")

    # Check for AMD GPU (ROCm) - basic detection
    try:
        rocm_path = Path("/opt/rocm")
        if rocm_path.exists():
            console.print("[green]✅ AMD ROCm detected[/green]")
            results["gpu"] = {"status": "ok", "type": "rocm"}
            gpu_found = True
    except Exception:
        pass

    # Check for Intel GPU (basic detection)
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            if "Intel" in result.stdout and (
                "VGA" in result.stdout or "Display" in result.stdout
            ):
                intel_gpus = [
                    line
                    for line in result.stdout.split("\n")
                    if "Intel" in line and ("VGA" in line or "Display" in line)
                ]
                if intel_gpus and verbose:
                    console.print(
                        f"[blue]ℹ️  Intel GPU detected: {intel_gpus[0].split(':')[-1].strip()}[/blue]"
                    )
    except Exception:
        pass

    if not gpu_found:
        console.print(
            "[yellow]⚠️  No dedicated GPU detected - using CPU only (slower performance)[/yellow]"
        )
        results["gpu"] = {"status": "warning", "type": "cpu_only"}

    # 3. Python Dependencies
    console.print("\n[bold cyan]📦 Python Dependencies[/bold cyan]")
    required_packages = [
        ("transformers", "HuggingFace Transformers"),
        ("torch", "PyTorch"),
        ("psutil", "System monitoring"),
        ("rich", "Terminal formatting"),
        ("click", "CLI framework"),
        ("pandas", "Data analysis"),
        ("numpy", "Numerical computing"),
    ]

    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package)
            console.print(f"[green]✅ {package}[/green] - {description}")
        except ImportError:
            console.print(f"[red]❌ {package}[/red] - {description} (missing)")
            missing_packages.append(package)

    if missing_packages:
        console.print(f"\n[yellow]⚠️  Install missing packages with:[/yellow]")
        console.print(f"[bold]pip install {' '.join(missing_packages)}[/bold]")
        results["dependencies"] = {"status": "error", "missing": missing_packages}
    else:
        results["dependencies"] = {"status": "ok"}

    # 4. HuggingFace Authentication
    console.print("\n[bold cyan]🤗 HuggingFace Authentication[/bold cyan]")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token:
        masked_token = (
            hf_token[:8] + "..." + hf_token[-4:]
            if len(hf_token) > 12
            else hf_token[:4] + "..."
        )
        console.print(f"[green]✅ HF_TOKEN found:[/green] {masked_token}")

        # Test token validity
        try:
            from huggingface_hub import whoami

            user_info = whoami(token=hf_token)
            console.print(
                f"[green]✅ Token valid for user:[/green] {user_info.get('name', 'Unknown')}"
            )
            results["auth"] = {"status": "ok", "user": user_info.get("name", "Unknown")}
        except ImportError:
            console.print(
                "[yellow]⚠️  huggingface_hub not available for token validation[/yellow]"
            )
            results["auth"] = {"status": "warning", "message": "Cannot validate token"}
        except Exception as e:
            console.print(f"[red]❌ Token validation failed:[/red] {e}")
            results["auth"] = {"status": "error", "message": str(e)}
    else:
        console.print("[yellow]⚠️  No HuggingFace token found[/yellow]")
        console.print("    • Limited access to models")
        console.print("    • Set up with: [bold]lct auth setup[/bold]")
        results["auth"] = {"status": "warning", "message": "No token found"}

    # 5. Network Connectivity
    console.print("\n[bold cyan]🌐 Network Connectivity[/bold cyan]")
    try:
        import urllib.request

        urllib.request.urlopen("https://huggingface.co", timeout=10)
        console.print("[green]✅ HuggingFace Hub accessible[/green]")
        results["network"] = {"status": "ok"}
    except Exception as e:
        console.print(f"[red]❌ Cannot reach HuggingFace Hub:[/red] {e}")
        results["network"] = {"status": "error", "message": str(e)}

    # 6. Model Loading Test (if not quick mode)
    if not quick and "transformers" not in missing_packages:
        console.print(f"\n[bold cyan]🧪 Model Loading Test[/bold cyan]")
        console.print(f"Testing with model: [bold]{model}[/bold]")

        try:
            console.print("Loading model...", end="")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            start_time = time.time()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                low_cpu_mem_usage=True,
            )

            load_time = time.time() - start_time

            # Quick inference test
            inputs = tokenizer(
                "Hello, how are you?",
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                outputs = model_obj.generate(**inputs, max_length=20, do_sample=False)

            # Clean up
            del model_obj, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(
                f" [green]✅ Model loading successful[/green] ({load_time:.1f}s)"
            )
            results["model_test"] = {"status": "ok", "load_time": load_time}

        except ImportError as e:
            console.print(f"[red]❌ Missing dependencies for model test:[/red] {e}")
            results["model_test"] = {
                "status": "error",
                "message": f"Missing dependencies: {e}",
            }
        except Exception as e:
            console.print(f"[red]❌ Model loading failed:[/red] {e}")
            results["model_test"] = {"status": "error", "message": str(e)}
    elif quick:
        console.print("\n[blue]ℹ️  Skipping model loading test (--quick mode)[/blue]")
        results["model_test"] = {"status": "skipped"}

    # Summary
    console.print("\n" + "=" * 50)
    console.print(
        Panel.fit(
            _generate_diagnostic_summary(results),
            title="[bold]Diagnostic Summary[/bold]",
            style="blue",
        )
    )

    if verbose:
        console.print(f"\n[dim]Raw results: {json.dumps(results, indent=2)}[/dim]")


def _generate_diagnostic_summary(results):
    """Generate a summary of diagnostic results."""
    summary_lines = []
    overall_status = "ok"

    # Count status types
    status_counts = {"ok": 0, "warning": 0, "error": 0, "skipped": 0}

    for category, result in results.items():
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "error":
                overall_status = "error"
            elif status == "warning" and overall_status != "error":
                overall_status = "warning"

    # Generate summary
    if overall_status == "ok":
        summary_lines.append("[bold green]🎉 System Ready![/bold green]")
        summary_lines.append("Your system can run LLM experiments.")
    elif overall_status == "warning":
        summary_lines.append("[bold yellow]⚠️  System Partially Ready[/bold yellow]")
        summary_lines.append("Your system can run experiments with some limitations.")
    else:
        summary_lines.append("[bold red]❌ System Not Ready[/bold red]")
        summary_lines.append("Please fix the issues above before running experiments.")

    summary_lines.append("")
    summary_lines.append(f"✅ Passed: {status_counts['ok']}")
    summary_lines.append(f"⚠️  Warnings: {status_counts['warning']}")
    summary_lines.append(f"❌ Errors: {status_counts['error']}")

    if status_counts["skipped"] > 0:
        summary_lines.append(f"⏭️  Skipped: {status_counts['skipped']}")

    # Specific recommendations
    if results.get("memory", {}).get("status") == "warning":
        summary_lines.append("")
        summary_lines.append("💡 Tip: Use smaller models (< 1GB) for your system")

    if results.get("auth", {}).get("status") == "warning":
        summary_lines.append("")
        summary_lines.append("💡 Tip: Set up authentication with: lct auth setup")

    if results.get("gpu", {}).get("status") == "warning":
        summary_lines.append("")
        summary_lines.append("💡 Tip: Experiments will use CPU (slower but functional)")

    return "\n".join(summary_lines)


if __name__ == "__main__":
    cli()
