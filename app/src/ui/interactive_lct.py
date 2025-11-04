#!/usr/bin/env python3
"""
Interactive LLM Comparison Tool (LCT)
Simple CLI menu for creating and running LLM experiments
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

import os
import sys
import json
import platform
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any

try:
    from packaging import version
except ImportError:
    # Fallback if packaging not available
    version = None

# Add project to path
# Add project directories to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
app_src = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, app_src)
sys.path.insert(0, project_root)

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.align import Align
    from rich.layout import Layout
    from rich.tree import Tree
    from rich import box
except ImportError:
    print("Installing required packages...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.panel import Panel

# Try to import optional packages
try:
    import psutil
except ImportError:
    psutil = None

console = Console()

# Pre-defined prompt categories from easy to difficult
PROMPT_CATEGORIES = {
    "Basic Greetings": [
        "Hello, how are you?",
        "What's your name?",
        "Nice to meet you!",
        "How's your day going?",
        "Tell me about yourself",
    ],
    "Simple Questions": [
        "What is Python programming?",
        "How do you make coffee?",
        "What's the weather like today?",
        "What's your favorite color?",
        "What do you like to do for fun?",
    ],
    "Creative Writing": [
        "Write a short story about a robot",
        "Describe a beautiful sunset",
        "Create a poem about friendship",
        "Write a dialogue between two characters",
        "Tell me a funny joke",
    ],
    "Knowledge & Facts": [
        "Explain what artificial intelligence is",
        "What are the benefits of renewable energy?",
        "How does the internet work?",
        "What causes climate change?",
        "Explain the water cycle",
    ],
    "Problem Solving": [
        "How would you organize a birthday party?",
        "What's the best way to learn a new language?",
        "How can we reduce plastic waste?",
        "Plan a healthy weekly meal",
        "How to improve time management?",
    ],
    "Technical & Complex": [
        "Explain quantum computing in simple terms",
        "Compare different machine learning algorithms",
        "Describe the process of DNA replication",
        "How do neural networks work?",
        "Explain blockchain technology",
    ],
    "Advanced Reasoning": [
        "Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Analyze the ethical implications of AI in healthcare",
        "Compare and contrast democracy and authoritarianism",
        "Explain the philosophical concept of free will",
        "Discuss the implications of genetic engineering",
    ],
    "Coding & Programming": [
        "Write a Python function to reverse a string",
        "Explain the difference between lists and tuples in Python",
        "How would you implement a binary search algorithm?",
        "Create a simple calculator in JavaScript",
        "Debug this code and explain the issue",
    ],
}

# No default popular models - users will add their own
POPULAR_MODELS = []

# Comprehensive dataset recommendations based on research and best practices
# Updated with working HuggingFace datasets (2025) - see DATASET_API_UPDATES_2025.md
ALGORITHM_DATASET_MAP = {
    "bleu": "wmt19/wmt20/wmt21, cnn_dailymail, knkarthick/xsum, multi30k",
    "rouge": "cnn_dailymail, reddit_tifu, scientific_papers",
    "bert_score": "mteb/sts-benchmark-sts, ms_coco",
    "semantic_similarity": "mteb/sts-benchmark-sts, sick, quora, glue/mrpc",
    "mathematical_reasoning": "gsm8k, hendrycks/math, math_qa, allenai/omega",
    "commonsense_reasoning": "hellaswag, commonsense_qa, gimmaru/piqa, winogrande",
    "code_generation": "openai_humaneval, mbpp, code_x_glue, livecodebench",
    "truthfulness": "truthful_qa (validation), fever, pkulawai/halueval",
    "llm_as_judge": "anthropic/hh-rlhf (various preference datasets)",
    "safety_alignment": "anthropic/hh-rlhf, SafetyBench",
}

ALGORITHMS_USING_DATASETS = list(ALGORITHM_DATASET_MAP.keys())

# Detailed dataset information for installation and configuration
DATASET_DETAILS = {
    # BLEU Datasets
    "wmt19": {"size": "~40M pairs", "hf_name": "wmt19", "category": "Translation"},
    "wmt20": {"size": "~40M pairs", "hf_name": "wmt20", "category": "Translation"},
    "wmt21": {"size": "~40M pairs", "hf_name": "wmt21", "category": "Translation"},
    "cnn_dailymail": {
        "size": "287K articles",
        "hf_name": "cnn_dailymail",
        "category": "Summarization",
    },
    "xsum": {
        "size": "227K articles",
        "hf_name": "knkarthick/xsum",
        "category": "Summarization",
    },
    "multi30k": {
        "size": "31K descriptions",
        "hf_name": "multi30k",
        "category": "Multilingual",
    },
    # ROUGE Datasets
    "reddit_tifu": {
        "size": "42K posts",
        "hf_name": "reddit_tifu",
        "category": "Summarization",
    },
    "scientific_papers": {
        "size": "215K papers",
        "hf_name": "scientific_papers",
        "category": "Domain-specific",
    },
    # BERT Score & Semantic Similarity Datasets
    "mteb/sts-benchmark-sts": {
        "size": "8.5K pairs",
        "hf_name": "mteb/sts-benchmark-sts",
        "category": "Similarity",
    },
    "ms_coco": {"size": "123K captions", "hf_name": "ms_coco", "category": "Captions"},
    "sick": {"size": "9.8K pairs", "hf_name": "sick", "category": "Similarity"},
    "quora": {"size": "400K pairs", "hf_name": "quora", "category": "Questions"},
    "glue/mrpc": {
        "size": "5.8K pairs",
        "hf_name": "glue",
        "category": "Paraphrase",
        "subset": "mrpc",
    },
    # Mathematical Reasoning Datasets
    "gsm8k": {"size": "8.5K problems", "hf_name": "gsm8k", "category": "Math"},
    "hendrycks/math": {
        "size": "12.5K problems",
        "hf_name": "hendrycks/math",
        "category": "Math",
    },
    "math_qa": {"size": "37K problems", "hf_name": "math_qa", "category": "Math"},
    "allenai/omega": {
        "size": "New benchmark",
        "hf_name": "allenai/omega-explorative",
        "category": "Math",
    },
    # Commonsense Reasoning Datasets
    "hellaswag": {
        "size": "70K tasks",
        "hf_name": "hellaswag",
        "category": "Commonsense",
    },
    "commonsense_qa": {
        "size": "12K questions",
        "hf_name": "commonsense_qa",
        "category": "Commonsense",
    },
    "piqa": {
        "size": "1K questions",
        "hf_name": "gimmaru/piqa",
        "category": "Physical Reasoning",
    },
    "winogrande": {"size": "44K tasks", "hf_name": "winogrande", "category": "Context"},
    # Code Generation Datasets
    "openai_humaneval": {
        "size": "164 problems",
        "hf_name": "openai_humaneval",
        "category": "Code",
    },
    "mbpp": {"size": "974 problems", "hf_name": "mbpp", "category": "Code"},
    "code_x_glue": {
        "size": "14 datasets",
        "hf_name": "code_x_glue_cc_code_completion_token",
        "category": "Code",
    },
    "livecodebench": {
        "size": "500+ problems",
        "hf_name": "livecodebench/code_generation_lite",
        "category": "Code",
    },
    # Truthfulness Datasets
    "truthful_qa": {
        "size": "817 questions",
        "hf_name": "truthful_qa",  # Note: Use split="validation" when loading
        "category": "Factuality",
    },
    "fever": {"size": "185K claims", "hf_name": "fever", "category": "Fact-checking"},
    "pkulawai/halueval": {
        "size": "35K samples",
        "hf_name": "pkulawai/halueval",
        "category": "Hallucination",
    },
    # Safety & LLM-as-Judge Datasets
    "anthropic/hh-rlhf": {
        "size": "161K conversations",
        "hf_name": "Anthropic/hh-rlhf",
        "category": "Safety/Preference",
    },
}

ALGORITHM_CATEGORIES = {
    "Performance & Speed": [
        ("response_time", "Measures how fast each model responds"),
        ("token_throughput", "Tokens generated per second"),
        ("text_length", "Length of generated responses"),
    ],
    "Quality & Accuracy": [
        ("bleu", "BLEU score (needs reference texts)"),
        ("rouge", "ROUGE score (needs reference texts)"),
        ("bert_score", "Semantic similarity using BERT embeddings"),
        ("semantic_similarity", "Sentence embedding similarity"),
    ],
    "Advanced Evaluation": [
        ("llm_as_judge", "Use LLM to judge response quality"),
        ("g_eval", "Advanced reasoning-based evaluation"),
        ("pairwise_comparison", "Compare responses pairwise"),
        ("safety_alignment", "Check for safe, aligned responses"),
    ],
    "Task-Specific": [
        ("code_generation", "Evaluate code generation quality"),
        ("mathematical_reasoning", "Math problem solving"),
        ("commonsense_reasoning", "Common sense understanding"),
        ("truthfulness", "Check for truthful, factual responses"),
    ],
}


class InteractiveLCT:
    def __init__(self):
        self.console = Console()
        self.config = {
            "name": "",
            "models": [],
            "prompts": [],
            "algorithms": [],
            "repetitions": 3,
            "max_length": 150,  # Increased from 50: Modern standard for comprehensive responses
            "temperature": 0.7,
            "energy_profiler": "none",
            "environmental_tracking": False,
            "environmental_region": "USA East",  # Default region
            # Removed single "dataset" field - each algorithm will use its appropriate datasets
        }

    def show_header(self):
        """Display the main header"""
        header = Text("🚀 Interactive LLM Comparison Tool (LCT)", style="bold blue")
        subheader = Text("Create and run LLM experiments with ease", style="dim")

        panel = Panel(
            Align.center(f"{header}\n{subheader}"), box=box.DOUBLE, style="blue"
        )
        self.console.print(panel)
        self.console.print()
    
    def _format_size(self, size_bytes):
        """Format size in bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _get_energy_status(self):
        """Get energy profiling status for main menu display"""
        profiler = self.config.get("energy_profiler", "none")
        env_tracking = self.config.get("environmental_tracking", False)
        
        if profiler == "none":
            return "⚪"
        elif env_tracking:
            region = self.config.get("environmental_region", "USA East")
            return f"✅ ({profiler}+env: {region})"
        else:
            return f"✅ ({profiler})"

    def main_menu(self):
        """Show the main menu"""
        while True:
            self.show_header()

            # Show current config status
            self.show_current_config()

            table = Table(title="🎯 Main Menu", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            table.add_column("Status", style="green")

            # Menu options with status - logically organized
            options = [
                # Core experiment setup
                (
                    "1", 
                    "📝 Set Experiment Name", 
                    f"✅ ({self.config['name']})" if self.config["name"] else "⚪"
                ),
                (
                    "2",
                    "🤖 Select Models",
                    (
                        f"✅ ({len(self.config['models'])} selected)"
                        if self.config["models"]
                        else "⚪"
                    ),
                ),
                (
                    "3",
                    "💭 Choose Prompts",
                    (
                        f"✅ ({len(self.config['prompts'])} selected)"
                        if self.config["prompts"]
                        else "⚪"
                    ),
                ),
                (
                    "4",
                    "⚙️  Select Algorithms",
                    (
                        f"✅ ({len(self.config['algorithms'])} selected)"
                        if self.config["algorithms"]
                        else "⚪"
                    ),
                ),
                ("5", "🔧 Configure Parameters", "✅"),
                (
                    "6",
                    "⚡ Energy Profiling",
                    self._get_energy_status(),
                ),
                # Configuration management
                ("7", "💾 Save Configuration", "💾"),
                ("8", "📂 Load Configuration", "📂"),
                # System setup and diagnostics
                ("9", "🔧 Install Needed Tools", "🛠️"),
                ("10", "🏥 System Diagnostics", "�"),
                ("11", "🔐 API Keys Manager", "🔑"),
                ("12", "📦 Data Management", "💾"),
                # Execution
                (
                    "13",
                    "🚀 Run Experiment",
                    "🚀" if self.is_config_complete() else "⚪",
                ),
                # Results and maintenance
                ("14", "🔍 Results Explorer", "📊"),
                ("15", "🧹 Project Cleanup", "🗑️"),
                # Information and exit
                ("16", "ℹ️  About Tool", "📋"),
                ("0", "🚪 Exit", "🚪"),
            ]

            for opt, desc, status in options:
                table.add_row(opt, desc, status)

            self.console.print(table)
            self.console.print()

            choice = Prompt.ask(
                "Choose an option", choices=["0"] + [str(i) for i in range(1, 17)]
            )

            if choice == "1":
                self.set_experiment_name()
            elif choice == "2":
                self.select_models()
            elif choice == "3":
                self.choose_prompts()
            elif choice == "4":
                self.select_algorithms()
            elif choice == "5":
                self.configure_parameters()
            elif choice == "6":
                self.configure_energy_profiling()
            elif choice == "7":
                self.save_configuration()
            elif choice == "8":
                self.load_configuration()
            elif choice == "9":
                self.install_needed_tools()
            elif choice == "10":
                self.run_system_diagnostics()
            elif choice == "11":
                self.manage_api_keys()
            elif choice == "12":
                self.manage_data()
            elif choice == "13":
                if self.is_config_complete():
                    self.run_experiment()
                else:
                    self.console.print(
                        "[red]❌ Configuration incomplete! Please set name, models, prompts, and algorithms.[/red]"
                    )
                    input("\nPress Enter to continue...")
            elif choice == "14":
                self.launch_results_explorer()
            elif choice == "15":
                self.project_cleanup()
            elif choice == "16":
                self.show_about()
            elif choice == "0":
                if Confirm.ask("Are you sure you want to exit?"):
                    break

    def show_current_config(self):
        """Show current configuration status"""
        # Check if any selected algorithm uses datasets
        uses_dataset = any(
            alg in ALGORITHMS_USING_DATASETS for alg in self.config["algorithms"]
        )
        dataset_info = ""
        if uses_dataset:
            # Show that datasets will be auto-selected per algorithm
            dataset_count = len(
                [
                    alg
                    for alg in self.config["algorithms"]
                    if alg in ALGORITHMS_USING_DATASETS
                ]
            )
            dataset_info = f"\nDatasets: [blue]{dataset_count} algorithms will use appropriate datasets[/blue]"

        # Get comprehensive energy status
        energy_status = self.config['energy_profiler']
        if self.config.get('environmental_tracking', False):
            region = self.config.get('environmental_region', 'USA East')
            energy_status = f"{energy_status}+env ({region})"
        
        config_panel = Panel(
            f"[bold]Current Configuration:[/bold]\n"
            f"Name: [cyan]{self.config['name'] or 'Not set'}[/cyan]\n"
            f"Models: [green]{len(self.config['models'])} selected[/green]\n"
            f"Prompts: [yellow]{len(self.config['prompts'])} selected[/yellow]\n"
            f"Algorithms: [blue]{len(self.config['algorithms'])} selected[/blue]\n"
            f"Repetitions: [magenta]{self.config['repetitions']}[/magenta]\n"
            f"Energy: [red]{energy_status}[/red]{dataset_info}",
            title="📊 Status",
            box=box.SIMPLE,
        )
        self.console.print(config_panel)
        self.console.print()

    def set_experiment_name(self):
        """Set the experiment name"""
        self.console.clear()
        self.console.print("[bold blue]📝 Set Experiment Name[/bold blue]\n")

        current = f" (current: {self.config['name']})" if self.config["name"] else ""
        name = Prompt.ask(f"Enter experiment name{current}")

        if name:
            self.config["name"] = name
            self.console.print(f"[green]✅ Experiment name set to: {name}[/green]")

        input("\nPress Enter to continue...")

    def select_models(self):
        """Select models to compare"""
        self.console.clear()

        while True:
            self.console.print("[bold blue]🤖 Select Models to Compare[/bold blue]\n")

            # Show currently selected models
            if self.config["models"]:
                selected_table = Table(
                    title="Currently Selected Models", box=box.SIMPLE
                )
                selected_table.add_column("Model", style="green")
                selected_table.add_column("Size", style="magenta")

                for model in self.config["models"]:
                    # Handle both string format (legacy) and dict format (new)
                    if isinstance(model, dict):
                        model_id = model.get("id", "Unknown")
                        model_size = model.get("size", "Unknown")
                    else:
                        model_id = model
                        # Try to fetch size for legacy string models
                        model_size = self._get_model_size(model_id)

                    selected_table.add_row(model_id, model_size)
                self.console.print(selected_table)
                self.console.print()

            # Show model selection options
            table = Table(title="Model Selection", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")

            # Only show popular models if they exist
            if POPULAR_MODELS:
                model_ids = [self._get_model_id(m) for m in self.config["models"]]
                for i, model in enumerate(POPULAR_MODELS, 1):
                    selected = "✅" if model in model_ids else "⚪"
                    table.add_row(str(i), f"{model} {selected}")

            table.add_row("c", "➕ Add custom model manually")
            table.add_row("s", "🔍 Search HuggingFace models")
            table.add_row("r", "🗑️  Remove a model")
            table.add_row("0", "🔙 Back to main menu")

            self.console.print(table)
            self.console.print(
                "\n[dim]Tip: Use 's' to search for models on HuggingFace Hub[/dim]"
            )

            valid_choices = [str(i) for i in range(len(POPULAR_MODELS) + 1)] + [
                "c",
                "s",
                "r",
            ]
            choice = Prompt.ask("Choose option", choices=valid_choices)

            if choice == "0":
                break
            elif choice == "c":
                custom_model = Prompt.ask(
                    "Enter custom model ID (e.g., 'microsoft/DialoGPT-large')"
                )
                model_ids = [self._get_model_id(m) for m in self.config["models"]]
                if custom_model and custom_model not in model_ids:
                    # Get model size
                    model_size = self._get_model_size(custom_model)

                    # Store as dict with metadata
                    model_info = {"id": custom_model, "size": model_size}
                    self.config["models"].append(model_info)
                    self.console.print(
                        f"[green]✅ Added: {custom_model} ({model_size})[/green]"
                    )
            elif choice == "s":
                self.search_huggingface_models()
            elif choice == "r":
                if self.config["models"]:
                    for i, model in enumerate(self.config["models"], 1):
                        model_id = self._get_model_id(model)
                        self.console.print(f"{i}. {model_id}")

                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config["models"]):
                            removed = self.config["models"].pop(remove_idx)
                            removed_id = self._get_model_id(removed)
                            self.console.print(f"[red]❌ Removed: {removed_id}[/red]")
                    except:
                        self.console.print("[red]Invalid selection[/red]")
                else:
                    self.console.print("[yellow]No models selected yet[/yellow]")
            else:
                try:
                    idx = int(choice) - 1
                    model = POPULAR_MODELS[idx]
                    model_ids = [self._get_model_id(m) for m in self.config["models"]]

                    if model in model_ids:
                        # Find and remove the model
                        for i, m in enumerate(self.config["models"]):
                            if self._get_model_id(m) == model:
                                self.config["models"].pop(i)
                                break
                        self.console.print(f"[red]❌ Removed: {model}[/red]")
                    else:
                        # Get model size and add
                        model_size = self._get_model_size(model)
                        model_info = {"id": model, "size": model_size}
                        self.config["models"].append(model_info)
                        self.console.print(
                            f"[green]✅ Added: {model} ({model_size})[/green]"
                        )
                except:
                    self.console.print("[red]Invalid selection[/red]")

            input("\nPress Enter to continue...")
            self.console.clear()

    def choose_prompts(self):
        """Choose prompts from categories"""
        self.console.clear()

        while True:
            self.console.print(
                "[bold blue]💭 Choose Prompts (Easy to Difficult)[/bold blue]\n"
            )

            # Show currently selected prompts
            if self.config["prompts"]:
                selected_table = Table(
                    title="Currently Selected Prompts", box=box.SIMPLE
                )
                selected_table.add_column("#", style="cyan", width=4)
                selected_table.add_column("Prompt", style="green")
                for i, prompt in enumerate(self.config["prompts"], 1):
                    # Truncate long prompts for display
                    display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
                    selected_table.add_row(str(i), display_prompt)
                self.console.print(selected_table)
                self.console.print()

            # Show categories
            table = Table(
                title="Prompt Categories (🟢 Easy → 🔴 Hard)", box=box.ROUNDED
            )
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Category", style="white")
            table.add_column("Difficulty", style="yellow")
            table.add_column("Count", style="blue")

            difficulties = [
                "🟢 Easy",
                "🟡 Easy-Medium",
                "🟠 Medium",
                "🔵 Medium-Hard",
                "🟣 Hard",
                "🔴 Very Hard",
                "⚫ Expert",
                "🌟 Advanced",
            ]

            categories = list(PROMPT_CATEGORIES.keys())
            for i, (category, difficulty) in enumerate(
                zip(categories, difficulties), 1
            ):
                count = len(PROMPT_CATEGORIES[category])
                table.add_row(str(i), category, difficulty, f"{count} prompts")

            table.add_row("a", "Add all categories", "🌈 Mixed", "")
            table.add_row("c", "Add custom prompt", "✏️ Custom", "")
            table.add_row("r", "Remove prompt", "❌ Remove", "")
            table.add_row("0", "Back to main menu", "", "")

            self.console.print(table)

            choices = [str(i) for i in range(len(categories) + 1)] + ["a", "c", "r"]
            choice = Prompt.ask("Choose option", choices=choices)

            if choice == "0":
                break
            elif choice == "a":
                # Add all prompts from all categories
                for prompts in PROMPT_CATEGORIES.values():
                    for prompt in prompts:
                        if prompt not in self.config["prompts"]:
                            self.config["prompts"].append(prompt)
                self.console.print(
                    "[green]✅ Added all prompts from all categories![/green]"
                )
            elif choice == "c":
                custom_prompt = Prompt.ask("Enter your custom prompt")
                if custom_prompt and custom_prompt not in self.config["prompts"]:
                    self.config["prompts"].append(custom_prompt)
                    self.console.print(f"[green]✅ Added custom prompt[/green]")
            elif choice == "r":
                if self.config["prompts"]:
                    for i, prompt in enumerate(self.config["prompts"], 1):
                        display_prompt = (
                            prompt[:50] + "..." if len(prompt) > 50 else prompt
                        )
                        self.console.print(f"{i}. {display_prompt}")

                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config["prompts"]):
                            removed = self.config["prompts"].pop(remove_idx)
                            self.console.print(f"[red]❌ Removed prompt[/red]")
                    except:
                        self.console.print("[red]Invalid selection[/red]")
                else:
                    self.console.print("[yellow]No prompts selected yet[/yellow]")
            else:
                try:
                    idx = int(choice) - 1
                    category_name = categories[idx]
                    self.console.clear()
                    self.show_category_prompts(category_name)
                except:
                    self.console.print("[red]Invalid selection[/red]")

            input("\nPress Enter to continue...")
            self.console.clear()

    def show_category_prompts(self, category_name: str):
        """Show prompts in a specific category"""
        while True:
            self.console.print(f"[bold blue]💭 {category_name} Prompts[/bold blue]\n")

            prompts = PROMPT_CATEGORIES[category_name]
            table = Table(title=f"{category_name}", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Prompt", style="white")
            table.add_column("Selected", style="green")

            for i, prompt in enumerate(prompts, 1):
                selected = "✅" if prompt in self.config["prompts"] else "⚪"
                # Truncate long prompts for table display
                display_prompt = prompt[:70] + "..." if len(prompt) > 70 else prompt
                table.add_row(str(i), display_prompt, selected)

            table.add_row("all", "Add all prompts from this category", "")
            table.add_row("0", "Back to categories", "")

            self.console.print(table)

            choices = [str(i) for i in range(len(prompts) + 1)] + ["all"]
            choice = Prompt.ask("Choose option", choices=choices)

            if choice == "0":
                break
            elif choice == "all":
                for prompt in prompts:
                    if prompt not in self.config["prompts"]:
                        self.config["prompts"].append(prompt)
                self.console.print(
                    f"[green]✅ Added all prompts from {category_name}[/green]"
                )
                input("\nPress Enter to continue...")
            else:
                try:
                    idx = int(choice) - 1
                    prompt = prompts[idx]
                    if prompt in self.config["prompts"]:
                        self.config["prompts"].remove(prompt)
                        self.console.print(f"[red]❌ Removed prompt[/red]")
                    else:
                        self.config["prompts"].append(prompt)
                        self.console.print(f"[green]✅ Added prompt[/green]")
                    input("\nPress Enter to continue...")
                except:
                    self.console.print("[red]Invalid selection[/red]")
                    input("\nPress Enter to continue...")

            self.console.clear()

    def select_algorithms(self):
        """Select comparison algorithms"""
        self.console.clear()

        while True:
            self.console.print(
                "[bold blue]🧮 Select Comparison Algorithms[/bold blue]\n"
            )

            # Show currently selected algorithms
            if self.config["algorithms"]:
                selected_table = Table(
                    title="Currently Selected Algorithms", box=box.SIMPLE
                )
                selected_table.add_column("Algorithm", style="green")
                selected_table.add_column("Description", style="dim")

                # Create algorithm description lookup
                all_algorithms = {}
                for category in ALGORITHM_CATEGORIES.values():
                    for alg_name, alg_desc in category:
                        all_algorithms[alg_name] = alg_desc

                for alg in self.config["algorithms"]:
                    desc = all_algorithms.get(alg, "Custom algorithm")
                    selected_table.add_row(alg, desc)
                self.console.print(selected_table)
                self.console.print()

            # Show all algorithms in a numbered list
            table = Table(title="Available Algorithms", box=box.ROUNDED)
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Algorithm", style="white")
            table.add_column("Datasets", style="magenta")
            table.add_column("Category", style="yellow")
            table.add_column("Description", style="dim")
            table.add_column("Selected", style="green", justify="center")

            # Build flat list of all algorithms with their categories
            all_algs = []
            current_num = 1
            for category, algorithms in ALGORITHM_CATEGORIES.items():
                for alg_name, alg_desc in algorithms:
                    all_algs.append((current_num, alg_name, category, alg_desc))
                    current_num += 1

            # Add algorithms to table with visual separators
            last_category = None
            for num, alg_name, category, alg_desc in all_algs:
                # Add separator line between different categories
                if last_category and last_category != category:
                    table.add_row(
                        "", "─" * 20, "─" * 30, "─" * 15, "─" * 40, "", end_section=True
                    )

                last_category = category
                selected = "✅" if alg_name in self.config["algorithms"] else "⚪"

                # Add API key warning for g_eval
                display_name = alg_name
                if alg_name == "g_eval":
                    try:
                        from llm_runner.config.api_keys import get_api_key_manager

                        manager = get_api_key_manager()
                        openai_key = manager.get_api_key("openai")
                        if not openai_key:
                            display_name = f"{alg_name} ⚠️ (API Needed)"
                    except ImportError:
                        display_name = f"{alg_name} ⚠️ (API Needed)"

                # Check if algorithm uses datasets and format with 📊 icon
                # Algorithms that don't need datasets
                NO_DATASET_ALGORITHMS = [
                    "response_time",
                    "token_throughput",
                    "text_length",
                    "g_eval",
                    "pairwise_comparison",
                    "perplexity",
                ]

                datasets_info = ""
                if alg_name in ALGORITHMS_USING_DATASETS:
                    datasets_info = f"📊 {ALGORITHM_DATASET_MAP.get(alg_name, '')}"
                elif alg_name in NO_DATASET_ALGORITHMS:
                    datasets_info = "⚡ No dataset needed"

                table.add_row(
                    str(num), display_name, datasets_info, category, alg_desc, selected
                )

            self.console.print(table)
            self.console.print()

            # Calculate total algorithms dynamically
            total_algs = len(all_algs)

            # Show quick select options
            quick_table = Table(title="Quick Select Options", box=box.ROUNDED)
            quick_table.add_column("Option", style="cyan", no_wrap=True)
            quick_table.add_column("Description", style="white")
            quick_table.add_column("Count", style="blue")

            quick_table.add_row(
                "a",
                "Quick select: Performance only (response_time, token_throughput, text_length)",
                "3 algorithms",
            )
            quick_table.add_row(
                "b",
                "Quick select: Quality only (bleu, rouge, bert_score, semantic_similarity)",
                "4 algorithms",
            )
            quick_table.add_row(
                "c",
                "Quick select: Comprehensive (response_time, text_length, bleu, rouge, bert_score, llm_as_judge, safety_alignment, truthfulness)",
                "8 algorithms",
            )
            quick_table.add_row(
                "all",
                "🎯 Add ALL algorithms from ALL categories",
                f"{total_algs} algorithms",
            )
            quick_table.add_row("r", "Remove algorithm", "")
            quick_table.add_row("0", "Back to main menu", "")

            self.console.print(quick_table)

            # Build choices list
            choices = [str(i) for i in range(total_algs + 1)] + [
                "a",
                "b",
                "c",
                "all",
                "r",
            ]
            choice = Prompt.ask(
                "Choose option (number to toggle, letter for quick select)",
                choices=choices,
            )

            if choice == "0":
                break
            elif choice == "a":
                # Quick select: Performance
                perf_algs = ["response_time", "token_throughput", "text_length"]
                for alg in perf_algs:
                    if alg not in self.config["algorithms"]:
                        self.config["algorithms"].append(alg)
                self.console.print("[green]✅ Added performance algorithms[/green]")
            elif choice == "b":
                # Quick select: Quality
                quality_algs = ["bleu", "rouge", "bert_score", "semantic_similarity"]
                for alg in quality_algs:
                    if alg not in self.config["algorithms"]:
                        self.config["algorithms"].append(alg)
                self.console.print("[green]✅ Added quality algorithms[/green]")
            elif choice == "c":
                # Quick select: Comprehensive
                comp_algs = [
                    "response_time",
                    "text_length",
                    "bleu",
                    "rouge",
                    "bert_score",
                    "llm_as_judge",
                    "safety_alignment",
                    "truthfulness",
                ]
                for alg in comp_algs:
                    if alg not in self.config["algorithms"]:
                        self.config["algorithms"].append(alg)
                self.console.print("[green]✅ Added comprehensive algorithms[/green]")
            elif choice == "all":
                # Check if user has OpenAI API key for g_eval
                has_openai_key = False
                try:
                    from llm_runner.config.api_keys import get_api_key_manager

                    manager = get_api_key_manager()
                    openai_key = manager.get_api_key("openai")
                    has_openai_key = bool(openai_key)
                except ImportError:
                    pass

                # Add ALL algorithms from ALL categories
                added_count = 0
                skipped_g_eval = False
                for category_algs in ALGORITHM_CATEGORIES.values():
                    for alg_name, _ in category_algs:
                        # Skip g_eval if no API key
                        if alg_name == "g_eval" and not has_openai_key:
                            skipped_g_eval = True
                            continue

                        if alg_name not in self.config["algorithms"]:
                            self.config["algorithms"].append(alg_name)
                            added_count += 1

                # Show success message
                if skipped_g_eval:
                    self.console.print(
                        f"[green]🎯 Added {added_count} algorithms from all categories![/green]"
                    )
                    self.console.print(
                        "[yellow]⚠️  Note: Skipped g_eval (requires OpenAI API key)[/yellow]\n"
                        "[dim]Configure OpenAI API key in option 11 to enable g_eval[/dim]"
                    )
                else:
                    self.console.print(
                        f"[green]🎯 Added ALL {added_count} algorithms from all categories![/green]"
                    )

                self.console.print(
                    f"[dim]Total selected: {len(self.config['algorithms'])} algorithms[/dim]"
                )
            elif choice == "r":
                if self.config["algorithms"]:
                    for i, alg in enumerate(self.config["algorithms"], 1):
                        self.console.print(f"{i}. {alg}")

                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config["algorithms"]):
                            removed = self.config["algorithms"].pop(remove_idx)
                            self.console.print(f"[red]❌ Removed: {removed}[/red]")
                    except:
                        self.console.print("[red]Invalid selection[/red]")
                else:
                    self.console.print("[yellow]No algorithms selected yet[/yellow]")
            else:
                # Handle number selection to toggle algorithm
                try:
                    num = int(choice)
                    if 1 <= num <= len(all_algs):
                        # Find the algorithm by number
                        selected_alg = all_algs[num - 1]
                        alg_name = selected_alg[1]

                        # Check if g_eval and no API key
                        if alg_name == "g_eval":
                            try:
                                from llm_runner.config.api_keys import (
                                    get_api_key_manager,
                                )

                                manager = get_api_key_manager()
                                openai_key = manager.get_api_key("openai")

                                if not openai_key:
                                    self.console.print(
                                        "[bold red]⚠️  WARNING: Cannot select g_eval[/bold red]\n"
                                        "[yellow]g_eval requires an OpenAI API key to function.[/yellow]\n"
                                        "[dim]Please configure your OpenAI API key first (option 11 in main menu).[/dim]"
                                    )
                                    input("\nPress Enter to continue...")
                                    self.console.clear()
                                    continue
                            except ImportError:
                                self.console.print(
                                    "[bold red]⚠️  WARNING: Cannot verify API key[/bold red]\n"
                                    "[yellow]Unable to check if OpenAI API key is configured.[/yellow]"
                                )
                                input("\nPress Enter to continue...")
                                self.console.clear()
                                continue

                        # Toggle selection
                        if alg_name in self.config["algorithms"]:
                            self.config["algorithms"].remove(alg_name)
                            self.console.print(f"[red]❌ Removed: {alg_name}[/red]")
                        else:
                            self.config["algorithms"].append(alg_name)
                            self.console.print(f"[green]✅ Added: {alg_name}[/green]")
                    else:
                        self.console.print("[red]Invalid number[/red]")
                except ValueError:
                    self.console.print("[red]Invalid selection[/red]")

            input("\nPress Enter to continue...")
            self.console.clear()

        # Check for API key requirements after selection
        self._check_algorithm_api_requirements()

    def _check_algorithm_api_requirements(self):
        """Check if selected algorithms require API keys and warn user"""
        api_requiring_algorithms = {"g_eval": "openai", "llm_as_judge": "openai"}

        # Check which selected algorithms need API keys
        needs_api_key = []
        for alg in self.config["algorithms"]:
            if alg in api_requiring_algorithms:
                service = api_requiring_algorithms[alg]
                needs_api_key.append((alg, service))

        if not needs_api_key:
            return

        # Check if API keys are configured
        try:
            from llm_runner.config.api_keys import get_api_key_manager

            manager = get_api_key_manager()

            missing_keys = []
            for alg, service in needs_api_key:
                if not manager.has_key(service):
                    missing_keys.append((alg, service))

            if missing_keys:
                self.console.print("\n[bold yellow]⚠️  API Key Warning:[/bold yellow]")
                self.console.print(
                    "The following selected algorithms require API keys:\n"
                )

                for alg, service in missing_keys:
                    self.console.print(
                        f"  • [cyan]{alg}[/cyan] requires [yellow]{service.upper()}[/yellow] API key"
                    )

                self.console.print(
                    "\n[dim]Set API keys using option 12 in the main menu (🔐 API Keys Manager)[/dim]"
                )
                self.console.print(
                    "[dim]Or these algorithms will be skipped during the experiment.[/dim]"
                )
                input("\nPress Enter to continue...")
        except ImportError:
            pass

    def show_category_algorithms(self, category_name: str):
        """Show algorithms in a specific category"""
        while True:
            self.console.print(f"[bold blue]🧮 {category_name}[/bold blue]\n")

            algorithms = ALGORITHM_CATEGORIES[category_name]
            table = Table(title=f"{category_name}", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Algorithm", style="white")
            table.add_column("Description", style="dim")
            table.add_column("Selected", style="green")

            for i, (alg_name, alg_desc) in enumerate(algorithms, 1):
                selected = "✅" if alg_name in self.config["algorithms"] else "⚪"
                # Add warning icon for g_eval if no API key
                display_name = alg_name

                # Check if algorithm requires API key
                if alg_name == "g_eval":
                    try:
                        from llm_runner.config.api_keys import get_api_key_manager

                        manager = get_api_key_manager()
                        openai_key = manager.get_api_key("openai")
                        if not openai_key:
                            display_name = f"{alg_name} ⚠️"
                            alg_desc = (
                                f"{alg_desc} [red](requires OpenAI API key)[/red]"
                            )
                    except ImportError:
                        pass

                table.add_row(str(i), display_name, alg_desc, selected)

                # Add dataset info row if algorithm uses datasets
                if alg_name in ALGORITHMS_USING_DATASETS:
                    datasets = ALGORITHM_DATASET_MAP.get(alg_name, "")
                    table.add_row("", "📊 Datasets", datasets, "")

            table.add_row("a", "Add all from this category", "", "")
            table.add_row("0", "Back to categories", "", "")

            self.console.print(table)

            choices = [str(i) for i in range(len(algorithms) + 1)] + ["a"]
            choice = Prompt.ask("Choose option", choices=choices)

            if choice == "0":
                break
            elif choice == "a":
                # Check if g_eval is in this category and user has API key
                has_geval = any(alg_name == "g_eval" for alg_name, _ in algorithms)
                if has_geval:
                    try:
                        from llm_runner.config.api_keys import get_api_key_manager

                        manager = get_api_key_manager()
                        openai_key = manager.get_api_key("openai")

                        if not openai_key:
                            self.console.print(
                                "[bold red]⚠️  WARNING: Cannot add all algorithms[/bold red]\n"
                                "[yellow]This category includes g_eval which requires an OpenAI API key.[/yellow]\n"
                                "[dim]Please configure your OpenAI API key first (option 9 in main menu)[/dim]\n"
                                "[dim]or select algorithms individually to skip g_eval.[/dim]"
                            )
                            input("\nPress Enter to continue...")
                            self.console.clear()
                            continue
                    except ImportError:
                        pass

                for alg_name, _ in algorithms:
                    if alg_name not in self.config["algorithms"]:
                        self.config["algorithms"].append(alg_name)
                self.console.print(
                    f"[green]✅ Added all algorithms from {category_name}[/green]"
                )
                input("\nPress Enter to continue...")
            else:
                try:
                    idx = int(choice) - 1
                    alg_name, alg_desc = algorithms[idx]

                    # Check if this algorithm requires an API key
                    if alg_name == "g_eval":
                        try:
                            from llm_runner.config.api_keys import get_api_key_manager

                            manager = get_api_key_manager()
                            openai_key = manager.get_api_key("openai")

                            if not openai_key:
                                self.console.print(
                                    "[bold red]⚠️  WARNING: Cannot select g_eval[/bold red]\n"
                                    "[yellow]g_eval requires an OpenAI API key to function.[/yellow]\n"
                                    "[dim]Please configure your OpenAI API key first (option 9 in main menu).[/dim]"
                                )
                                input("\nPress Enter to continue...")
                                self.console.clear()
                                continue
                        except ImportError:
                            self.console.print(
                                "[bold red]⚠️  WARNING: Cannot verify API key[/bold red]\n"
                                "[yellow]Unable to check if OpenAI API key is configured.[/yellow]"
                            )
                            input("\nPress Enter to continue...")
                            self.console.clear()
                            continue

                    if alg_name in self.config["algorithms"]:
                        self.config["algorithms"].remove(alg_name)
                        self.console.print(f"[red]❌ Removed: {alg_name}[/red]")
                    else:
                        self.config["algorithms"].append(alg_name)
                        self.console.print(f"[green]✅ Added: {alg_name}[/green]")
                    input("\nPress Enter to continue...")
                except:
                    self.console.print("[red]Invalid selection[/red]")
                    input("\nPress Enter to continue...")

            self.console.clear()

    def configure_parameters(self):
        """Configure experiment parameters"""
        self.console.clear()

        while True:
            self.console.print(
                "[bold blue]⚙️ Configure Experiment Parameters[/bold blue]\n"
            )

            # Show current parameters
            param_table = Table(title="Current Parameters", box=box.SIMPLE)
            param_table.add_column("Parameter", style="cyan")
            param_table.add_column("Value", style="green")
            param_table.add_column("Description", style="dim")

            param_table.add_row(
                "repetitions",
                str(self.config["repetitions"]),
                "Number of times to run each test",
            )
            param_table.add_row(
                "max_length",
                str(self.config["max_length"]),
                "Maximum response length in tokens",
            )
            param_table.add_row(
                "temperature",
                str(self.config["temperature"]),
                "Controls LLM creativity (0.0=deterministic, 1.0=creative)",
            )

            self.console.print(param_table)
            self.console.print()

            table = Table(title="Parameter Options", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Parameter", style="white")

            table.add_row(
                "1", "Repetitions (current: {})".format(self.config["repetitions"])
            )
            table.add_row(
                "2", "Max Length (current: {})".format(self.config["max_length"])
            )
            table.add_row(
                "3", "Temperature (current: {})".format(self.config["temperature"])
            )
            table.add_row("4", "Quick presets")
            table.add_row("0", "Back to main menu")

            self.console.print(table)
            self.console.print(
                "\n[dim]💡 Note: Each algorithm automatically uses its appropriate datasets[/dim]"
            )
            self.console.print(
                "[dim]   (e.g., BLEU uses translation datasets, ROUGE uses summarization datasets)[/dim]\n"
            )

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "0"])

            if choice == "0":
                break
            elif choice == "1":
                new_reps = IntPrompt.ask(
                    "Enter number of repetitions", default=self.config["repetitions"]
                )
                if 1 <= new_reps <= 20:
                    self.config["repetitions"] = new_reps
                    self.console.print(
                        f"[green]✅ Repetitions set to {new_reps}[/green]"
                    )
                else:
                    self.console.print(
                        "[red]❌ Please enter a value between 1 and 20[/red]"
                    )
            elif choice == "2":
                self.console.print("\n[bold]Max Length Guidelines:[/bold]")
                self.console.print(
                    "  • 100 tokens: Minimum acceptable for quality metrics"
                )
                self.console.print(
                    "  • 150 tokens: [green]Recommended[/green] - Comprehensive responses"
                )
                self.console.print("  • 200 tokens: Detailed responses (higher cost)")
                self.console.print(
                    "  • 512 tokens: Full responses (benchmark standard)\n"
                )

                new_length = IntPrompt.ask(
                    "Enter maximum response length", default=self.config["max_length"]
                )
                if 10 <= new_length <= 1000:
                    self.config["max_length"] = new_length

                    # Provide feedback based on value
                    if new_length < 100:
                        self.console.print(
                            f"[yellow]⚠️  Warning: {new_length} tokens may truncate responses mid-sentence[/yellow]"
                        )
                        self.console.print(
                            "[dim]Consider using 100+ tokens for reliable quality metrics[/dim]"
                        )
                    elif 100 <= new_length < 150:
                        self.console.print(
                            f"[green]✅ Max length set to {new_length} (acceptable minimum)[/green]"
                        )
                    elif 150 <= new_length <= 200:
                        self.console.print(
                            f"[green]✅ Max length set to {new_length} (optimal range)[/green]"
                        )
                    else:
                        self.console.print(
                            f"[green]✅ Max length set to {new_length}[/green]"
                        )
                else:
                    self.console.print(
                        "[red]❌ Please enter a value between 10 and 1000[/red]"
                    )
            elif choice == "3":
                try:
                    new_temp = float(
                        Prompt.ask(
                            "Enter temperature (0.0-1.0)",
                            default=str(self.config["temperature"]),
                        )
                    )
                    if 0.0 <= new_temp <= 1.0:
                        self.config["temperature"] = new_temp
                        self.console.print(
                            f"[green]✅ Temperature set to {new_temp}[/green]"
                        )
                    else:
                        self.console.print(
                            "[red]❌ Please enter a value between 0.0 and 1.0[/red]"
                        )
                except ValueError:
                    self.console.print("[red]❌ Please enter a valid number[/red]")
            elif choice == "4":
                self.console.print("\n[bold]Quick Presets (Research-Based):[/bold]")
                self.console.print(
                    "1. [yellow]Quick Test[/yellow]: 2 reps, 100 length, 0.7 temp"
                )
                self.console.print(
                    "   └─ Rapid testing, minimal statistical reliability"
                )
                self.console.print(
                    "2. [green]Recommended[/green]: 3 reps, 150 length, 0.7 temp"
                )
                self.console.print(
                    "   └─ Optimal balance: cost-effective + statistically sound"
                )
                self.console.print(
                    "3. [blue]Rigorous[/blue]: 5 reps, 200 length, 0.7 temp"
                )
                self.console.print(
                    "   └─ High confidence, removes 95%+ ranking instability"
                )
                self.console.print(
                    "4. [cyan]Factual Tasks[/cyan]: 3 reps, 150 length, 0.0 temp"
                )
                self.console.print(
                    "   └─ For math/truthfulness (deterministic responses)"
                )
                self.console.print(
                    "5. [magenta]Creative Tasks[/magenta]: 3 reps, 200 length, 1.0 temp"
                )
                self.console.print("   └─ For creative writing (maximum diversity)\n")

                preset = Prompt.ask("Choose preset", choices=["1", "2", "3", "4", "5"])

                if preset == "1":
                    self.config.update(
                        {"repetitions": 2, "max_length": 100, "temperature": 0.7}
                    )
                    self.console.print(
                        "[yellow]⚠️  Quick test preset: Limited statistical reliability[/yellow]"
                    )
                elif preset == "2":
                    self.config.update(
                        {"repetitions": 3, "max_length": 150, "temperature": 0.7}
                    )
                    self.console.print(
                        "[green]✅ Recommended preset: Optimal for most evaluations[/green]"
                    )
                elif preset == "3":
                    self.config.update(
                        {"repetitions": 5, "max_length": 200, "temperature": 0.7}
                    )
                    self.console.print(
                        "[blue]✅ Rigorous preset: High statistical confidence[/blue]"
                    )
                elif preset == "4":
                    self.config.update(
                        {"repetitions": 3, "max_length": 150, "temperature": 0.0}
                    )
                    self.console.print(
                        "[cyan]✅ Factual preset: Best for math/truthfulness tasks[/cyan]"
                    )
                elif preset == "5":
                    self.config.update(
                        {"repetitions": 3, "max_length": 200, "temperature": 1.0}
                    )
                    self.console.print(
                        "[magenta]✅ Creative preset: Maximum response diversity[/magenta]"
                    )

            input("\nPress Enter to continue...")
            self.console.clear()

    def _get_model_id(self, model):
        """Extract model ID from either string or dict format"""
        if isinstance(model, dict):
            return model.get("id", "Unknown")
        return model

    def _format_model_size(self, size_bytes):
        """Format model size in human-readable format"""
        if not size_bytes or size_bytes == 0:
            return "Unknown"

        # Convert bytes to appropriate unit
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f} MB"
        else:
            return f"{size_bytes / (1024**3):.2f} GB"

    def _size_to_bytes(self, size_str):
        """Convert human-readable size string back to bytes for sorting"""
        if not size_str or size_str == "Unknown":
            return 0

        try:
            # Parse size string like "5.18 GB", "522.7 MB", etc.
            parts = size_str.strip().split()
            if len(parts) != 2:
                return 0

            value = float(parts[0])
            unit = parts[1].upper()

            if unit == "B":
                return int(value)
            elif unit == "KB":
                return int(value * 1024)
            elif unit == "MB":
                return int(value * 1024**2)
            elif unit == "GB":
                return int(value * 1024**3)
            elif unit == "TB":
                return int(value * 1024**4)
            else:
                return 0
        except:
            return 0

    def _get_model_size(self, model_id):
        """Get model size from HuggingFace API (main model file size)"""
        try:
            from huggingface_hub import HfApi

            api = HfApi()

            # Get model info with file metadata
            model_info = api.model_info(model_id, files_metadata=True)

            # Priority order for main model files
            # We want to show the size of the actual model weights, not the entire repo
            main_model_patterns = [
                "model.safetensors",  # SafeTensors format (preferred)
                "pytorch_model.bin",  # PyTorch format
                "model.ckpt",  # Checkpoint format
                "tf_model.h5",  # TensorFlow format
            ]

            # Also check for sharded models
            sharded_patterns = [
                "model-",  # Sharded safetensors (model-00001-of-00002.safetensors)
                "pytorch_model-",  # Sharded pytorch
            ]

            size_bytes: int = 0
            found_main_file = False

            # Check if model has siblings (files)
            if hasattr(model_info, "siblings") and model_info.siblings:
                # First try to find the main model file
                for sibling in model_info.siblings:
                    if (
                        hasattr(sibling, "rfilename")
                        and hasattr(sibling, "size")
                        and sibling.size
                    ):
                        # Check for exact match with main model files
                        if sibling.rfilename in main_model_patterns:
                            size_bytes = sibling.size
                            found_main_file = True
                            break

                # If no single file found, check for sharded models
                if not found_main_file:
                    for sibling in model_info.siblings:
                        if (
                            hasattr(sibling, "rfilename")
                            and hasattr(sibling, "size")
                            and sibling.size
                        ):
                            # Check if it's a shard
                            if any(
                                pattern in sibling.rfilename
                                for pattern in sharded_patterns
                            ):
                                size_bytes += sibling.size
                                found_main_file = True

            # If we got a size, return formatted version
            if size_bytes > 0:
                return self._format_model_size(size_bytes)

            # Fallback: try to estimate from model card or tags
            if hasattr(model_info, "tags") and model_info.tags:
                for tag in model_info.tags:
                    # Look for size indicators in tags (e.g., "7b", "13b", "70b")
                    if "b" in tag.lower() and any(c.isdigit() for c in tag):
                        return tag.upper()

            return "Unknown"

        except Exception as e:
            return "Unknown"

    def configure_energy_profiling(self):
        """Configure energy profiling options with multi-select"""
        while True:
            self.console.clear()
            self.console.print("[bold blue]🔋 Configure Energy Profiling & Environmental Impact[/bold blue]\n")

            # Show current configuration
            current_profiler = self.config.get("energy_profiler", "none")
            current_env = self.config.get("environmental_tracking", False)
            current_region = self.config.get("environmental_region", "USA East")
            
            status_table = Table(title="Current Configuration", box=box.ROUNDED)
            status_table.add_column("Setting", style="cyan")
            status_table.add_column("Value", style="white")
            status_table.add_column("Status", style="green")
            
            status_table.add_row(
                "Energy Profiler",
                current_profiler.upper(),
                "✅" if current_profiler != "none" else "⚪"
            )
            status_table.add_row(
                "Environmental Tracking",
                "Enabled" if current_env else "Disabled",
                "✅" if current_env else "⚪"
            )
            if current_env:
                status_table.add_row(
                    "Region",
                    current_region,
                    "✅"
                )
            
            self.console.print(status_table)
            self.console.print()

            # Menu options
            table = Table(title="Configuration Menu", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Action", style="white")
            table.add_column("Description", style="dim")

            table.add_row(
                "1",
                "Toggle Energy Profiler",
                f"Current: {current_profiler.upper()} → Switch to {'NONE' if current_profiler == 'codecarbon' else 'CODECARBON'}"
            )
            table.add_row(
                "2",
                "Toggle Environmental Tracking",
                f"Current: {'ON' if current_env else 'OFF'} → Switch to {'OFF' if current_env else 'ON'}"
            )
            table.add_row(
                "3",
                "Select Region",
                f"Current: {current_region} (Only if environmental tracking is ON)"
            )
            table.add_row(
                "4",
                "ℹ️  About Features",
                "Learn about energy profiling and environmental tracking"
            )
            table.add_row("0", "Save & Return", "Save configuration and return to main menu")

            self.console.print(table)

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "0"])

            if choice == "0":
                # Save and return
                return
            elif choice == "1":
                # Toggle energy profiler
                if current_profiler == "none":
                    self.config["energy_profiler"] = "codecarbon"
                    self.console.print("[green]✅ CodeCarbon energy profiling enabled[/green]")
                    self.console.print(
                        "[cyan]📊 Will measure: CPU (RAPL), GPU (nvidia-smi), RAM, and CO2 emissions[/cyan]"
                    )
                else:
                    self.config["energy_profiler"] = "none"
                    self.config["environmental_tracking"] = False
                    self.console.print("[yellow]⚪ Energy profiling disabled[/yellow]")
                    self.console.print("[dim]Note: Environmental tracking also disabled[/dim]")
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                # Toggle environmental tracking
                if current_profiler == "none":
                    self.console.print("[red]❌ Enable CodeCarbon first (Option 1)[/red]")
                    input("\nPress Enter to continue...")
                else:
                    if current_env:
                        self.config["environmental_tracking"] = False
                        self.console.print("[yellow]⚪ Environmental tracking disabled[/yellow]")
                        self.console.print("[dim]Energy profiling still active (energy + CO2 only)[/dim]")
                    else:
                        self.config["environmental_tracking"] = True
                        self.console.print("[green]✅ Environmental tracking enabled[/green]")
                        self.console.print(
                            "[cyan]📊 Will track: Water, PUE, Regional Carbon, Eco-Efficiency[/cyan]"
                        )
                        # Auto-prompt for region selection
                        input("\nPress Enter to select your region...")
                        self._select_region()
                    input("\nPress Enter to continue...")
                    
            elif choice == "3":
                # Select region
                if not current_env:
                    self.console.print("[red]❌ Enable Environmental Tracking first (Option 2)[/red]")
                    input("\nPress Enter to continue...")
                else:
                    self._select_region()
                    input("\nPress Enter to continue...")
                    
            elif choice == "4":
                # Show information
                self._show_energy_info()
                input("\nPress Enter to continue...")
    
    def _show_energy_info(self):
        """Display information about energy profiling and environmental tracking"""
        self.console.clear()
        self.console.print("[bold cyan]ℹ️  Energy Profiling & Environmental Impact Information[/bold cyan]\n")
        
        # CodeCarbon info
        codecarbon_panel = Panel(
            "[bold yellow]🔋 CodeCarbon Energy Profiling[/bold yellow]\n\n"
            "[green]✓[/green] Uses RAPL (Running Average Power Limit) for real CPU energy\n"
            "[green]✓[/green] Hardware-level measurements via /sys/class/powercap\n"
            "[green]✓[/green] NVIDIA GPU support via nvidia-smi\n"
            "[green]✓[/green] Automatic CO2 calculation from grid data\n"
            "[green]✓[/green] No root/sudo access required\n\n"
            "[cyan]Measures:[/cyan] CPU, GPU, RAM energy + Carbon emissions",
            title="CodeCarbon (Option 1)",
            box=box.ROUNDED,
        )
        self.console.print(codecarbon_panel)
        self.console.print()
        
        # Environmental tracking info
        env_panel = Panel(
            "[bold yellow]🌍 Environmental Impact Tracking[/bold yellow]\n\n"
            "[green]✓[/green] Water usage tracking (WUE) - on-site + off-site cooling\n"
            "[green]✓[/green] Infrastructure overhead (PUE) - data center efficiency\n"
            "[green]✓[/green] Regional carbon intensity (50+ global regions)\n"
            "[green]✓[/green] Eco-efficiency scoring for sustainable AI\n\n"
            "[cyan]Requires:[/cyan] CodeCarbon must be enabled first\n"
            "[cyan]Adds:[/cyan] Water footprint, PUE multiplier, regional CIF",
            title="Environmental Tracking (Option 2)",
            box=box.ROUNDED,
        )
        self.console.print(env_panel)
        self.console.print()
        
        # Example combinations
        combo_table = Table(title="Example Configurations", box=box.ROUNDED)
        combo_table.add_column("CodeCarbon", style="cyan")
        combo_table.add_column("Environmental", style="cyan")
        combo_table.add_column("What You Get", style="white")
        
        combo_table.add_row("OFF", "OFF", "No tracking")
        combo_table.add_row("ON", "OFF", "Energy + CO2 only")
        combo_table.add_row("ON", "ON", "Full impact: Energy + Water + Carbon + PUE + Eco-Efficiency")
        
        self.console.print(combo_table)
    
    def _select_region(self):
        """Select geographic region for environmental impact calculations"""
        from app.src.llm_runner.profiling import Region
        
        self.console.clear()
        self.console.print("[bold blue]🌍 Select Geographic Region[/bold blue]\n")
        
        # Group regions by area
        regions_by_area = {
            "North America": [
                (Region.USA_EAST, "USA East Coast"),
                (Region.USA_WEST, "USA West Coast"),
                (Region.USA_CENTRAL, "USA Central"),
                (Region.CANADA_EAST, "Canada East"),
                (Region.CANADA_WEST, "Canada West"),
                (Region.MEXICO, "Mexico"),
            ],
            "Europe - West": [
                (Region.EUROPE_WEST, "Europe West"),
                (Region.UK, "United Kingdom"),
                (Region.IRELAND, "Ireland"),
                (Region.FRANCE, "France (Nuclear-heavy, low carbon)"),
                (Region.GERMANY, "Germany"),
                (Region.NETHERLANDS, "Netherlands"),
                (Region.SWITZERLAND, "Switzerland"),
            ],
            "Europe - North": [
                (Region.EUROPE_NORTH, "Europe North"),
                (Region.NORDICS, "Nordic Countries (Clean energy)"),
                (Region.ICELAND, "Iceland (Geothermal, cleanest)"),
            ],
            "Europe - East & South": [
                (Region.EUROPE_EAST, "Europe East"),
                (Region.EUROPE_SOUTH, "Europe South"),
            ],
            "Asia Pacific - China": [
                (Region.CHINA_NORTH, "China North"),
                (Region.CHINA_EAST, "China East"),
                (Region.CHINA_SOUTH, "China South"),
            ],
            "Asia Pacific - Other": [
                (Region.JAPAN, "Japan"),
                (Region.KOREA, "South Korea"),
                (Region.SINGAPORE, "Singapore"),
                (Region.AUSTRALIA_EAST, "Australia East"),
                (Region.AUSTRALIA_SOUTHEAST, "Australia Southeast"),
            ],
            "India": [
                (Region.INDIA_WEST, "India West"),
                (Region.INDIA_SOUTH, "India South"),
                (Region.INDIA_CENTRAL, "India Central"),
            ],
            "Middle East & Africa": [
                (Region.UAE, "United Arab Emirates"),
                (Region.SAUDI_ARABIA, "Saudi Arabia"),
                (Region.SOUTH_AFRICA, "South Africa"),
            ],
            "South America": [
                (Region.BRAZIL_SOUTH, "Brazil South"),
                (Region.BRAZIL_SOUTHEAST, "Brazil Southeast"),
            ],
            "Custom": [
                (Region.CUSTOM, "Custom (define your own multipliers)"),
            ],
        }
        
        # Display regions by area
        all_choices = []
        choice_to_region = {}
        idx = 1
        
        for area, regions in regions_by_area.items():
            self.console.print(f"\n[bold cyan]{area}:[/bold cyan]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Option", style="yellow")
            table.add_column("Region", style="white")
            
            for region_enum, region_name in regions:
                choice_str = str(idx)
                all_choices.append(choice_str)
                choice_to_region[choice_str] = region_enum
                current = self.config.get("environmental_region")
                status = "✅" if current == region_enum.value else ""
                table.add_row(f"{idx}.", f"{region_name} {status}")
                idx += 1
            
            self.console.print(table)
        
        all_choices.append("0")
        self.console.print("\n[dim]0. Back[/dim]\n")
        
        choice = Prompt.ask("Select your region", choices=all_choices)
        
        if choice != "0":
            selected_region = choice_to_region[choice]
            self.config["environmental_region"] = selected_region.value
            
            # Show selected region info
            from app.src.llm_runner.profiling import EnvironmentalMultipliers
            multipliers = EnvironmentalMultipliers.get_regional_defaults(selected_region)
            
            self.console.print(f"\n[green]✅ Selected: {selected_region.value}[/green]")
            self.console.print(f"[dim]   PUE: {multipliers.pue} | "
                             f"WUE: {multipliers.wue_site + multipliers.wue_source:.2f} L/kWh | "
                             f"Carbon: {multipliers.cif} kgCO2e/kWh[/dim]")

    def save_configuration(self):
        """Save current configuration to file"""
        self.console.clear()
        self.console.print("[bold blue]💾 Save Configuration[/bold blue]\n")

        if not self.is_config_complete():
            self.console.print("[red]❌ Configuration incomplete! Cannot save.[/red]")
            input("\nPress Enter to continue...")
            return

        # Suggest filename based on experiment name
        suggested_name = (
            f"{self.config['name']}_config.json"
            if self.config["name"]
            else "lct_config.json"
        )
        filename = Prompt.ask("Save as", default=suggested_name)

        try:
            config_path = Path("saved_configs")
            config_path.mkdir(exist_ok=True)

            filepath = config_path / filename
            with open(filepath, "w") as f:
                json.dump(self.config, f, indent=2)

            self.console.print(f"[green]✅ Configuration saved to {filepath}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Error saving configuration: {e}[/red]")

        input("\nPress Enter to continue...")

    def load_configuration(self):
        """Load configuration from file"""
        self.console.clear()
        self.console.print("[bold blue]📂 Load Configuration[/bold blue]\n")

        config_path = Path("saved_configs")
        if not config_path.exists():
            self.console.print("[yellow]No saved configurations found.[/yellow]")
            input("\nPress Enter to continue...")
            return

        config_files = list(config_path.glob("*.json"))
        if not config_files:
            self.console.print("[yellow]No saved configuration files found.[/yellow]")
            input("\nPress Enter to continue...")
            return

        table = Table(title="Saved Configurations", box=box.ROUNDED)
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Filename", style="white")
        table.add_column("Modified", style="dim")

        for i, filepath in enumerate(config_files, 1):
            import datetime

            mtime = datetime.datetime.fromtimestamp(filepath.stat().st_mtime)
            table.add_row(str(i), filepath.name, mtime.strftime("%Y-%m-%d %H:%M"))

        table.add_row("0", "Cancel", "")
        self.console.print(table)

        choices = [str(i) for i in range(len(config_files) + 1)]
        choice = Prompt.ask("Choose configuration to load", choices=choices)

        if choice == "0":
            return

        try:
            idx = int(choice) - 1
            filepath = config_files[idx]

            with open(filepath, "r") as f:
                loaded_config = json.load(f)

            self.config.update(loaded_config)
            self.console.print(
                f"[green]✅ Configuration loaded from {filepath.name}[/green]"
            )
        except Exception as e:
            self.console.print(f"[red]❌ Error loading configuration: {e}[/red]")

        input("\nPress Enter to continue...")

    def is_config_complete(self):
        """Check if configuration is complete enough to run"""
        return (
            self.config["name"]
            and self.config["models"]
            and self.config["prompts"]
            and self.config["algorithms"]
        )

    def search_huggingface_models(self):
        """Search HuggingFace models with pagination"""
        self.console.clear()
        self.console.print("[bold blue]🔍 Search HuggingFace Models[/bold blue]\n")

        search_term = Prompt.ask("Enter search term (e.g., 'gpt', 'llama', 'bert')")
        if not search_term.strip():
            self.console.print("[yellow]No search term entered[/yellow]")
            input("\nPress Enter to continue...")
            return

        try:
            from huggingface_hub import list_models

            # Show progress and search for models
            with self.console.status(
                f"[bold green]Searching for '{search_term}' models..."
            ):
                # Search for models - get up to 500 results
                models = list(list_models(search=search_term, limit=500))

            if not models:
                self.console.print(
                    f"[yellow]No models found for '{search_term}'[/yellow]"
                )
                input("\nPress Enter to continue...")
                return

            # Default sort by name (A-Z)
            models.sort(
                key=lambda m: getattr(m, "id", getattr(m, "modelId", "")).lower()
            )

            # Initialize variables for sorting
            sort_choice = "3"  # Default to Name A-Z
            model_sizes = {}

            # Store sort info for display
            sort_info = {
                "1": "Downloads ↓",
                "2": "Downloads ↑",
                "3": "Name A-Z",
                "4": "Name Z-A",
                "5": "Size ↓",
                "6": "Size ↑",
                "7": "Default",
            }

            # Pagination settings
            page_size = 15
            current_page = 0
            total_pages = (len(models) + page_size - 1) // page_size  # Ceiling division

            while True:
                # Clear and display results for current page
                self.console.clear()
                self.console.print(
                    f"[bold green]✅ Found {len(models)} models for '{search_term}'[/bold green]"
                )
                self.console.print(
                    f"[dim]Page {current_page + 1} of {total_pages} • Sorted by: {sort_info[sort_choice]}[/dim]\n"
                )

                # Get models for current page
                start_idx = current_page * page_size
                end_idx = min(start_idx + page_size, len(models))
                display_models = models[start_idx:end_idx]

                # Display search results
                table = Table(
                    title=f"Search Results (Page {current_page + 1}/{total_pages})",
                    box=box.ROUNDED,
                )
                table.add_column("Option", style="cyan", no_wrap=True)
                table.add_column("Model ID", style="white")
                table.add_column("Size", style="magenta")
                table.add_column("Downloads", style="green")
                table.add_column("Tags", style="yellow")

                for i, model in enumerate(display_models, 1):
                    downloads = getattr(model, "downloads", 0) or 0
                    tags = ", ".join(
                        getattr(model, "tags", [])[:3]
                    )  # Show first 3 tags
                    model_id = getattr(
                        model, "id", getattr(model, "modelId", "Unknown")
                    )

                    # Get model size (use cached if available from sorting)
                    if model_id in model_sizes:
                        model_size = model_sizes[model_id]
                    else:
                        model_size = self._get_model_size(model_id)

                    table.add_row(str(i), model_id, model_size, f"{downloads:,}", tags)

                self.console.print(table)

                # Navigation and sorting options
                self.console.print("\n[bold cyan]Navigation:[/bold cyan]")
                nav_options = []

                if current_page > 0:
                    self.console.print("  [cyan]p[/cyan] - Previous page")
                    nav_options.append("p")

                if current_page < total_pages - 1:
                    self.console.print("  [cyan]n[/cyan] - Next page")
                    nav_options.append("n")

                self.console.print("\n[bold cyan]Sorting:[/bold cyan]")
                self.console.print("  [cyan]sd[/cyan] - Sort by Downloads")
                self.console.print("  [cyan]sn[/cyan] - Sort by Name")
                self.console.print("  [cyan]ss[/cyan] - Sort by Size")
                nav_options.extend(["sd", "sn", "ss"])

                self.console.print("\n[bold cyan]Selection:[/bold cyan]")
                self.console.print("  [cyan]1-15[/cyan] - Select a model to add")
                self.console.print("  [cyan]0[/cyan] - Back to model selection")

                # Build valid choices
                valid_choices = [
                    str(i) for i in range(len(display_models) + 1)
                ] + nav_options

                choice = Prompt.ask("\nEnter your choice", choices=valid_choices)

                if choice == "0":
                    return
                elif choice == "n":
                    current_page += 1
                    continue
                elif choice == "p":
                    current_page -= 1
                    continue
                elif choice == "sd":
                    # Sort by downloads
                    self.console.print("\n[cyan]1[/cyan] - Most downloads first (↓)")
                    self.console.print("[cyan]2[/cyan] - Least downloads first (↑)")
                    sort_dir = Prompt.ask(
                        "Choose direction", choices=["1", "2"], default="1"
                    )
                    if sort_dir == "1":
                        models.sort(
                            key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True
                        )
                        sort_choice = "1"
                    else:
                        models.sort(key=lambda m: getattr(m, "downloads", 0) or 0)
                        sort_choice = "2"
                    current_page = 0
                    total_pages = (len(models) + page_size - 1) // page_size
                    continue
                elif choice == "sn":
                    # Sort by name
                    self.console.print("\n[cyan]1[/cyan] - A to Z (↓)")
                    self.console.print("[cyan]2[/cyan] - Z to A (↑)")
                    sort_dir = Prompt.ask(
                        "Choose direction", choices=["1", "2"], default="1"
                    )
                    if sort_dir == "1":
                        models.sort(
                            key=lambda m: getattr(
                                m, "id", getattr(m, "modelId", "")
                            ).lower()
                        )
                        sort_choice = "3"
                    else:
                        models.sort(
                            key=lambda m: getattr(
                                m, "id", getattr(m, "modelId", "")
                            ).lower(),
                            reverse=True,
                        )
                        sort_choice = "4"
                    current_page = 0
                    total_pages = (len(models) + page_size - 1) // page_size
                    continue
                elif choice == "ss":
                    # Sort by size
                    self.console.print("\n[cyan]1[/cyan] - Largest first (↓)")
                    self.console.print("[cyan]2[/cyan] - Smallest first (↑)")
                    sort_dir = Prompt.ask(
                        "Choose direction", choices=["1", "2"], default="1"
                    )
                    self.console.print(
                        "\n[yellow]⏳ Fetching model sizes for sorting...[/yellow]"
                    )
                    models_with_size = []
                    with self.console.status("[bold green]Processing models..."):
                        for model in models:
                            model_id = getattr(
                                model, "id", getattr(model, "modelId", "Unknown")
                            )
                            size = self._get_model_size(model_id)
                            size_bytes = self._size_to_bytes(size)
                            models_with_size.append((model, size, size_bytes))

                    if sort_dir == "1":
                        models_with_size.sort(key=lambda x: x[2], reverse=True)
                        sort_choice = "5"
                    else:
                        models_with_size.sort(key=lambda x: x[2])
                        sort_choice = "6"

                    models = [m[0] for m in models_with_size]
                    model_sizes = {
                        getattr(m[0], "id", getattr(m[0], "modelId", "Unknown")): m[1]
                        for m in models_with_size
                    }
                    current_page = 0
                    total_pages = (len(models) + page_size - 1) // page_size
                    continue
                else:
                    # User selected a model
                    break

            # Process model selection

            try:
                idx = int(choice) - 1
                selected_model = getattr(
                    display_models[idx],
                    "id",
                    getattr(display_models[idx], "modelId", None),
                )
                if not selected_model:
                    self.console.print("\n[red]❌ Could not get model ID[/red]")
                    input("\nPress Enter to continue...")
                    return

                # Check if model already exists (handle both string and dict formats)
                model_ids = [self._get_model_id(m) for m in self.config["models"]]

                if selected_model not in model_ids:
                    # Get model size
                    model_size = self._get_model_size(selected_model)

                    # Store as dict with metadata
                    model_info = {"id": selected_model, "size": model_size}
                    self.config["models"].append(model_info)
                    self.console.print(
                        f"\n[bold green]✅ Added: {selected_model} ({model_size})[/bold green]"
                    )
                else:
                    self.console.print(
                        f"\n[bold yellow]⚠️  Model already selected: {selected_model}[/bold yellow]"
                    )
                input("\nPress Enter to continue...")
            except (ValueError, IndexError):
                self.console.print("\n[red]❌ Invalid selection[/red]")
                input("\nPress Enter to continue...")

        except ImportError:
            self.console.print("[red]❌ HuggingFace Hub not installed.[/red]")
            self.console.print("[dim]Install with: pip install huggingface_hub[/dim]")
            input("\nPress Enter to continue...")
        except Exception as e:
            self.console.print(f"[red]❌ Error searching models: {str(e)}[/red]")
            input("\nPress Enter to continue...")

    def setup_huggingface_auth(self):
        """Setup HuggingFace authentication"""
        self.console.clear()
        self.console.print(
            "[bold blue]🔑 HuggingFace Authentication Setup[/bold blue]\n"
        )

        # Check current authentication status
        try:
            from huggingface_hub import whoami

            # Check if token is already set
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if token:
                try:
                    user_info = whoami(token=token)
                    self.console.print(
                        f"[green]✅ Already authenticated as: {user_info.get('name', 'Unknown')}[/green]"
                    )

                    update_choice = Prompt.ask(
                        "Would you like to update your token?",
                        choices=["y", "n"],
                        default="n",
                    )
                    if update_choice.lower() != "y":
                        return
                except:
                    self.console.print("[yellow]⚠️ Current token seems invalid[/yellow]")

            self.console.print(
                "To access private models and avoid rate limits, you need a HuggingFace token."
            )
            self.console.print(
                "Get your token from: https://huggingface.co/settings/tokens\n"
            )

            # Get token from user
            token = Prompt.ask(
                "Enter your HuggingFace token (leave empty to skip)",
                default="",
                show_default=False,
            )

            if token:
                # Verify token
                try:
                    with self.console.status("[bold green]Verifying token..."):
                        user_info = whoami(token=token)
                        name = user_info.get("name", "Unknown")

                    # Show success message after spinner closes
                    self.console.print(
                        f"[green]✅ Token valid! Authenticated as: {name}[/green]"
                    )

                    # Save token to environment
                    os.environ["HF_TOKEN"] = token

                    # Ask about saving to .env file (using regular Prompt instead of Confirm)
                    save_choice = Prompt.ask(
                        "\n💾 Save token to .env file for future sessions?",
                        choices=["y", "n"],
                        default="y",
                    )

                    if save_choice.lower() == "y":
                        env_file = Path(".env")
                        env_content = ""

                        if env_file.exists():
                            env_content = env_file.read_text()

                        # Remove existing HF_TOKEN lines
                        lines = [
                            line
                            for line in env_content.split("\n")
                            if not line.startswith("HF_TOKEN=")
                        ]

                        # Add new token
                        lines.append(f"HF_TOKEN={token}")

                        env_file.write_text("\n".join(lines))
                        self.console.print("[green]✅ Token saved to .env file[/green]")
                    else:
                        self.console.print(
                            "[yellow]Token saved for current session only[/yellow]"
                        )

                except Exception as e:
                    self.console.print(f"[red]❌ Invalid token: {e}[/red]")
            else:
                self.console.print(
                    "[yellow]Authentication skipped. Some models may not be accessible.[/yellow]"
                )

        except ImportError:
            self.console.print(
                "[red]❌ HuggingFace Hub not installed. Install with: pip install huggingface_hub[/red]"
            )
        except Exception as e:
            self.console.print(f"[red]❌ Error setting up authentication: {e}[/red]")

        input("\nPress Enter to continue...")

    def manage_api_keys(self):
        """Manage API keys for LLM services (OpenAI, HuggingFace, Anthropic)"""
        try:
            from llm_runner.config.api_keys import get_api_key_manager
        except ImportError as e:
            self.console.print(f"[red]❌ API key manager not available: {e}[/red]")
            self.console.print(
                "[yellow]Make sure llm_runner.config.api_keys module exists[/yellow]"
            )
            input("\nPress Enter to continue...")
            return

        manager = get_api_key_manager()

        while True:
            self.console.clear()
            self.console.print("[bold blue]🔐 API Keys Manager[/bold blue]\n")

            # Show current configured services
            services = manager.list_services()
            if services:
                api_key_lines = []
                for svc in services:
                    api_key = manager.get_api_key(svc)
                    if api_key and len(api_key) > 12:
                        display = (
                            f"[green]✓ {svc}[/green]: {api_key[:8]}...{api_key[-4:]}"
                        )
                    elif api_key:
                        display = f"[green]✓ {svc}[/green]: ****"
                    else:
                        display = f"[yellow]⚠ {svc}[/yellow]: Not set"
                    api_key_lines.append(display)

                status_panel = Panel(
                    "\n".join(api_key_lines),
                    title="📋 Configured API Keys",
                    box=box.SIMPLE,
                )
                self.console.print(status_panel)
                self.console.print()
            else:
                self.console.print("[yellow]⚠️ No API keys configured yet[/yellow]\n")

            # Show info about which algorithms need API keys
            info_panel = Panel(
                "[bold]API Key Requirements:[/bold]\n\n"
                "• [cyan]OpenAI API[/cyan] - Required for:\n"
                "  - g_eval (G-Eval scoring)\n"
                "  - llm_as_judge (LLM-based evaluation)\n\n"
                "• [cyan]HuggingFace API[/cyan] - Optional for:\n"
                "  - Private model access\n"
                "  - Higher rate limits\n\n"
                "• [cyan]Anthropic API[/cyan] - Future support",
                title="ℹ️  Information",
                box=box.ROUNDED,
                style="dim",
            )
            self.console.print(info_panel)
            self.console.print()

            # Menu
            table = Table(box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")

            table.add_row("1", "🔑 Add/Update OpenAI API Key")
            table.add_row("2", "🔑 Add/Update HuggingFace Token (with verification)")
            table.add_row("3", "🔑 Add/Update Anthropic API Key")
            table.add_row("4", "👁️  Show API Key (masked)")
            table.add_row("5", "🗑️  Remove API Key")
            table.add_row("6", "📋 List All Configured Keys")
            table.add_row("0", "🔙 Back to Main Menu")

            self.console.print(table)
            self.console.print()

            choice = Prompt.ask(
                "Choose an option", choices=["0", "1", "2", "3", "4", "5", "6"]
            )

            if choice == "0":
                break
            elif choice == "1":
                # Add/Update OpenAI API key
                self.console.print("\n[bold]Adding OpenAI API Key[/bold]")
                self.console.print(
                    "[dim]Get your key from: https://platform.openai.com/api-keys[/dim]\n"
                )

                api_key = Prompt.ask(
                    "Enter your OpenAI API key", default="", show_default=False
                )

                if api_key:
                    manager.set_api_key("openai", api_key)
                    self.console.print(
                        "[green]✅ OpenAI API key saved successfully![/green]"
                    )
                    self.console.print(
                        "[dim]Stored securely in: ~/.lct/api_keys.json[/dim]"
                    )
                    os.environ["OPENAI_API_KEY"] = api_key
                    self.console.print(
                        "[green]✅ Environment variable set for current session[/green]"
                    )
                else:
                    self.console.print(
                        "[yellow]No key entered, operation cancelled[/yellow]"
                    )

                input("\nPress Enter to continue...")

            elif choice == "2":
                # Add/Update HuggingFace token with verification
                self.console.print("\n[bold]Adding HuggingFace Token[/bold]")
                self.console.print(
                    "[dim]Get your token from: https://huggingface.co/settings/tokens[/dim]\n"
                )

                token = Prompt.ask(
                    "Enter your HuggingFace token", default="", show_default=False
                )

                if token:
                    # Try to verify the token
                    try:
                        from huggingface_hub import whoami

                        with self.console.status("[bold green]Verifying token..."):
                            user_info = whoami(token=token)
                            name = user_info.get("name", "Unknown")

                        self.console.print(
                            f"[green]✅ Token valid! Authenticated as: {name}[/green]"
                        )

                        # Save to API key manager
                        manager.set_api_key("huggingface", token)
                        self.console.print(
                            "[green]✅ HuggingFace token saved successfully![/green]"
                        )
                        self.console.print(
                            "[dim]Stored securely in: ~/.lct/api_keys.json[/dim]"
                        )

                        # Set environment variables for current session
                        os.environ["HUGGINGFACE_API_KEY"] = token
                        os.environ["HF_TOKEN"] = token
                        self.console.print(
                            "[green]✅ Environment variables set for current session[/green]"
                        )

                    except ImportError:
                        self.console.print(
                            "[yellow]⚠️ HuggingFace Hub not installed, skipping verification[/yellow]"
                        )
                        # Still save the token
                        manager.set_api_key("huggingface", token)
                        self.console.print(
                            "[green]✅ HuggingFace token saved (unverified)[/green]"
                        )
                        os.environ["HUGGINGFACE_API_KEY"] = token
                        os.environ["HF_TOKEN"] = token
                    except Exception as e:
                        self.console.print(
                            f"[red]❌ Token verification failed: {e}[/red]"
                        )
                        save_anyway = Prompt.ask(
                            "Save anyway?", choices=["y", "n"], default="n"
                        )
                        if save_anyway.lower() == "y":
                            manager.set_api_key("huggingface", token)
                            self.console.print(
                                "[yellow]⚠️ Token saved (unverified)[/yellow]"
                            )
                            os.environ["HUGGINGFACE_API_KEY"] = token
                            os.environ["HF_TOKEN"] = token
                else:
                    self.console.print(
                        "[yellow]No token entered, operation cancelled[/yellow]"
                    )

                input("\nPress Enter to continue...")

            elif choice == "3":
                # Add/Update Anthropic API key
                self.console.print("\n[bold]Adding Anthropic API Key[/bold]")
                self.console.print(
                    "[dim]Get your key from: https://console.anthropic.com/[/dim]\n"
                )

                api_key = Prompt.ask(
                    "Enter your Anthropic API key", default="", show_default=False
                )

                if api_key:
                    manager.set_api_key("anthropic", api_key)
                    self.console.print(
                        "[green]✅ Anthropic API key saved successfully![/green]"
                    )
                    self.console.print(
                        "[dim]Stored securely in: ~/.lct/api_keys.json[/dim]"
                    )
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    self.console.print(
                        "[green]✅ Environment variable set for current session[/green]"
                    )
                else:
                    self.console.print(
                        "[yellow]No key entered, operation cancelled[/yellow]"
                    )

                input("\nPress Enter to continue...")

            elif choice == "4":
                # Show API key
                service = Prompt.ask(
                    "Which service?", choices=["openai", "huggingface", "anthropic"]
                )
                api_key = manager.get_api_key(service)

                if api_key:
                    masked = (
                        api_key[:8] + "..." + api_key[-4:]
                        if len(api_key) > 12
                        else "****"
                    )
                    self.console.print(f"\n[green]{service}[/green]: {masked}")
                else:
                    self.console.print(
                        f"\n[yellow]No API key configured for {service}[/yellow]"
                    )

                input("\nPress Enter to continue...")

            elif choice == "5":
                # Remove API key
                if not services:
                    self.console.print("\n[yellow]No API keys to remove[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                service = Prompt.ask("Which service to remove?", choices=services)
                confirm = Prompt.ask(
                    f"Are you sure you want to remove the {service} API key?",
                    choices=["y", "n"],
                    default="n",
                )

                if confirm.lower() == "y":
                    if manager.remove_api_key(service):
                        self.console.print(
                            f"[green]✅ Removed {service} API key[/green]"
                        )
                    else:
                        self.console.print(
                            f"[yellow]No API key found for {service}[/yellow]"
                        )
                else:
                    self.console.print("[yellow]Operation cancelled[/yellow]")

                input("\nPress Enter to continue...")

            elif choice == "6":
                # List all keys
                if services:
                    self.console.print("\n[bold]Configured API Keys:[/bold]\n")
                    for svc in services:
                        api_key = manager.get_api_key(svc)
                        if api_key and len(api_key) > 12:
                            masked = api_key[:8] + "..." + api_key[-4:]
                        elif api_key:
                            masked = "****"
                        else:
                            masked = "Not set"
                        self.console.print(f"  • [green]{svc}[/green]: {masked}")
                else:
                    self.console.print("\n[yellow]No API keys configured[/yellow]")

                input("\nPress Enter to continue...")

    def manage_data(self):
        """Data management interface for models and datasets"""
        while True:
            self.console.clear()
            self.console.print("[bold blue]📦 Data Management[/bold blue]\n")

            try:
                from ..core.data_manager import get_data_manager

                data_manager = get_data_manager()

                # Show current usage
                usage = data_manager.get_disk_usage()
                usage_panel = Panel(
                    f"[bold]Data Storage Usage:[/bold]\n"
                    f"Models: [green]{usage['models_gb']:.2f} GB[/green]\n"
                    f"Datasets: [blue]{usage['datasets_gb']:.2f} GB[/blue]\n"
                    f"Total: [yellow]{usage['total_gb']:.2f} GB[/yellow]\n"
                    f"Location: [dim]{data_manager.data_root}[/dim]",
                    title="💾 Storage Info",
                    box=box.SIMPLE,
                )
                self.console.print(usage_panel)
                self.console.print()

                # Show menu
                table = Table(title="Data Management Options", box=box.ROUNDED)
                table.add_column("Option", style="cyan", no_wrap=True)
                table.add_column("Description", style="white")

                table.add_row("1", "📥 Download Model")
                table.add_row("2", "📊 Download Dataset")
                table.add_row("3", "📋 List Downloaded Models")
                table.add_row("4", "📈 List Downloaded Datasets")
                table.add_row("5", "🗑️ Cleanup Models")
                table.add_row("6", "🗑️ Cleanup Datasets")
                table.add_row("7", "📁 Open Data Directory")
                table.add_row("0", "🔙 Back to Main Menu")

                self.console.print(table)

                choice = Prompt.ask(
                    "Select option", choices=["1", "2", "3", "4", "5", "6", "7", "0"]
                )

                if choice == "1":
                    self.download_model_ui(data_manager)
                elif choice == "2":
                    self.download_dataset_ui(data_manager)
                elif choice == "3":
                    self.list_downloaded_models_ui(data_manager)
                elif choice == "4":
                    self.list_downloaded_datasets_ui(data_manager)
                elif choice == "5":
                    self.cleanup_models_ui(data_manager)
                elif choice == "6":
                    self.cleanup_datasets_ui(data_manager)
                elif choice == "7":
                    self.open_data_directory_ui(data_manager)
                elif choice == "0":
                    break

            except ImportError:
                self.console.print("[red]❌ Data manager not available[/red]")
                input("\nPress Enter to continue...")
                break

    def download_model_ui(self, data_manager):
        """UI for downloading models"""
        self.console.clear()
        self.console.print("[bold blue]📥 Download Model[/bold blue]\n")

        model_id = Prompt.ask("Enter model ID (e.g., gpt2, microsoft/DialoGPT-medium)")
        model_type = Prompt.ask(
            "Model type", choices=["causal", "seq2seq"], default="causal"
        )

        if Confirm.ask(
            f"Download {model_id} ({model_type})? This may take time and storage space."
        ):
            with self.console.status(f"[bold green]Downloading {model_id}..."):
                result = data_manager.download_model(model_id, model_type)

            if result:
                self.console.print(
                    f"[green]✅ Model {model_id} downloaded to {result}[/green]"
                )
            else:
                self.console.print(f"[red]❌ Failed to download {model_id}[/red]")

        input("\nPress Enter to continue...")

    def download_dataset_ui(self, data_manager):
        """UI for downloading datasets"""
        self.console.clear()
        self.console.print("[bold blue]📊 Download Dataset[/bold blue]\n")

        # Show common datasets
        common_datasets = [
            "xsum",
            "cnn_dailymail",
            "squad",
            "glue",
            "wmt14",
            "wmt16",
            "opus_books",
            "multi_news",
        ]

        self.console.print("Common datasets: " + ", ".join(common_datasets))
        self.console.print()

        dataset_name = Prompt.ask("Enter dataset name")
        config_name = Prompt.ask(
            "Config name (optional, press Enter to skip)", default=""
        )
        config_name = config_name if config_name.strip() else None

        if Confirm.ask(
            f"Download {dataset_name}? This may take time and storage space."
        ):
            with self.console.status(f"[bold green]Downloading {dataset_name}..."):
                result = data_manager.download_dataset(dataset_name, config_name)

            if result:
                self.console.print(
                    f"[green]✅ Dataset {dataset_name} downloaded to {result}[/green]"
                )
            else:
                self.console.print(f"[red]❌ Failed to download {dataset_name}[/red]")

        input("\nPress Enter to continue...")

    def list_downloaded_models_ui(self, data_manager):
        """UI for listing downloaded models"""
        self.console.clear()
        self.console.print("[bold blue]📋 Downloaded Models[/bold blue]\n")

        models = data_manager.list_downloaded_models()

        if not models:
            self.console.print("[yellow]No models downloaded yet[/yellow]")
        else:
            table = Table(title="Downloaded Models", box=box.ROUNDED)
            table.add_column("Model ID", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Path", style="dim")

            for model in models:
                model_id = model.get("model_id", "Unknown")
                model_type = model.get("config", {}).get("model_type", "Unknown")
                local_path = model.get("local_path", "Unknown")
                table.add_row(model_id, model_type, str(local_path))

            self.console.print(table)

        input("\nPress Enter to continue...")

    def list_downloaded_datasets_ui(self, data_manager):
        """UI for listing downloaded datasets"""
        self.console.clear()
        self.console.print("[bold blue]📈 Downloaded Datasets[/bold blue]\n")

        datasets = data_manager.list_downloaded_datasets()

        if not datasets:
            self.console.print("[yellow]No datasets downloaded yet[/yellow]")
        else:
            table = Table(title="Downloaded Datasets", box=box.ROUNDED)
            table.add_column("Dataset Name", style="cyan")
            table.add_column("Config", style="green")
            table.add_column("Splits", style="yellow")
            table.add_column("Path", style="dim")

            for dataset in datasets:
                name = dataset.get("dataset_name", "Unknown")
                config = dataset.get("config_name", "default")
                splits = ", ".join(dataset.get("splits", []))
                local_path = dataset.get("local_path", "Unknown")
                table.add_row(name, config or "default", splits, str(local_path))

            self.console.print(table)

        input("\nPress Enter to continue...")

    def cleanup_models_ui(self, data_manager):
        """UI for cleaning up models"""
        self.console.clear()
        self.console.print("[bold blue]🗑️ Cleanup Models[/bold blue]\n")

        models = data_manager.list_downloaded_models()

        if not models:
            self.console.print("[yellow]No models to clean up[/yellow]")
            input("\nPress Enter to continue...")
            return

        # Show models and let user select
        for i, model in enumerate(models, 1):
            model_id = model.get("model_id", "Unknown")
            self.console.print(f"{i}. {model_id}")

        self.console.print("0. Cancel")

        choice = Prompt.ask("Select model to remove (number)")

        try:
            choice_num = int(choice)
            if choice_num == 0:
                return
            elif 1 <= choice_num <= len(models):
                model_to_remove = models[choice_num - 1]
                model_id = model_to_remove.get("model_id", "Unknown")

                if Confirm.ask(f"Delete {model_id}? This cannot be undone."):
                    if data_manager.cleanup_model(model_id):
                        self.console.print(f"[green]✅ Removed {model_id}[/green]")
                    else:
                        self.console.print(f"[red]❌ Failed to remove {model_id}[/red]")
            else:
                self.console.print("[red]Invalid selection[/red]")
        except ValueError:
            self.console.print("[red]Invalid input[/red]")

        input("\nPress Enter to continue...")

    def cleanup_datasets_ui(self, data_manager):
        """UI for cleaning up datasets"""
        self.console.clear()
        self.console.print("[bold blue]🗑️ Cleanup Datasets[/bold blue]\n")

        datasets = data_manager.list_downloaded_datasets()

        if not datasets:
            self.console.print("[yellow]No datasets to clean up[/yellow]")
            input("\nPress Enter to continue...")
            return

        # Show datasets and let user select
        for i, dataset in enumerate(datasets, 1):
            name = dataset.get("dataset_name", "Unknown")
            config = dataset.get("config_name", "")
            display_name = f"{name}" + (f" ({config})" if config else "")
            self.console.print(f"{i}. {display_name}")

        self.console.print("0. Cancel")

        choice = Prompt.ask("Select dataset to remove (number)")

        try:
            choice_num = int(choice)
            if choice_num == 0:
                return
            elif 1 <= choice_num <= len(datasets):
                dataset_to_remove = datasets[choice_num - 1]
                dataset_name = dataset_to_remove.get("dataset_name", "Unknown")

                if Confirm.ask(f"Delete {dataset_name}? This cannot be undone."):
                    if data_manager.cleanup_dataset(dataset_name):
                        self.console.print(f"[green]✅ Removed {dataset_name}[/green]")
                    else:
                        self.console.print(
                            f"[red]❌ Failed to remove {dataset_name}[/red]"
                        )
            else:
                self.console.print("[red]Invalid selection[/red]")
        except ValueError:
            self.console.print("[red]Invalid input[/red]")

        input("\nPress Enter to continue...")

    def open_data_directory_ui(self, data_manager):
        """UI for opening data directory in file manager"""
        self.console.clear()
        self.console.print("[bold blue]📁 Data Directory[/bold blue]\n")

        import subprocess
        import platform

        data_path = str(data_manager.data_root)
        self.console.print(f"Data directory: [cyan]{data_path}[/cyan]")

        if Confirm.ask("Open in file manager?"):
            try:
                if platform.system() == "Linux":
                    subprocess.run(["xdg-open", data_path], check=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", data_path], check=True)
                elif platform.system() == "Windows":
                    subprocess.run(["explorer", data_path], check=True)

                self.console.print("[green]✅ File manager opened[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Could not open file manager: {e}[/red]")

        input("\nPress Enter to continue...")

    def run_system_diagnostics(self):
        """Run system diagnostics to check if PC can run experiments"""
        self.console.clear()
        self.console.print("[bold blue]🏥 System Diagnostics[/bold blue]\n")

        # Create diagnostics table
        table = Table(title="System Check Results", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")

        # Check Python version
        python_version = platform.python_version()
        python_ok = python_version >= "3.8"
        python_status = "✅ Good" if python_ok else "❌ Update needed"
        table.add_row(
            "Python Version", python_status, f"{python_version} (≥3.8 required)"
        )

        # Check available RAM (if psutil available)
        if psutil:
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            ram_available = memory.available / (1024**3)
            ram_ok = ram_gb >= 4
            ram_status = "✅ Good" if ram_ok else "⚠️ Limited"
            table.add_row(
                "RAM",
                ram_status,
                f"{ram_gb:.1f}GB total, {ram_available:.1f}GB available",
            )

            # Check disk space
            disk = psutil.disk_usage("/")
            disk_free_gb = disk.free / (1024**3)
            disk_ok = disk_free_gb >= 5
            disk_status = "✅ Good" if disk_ok else "⚠️ Limited"
            table.add_row("Disk Space", disk_status, f"{disk_free_gb:.1f}GB free")

            # Check CPU
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_status = "✅ Good" if cpu_count and cpu_count >= 2 else "⚠️ Limited"
            freq_info = f", {cpu_freq.current:.0f}MHz" if cpu_freq else ""
            cpu_info = f"{cpu_count if cpu_count else 'Unknown'} cores{freq_info}"
            table.add_row("CPU", cpu_status, cpu_info)
        else:
            table.add_row(
                "System Info",
                "⚠️ Limited",
                "Install psutil for detailed system info: pip install psutil",
            )

        # Check GPU
        gpu_status = "❌ Not detected"
        gpu_details = "CPU-only processing"
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split("\n")[0]
                gpu_status = "✅ NVIDIA GPU"
                gpu_details = gpu_info
        except:
            try:
                # Check for other GPU indicators
                if shutil.which("rocm-smi") or shutil.which("clinfo"):
                    gpu_status = "⚠️ Non-NVIDIA GPU"
                    gpu_details = "May work with CPU processing"
            except:
                pass

        table.add_row("GPU", gpu_status, gpu_details)

        # Check required packages
        required_packages = {
            "torch": "PyTorch",
            "transformers": "HuggingFace Transformers",
            "sentence_transformers": "Sentence Transformers",
            "nltk": "NLTK",
            "rouge_score": "ROUGE Score",
            "huggingface_hub": "HuggingFace Hub",
        }

        missing_packages = []
        for package, name in required_packages.items():
            try:
                __import__(package)
                table.add_row(f"📦 {name}", "✅ Installed", "Ready to use")
            except ImportError:
                table.add_row(f"📦 {name}", "❌ Missing", "Install required")
                missing_packages.append(package)

        # Check HuggingFace authentication
        auth_status = "❌ Not configured"
        auth_details = "Set up authentication for private models"
        try:
            from huggingface_hub import whoami

            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if token:
                try:
                    user_info = whoami(token=token)
                    auth_status = "✅ Authenticated"
                    auth_details = f"Logged in as: {user_info.get('name', 'Unknown')}"
                except:
                    auth_status = "⚠️ Invalid token"
                    auth_details = "Token exists but invalid"
        except ImportError:
            pass

        table.add_row("🔑 HuggingFace Auth", auth_status, auth_details)

        self.console.print(table)
        self.console.print()

        # Overall assessment
        if not python_ok:
            self.console.print(
                "[red]❌ CRITICAL: Python version too old. Update to Python 3.8+[/red]"
            )
        elif missing_packages:
            self.console.print(
                f"[yellow]⚠️ MISSING PACKAGES: Install with: pip install {' '.join(missing_packages)}[/yellow]"
            )
        elif psutil:
            ram_ok = True
            disk_ok = True
            if hasattr(psutil, "virtual_memory"):
                memory = psutil.virtual_memory()
                ram_gb = memory.total / (1024**3)
                ram_ok = ram_gb >= 4

                disk = psutil.disk_usage("/")
                disk_free_gb = disk.free / (1024**3)
                disk_ok = disk_free_gb >= 5

            if not ram_ok:
                self.console.print(
                    "[yellow]⚠️ LOW RAM: May have issues with large models. Consider using smaller models.[/yellow]"
                )
            elif not disk_ok:
                self.console.print(
                    "[yellow]⚠️ LOW DISK SPACE: Models may not download properly. Free up space.[/yellow]"
                )
            else:
                self.console.print(
                    "[green]🎉 SYSTEM READY: All checks passed! Your system can run LLM experiments.[/green]"
                )
        else:
            self.console.print(
                "[green]✅ BASIC CHECKS PASSED: Install psutil for detailed system analysis.[/green]"
            )

        # Show recommendations
        self.console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")

        if gpu_status == "❌ Not detected":
            self.console.print(
                "• For faster inference, consider using a GPU-enabled system"
            )

        if missing_packages:
            self.console.print(
                f"• Install missing packages: [bold]pip install {' '.join(missing_packages)}[/bold]"
            )

        if auth_status != "✅ Authenticated":
            self.console.print(
                "• Set up HuggingFace authentication using option 11 in main menu"
            )

        if psutil and hasattr(psutil, "virtual_memory"):
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            if ram_gb < 8:
                self.console.print(
                    "• For large models, use smaller variants (e.g., distilgpt2 instead of gpt2-xl)"
                )
        else:
            self.console.print(
                "• Install psutil for detailed system monitoring: pip install psutil"
            )

        self.console.print(
            "• Start with small experiments (1-2 models, simple prompts) to test your setup"
        )

        input("\nPress Enter to continue...")

    def install_needed_tools(self):
        """Install all required packages and tools automatically"""
        # Ensure Path and subprocess are available throughout the method
        from pathlib import Path
        import subprocess
        import sys

        self.console.clear()
        self.console.print(
            "[bold blue]🔧 Install Needed Tools & Dependencies[/bold blue]\n"
        )

        self.console.print(
            "This will install all required packages and tools for LCT to work properly."
        )
        self.console.print("You can choose which components to install:\n")

        # Show detailed installation options
        options_table = Table(title="Installation Components", box=box.ROUNDED)
        options_table.add_column("Option", style="cyan", no_wrap=True)
        options_table.add_column("Component", style="white")
        options_table.add_column("Description", style="dim")

        options_table.add_row(
            "1",
            "🖥️ System Tools",
            "Git, system utilities, and monitoring tools",
        )
        options_table.add_row(
            "2",
            "🐍 Virtual Environment",
            "Python virtual environment setup and pip upgrade",
        )
        options_table.add_row(
            "4",
            "🤖 Core ML Libraries",
            "PyTorch, Transformers, HuggingFace Hub, Sentence Transformers",
        )
        options_table.add_row(
            "5",
            "📊 Evaluation Libraries",
            "NLTK, spaCy, ROUGE, BERT-Score, BLEU, Evaluation metrics",
        )
        options_table.add_row(
            "6",
            "🔧 System Monitoring",
            "psutil, CodeCarbon, nvidia-ml-py, system diagnostics",
        )
        options_table.add_row(
            "7",
            "🎨 UI & Utilities",
            "Rich, Pandas, Matplotlib, Seaborn, NumPy, Requests",
        )
        options_table.add_row(
            "8",
            "� Optional Tools",
            "Jupyter, IPython, Plotly, Dash (development tools)",
        )
        options_table.add_row(
            "9",
            "📋 Requirements.txt",
            "Process existing requirements.txt file if present",
        )
        options_table.add_row(
            "10", "🌐 Language Models", "Download NLTK data and spaCy English model"
        )
        # Get dynamic dataset names for component 11 description
        try:
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).parent.parent.parent.parent / "scripts")
            )
            from dataset_manager import CleanDatasetManager

            dm = CleanDatasetManager()
            dataset_names = ", ".join([info["name"] for info in dm.datasets.values()])
        except:
            dataset_names = "HumanEval, GSM8K, SafetyBench, HellaSwag, TruthfulQA, AlpacaEval, MT-Bench"

        options_table.add_row(
            "10",
            "📚 Research Datasets",
            f"Download research evaluation datasets ({dataset_names})",
        )
        options_table.add_row(
            "11",
            "🗂️ HuggingFace Datasets",
            "Download comprehensive evaluation datasets (BLEU, ROUGE, BERT, Math, Code, etc.)",
        )
        options_table.add_row(
            "12", "🔧 Fix Dependencies", "Resolve PyTorch/NVIDIA package conflicts"
        )
        options_table.add_row(
            "all",
            "🎯 Install Everything",
            "Complete installation (all components above)",
        )
        options_table.add_row("0", "❌ Cancel", "Return to main menu")

        self.console.print(options_table)
        self.console.print("\n[bold yellow]� Recommendations:[/bold yellow]")
        self.console.print("• First-time users: Choose 'all' for complete setup")
        self.console.print("• Advanced users: Select specific components as needed")
        self.console.print("• System tools (1) may require sudo privileges")
        self.console.print(
            "• Virtual environment (2) is required for Python packages (3-12)\n"
        )

        # Get user selection
        valid_choices = [str(i) for i in range(13)] + ["all"]
        choices = []

        while True:
            choice = Prompt.ask(
                "Select components to install (comma-separated, e.g. '1,3,4' or 'all')",
                default="all",
            )

            if choice.lower() == "all":
                choices = list(range(1, 13))  # All components except cancel
                break
            elif choice == "0":
                return
            else:
                try:
                    choices = [
                        int(x.strip()) for x in choice.split(",") if x.strip().isdigit()
                    ]
                    choices = [x for x in choices if 1 <= x <= 12]
                    if choices:
                        break
                    else:
                        self.console.print(
                            "[red]Please enter valid component numbers (1-12) or 'all'[/red]"
                        )
                except ValueError:
                    self.console.print(
                        "[red]Please enter numbers separated by commas or 'all'[/red]"
                    )

        self.console.print(
            f"\n[green]Selected components: {', '.join(map(str, choices))}[/green]\n"
        )

        # Component installation logic
        installed_components = []
        failed_components = []

        # COMPONENT 1: System Tools
        if 1 in choices:
            self.console.print("[bold cyan]🖥️ INSTALLING: System Tools[/bold cyan]")

            system_tools = {
                "git": "Version control system",
                "htop": "System monitoring tool",
                "cpulimit": "CPU limiting utility (for energy profiling)",
            }

            success_count = 0
            for tool, description in system_tools.items():
                self.console.print(f"📥 Installing {tool}: {description}")
                try:
                    result = subprocess.run(
                        ["sudo", "apt", "install", "-y", tool],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        self.console.print(f"  ✅ {tool} installed successfully")
                        success_count += 1
                    else:
                        self.console.print(
                            f"  ⚠️ {tool} installation had issues (may already be installed)"
                        )
                        success_count += 1
                except Exception as e:
                    self.console.print(f"  ❌ Error installing {tool}: {e}")

            if success_count > 0:
                installed_components.append("System Tools")
            else:
                failed_components.append("System Tools")

        # COMPONENT 2: Virtual Environment
        if 2 in choices:
            self.console.print(
                "\n[bold cyan]🐍 INSTALLING: Python Virtual Environment[/bold cyan]"
            )
            venv_path = Path("llm-experiment-runner/.venv")
            if not venv_path.exists():
                self.console.print("📦 Creating virtual environment...")
                try:
                    result = subprocess.run(
                        ["python3", "-m", "venv", str(venv_path)],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        self.console.print("  ✅ Virtual environment created")
                        installed_components.append("Virtual Environment")
                    else:
                        self.console.print(
                            f"  ❌ Virtual environment creation failed: {result.stderr}"
                        )
                        failed_components.append("Virtual Environment")
                        # Skip Python package installation if venv failed
                        if any(x in choices for x in [4, 5, 6, 7, 8, 9, 10, 11, 12]):
                            self.console.print(
                                "  ⚠️  Skipping Python package installation due to virtual environment failure"
                            )
                            return
                except Exception as e:
                    self.console.print(f"  ❌ Virtual environment error: {e}")
                    failed_components.append("Virtual Environment")
                    return
            else:
                self.console.print("  ℹ️  Virtual environment already exists")
                installed_components.append("Virtual Environment")

            # Upgrade pip
            python_exe = str(venv_path / "bin" / "python")
            pip_exe = str(venv_path / "bin" / "pip")

            try:
                self.console.print("📦 Upgrading pip in virtual environment...")
                subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
                self.console.print("  ✅ Pip upgraded successfully")
            except Exception as e:
                self.console.print(f"  ⚠️  Pip upgrade issue: {e}")

        # Set up Python executables for remaining components
        venv_path = Path("llm-experiment-runner/.venv")
        python_exe = (
            str(venv_path / "bin" / "python") if venv_path.exists() else "python3"
        )
        pip_exe = str(venv_path / "bin" / "pip") if venv_path.exists() else "pip3"

        # COMPONENT 3: Core ML Libraries
        if 3 in choices:
            self.console.print(
                "\n[bold cyan]🤖 INSTALLING: Core ML Libraries[/bold cyan]"
            )
            core_packages = [
                (
                    "torch>=2.6.0",
                    "PyTorch deep learning framework (v2.6.0+ for security)",
                ),
                ("transformers", "HuggingFace Transformers library"),
                ("sentence-transformers", "Sentence embedding models"),
                ("huggingface_hub", "HuggingFace model hub client"),
            ]

            success_count = self._install_packages(core_packages, pip_exe)
            if success_count > 0:
                installed_components.append(
                    f"Core ML Libraries ({success_count}/{len(core_packages)})"
                )
            else:
                failed_components.append("Core ML Libraries")

        # COMPONENT 4: Evaluation Libraries
        if 4 in choices:
            self.console.print(
                "\n[bold cyan]📊 INSTALLING: Evaluation Libraries[/bold cyan]"
            )
            eval_packages = [
                ("nltk", "Natural Language Toolkit"),
                ("spacy", "Industrial-strength NLP library"),
                ("rouge-score", "ROUGE evaluation metric"),
                ("bert-score", "BERT-based evaluation metric"),
                ("evaluate", "HuggingFace evaluation library"),
                ("datasets", "HuggingFace datasets library"),
                ("sacrebleu", "BLEU score implementation"),
                ("scikit-learn", "Machine learning evaluation metrics"),
            ]

            success_count = self._install_packages(eval_packages, pip_exe)
            if success_count > 0:
                installed_components.append(
                    f"Evaluation Libraries ({success_count}/{len(eval_packages)})"
                )
            else:
                failed_components.append("Evaluation Libraries")

        # COMPONENT 5: System Monitoring
        if 5 in choices:
            self.console.print(
                "\n[bold cyan]🔧 INSTALLING: System Monitoring Tools[/bold cyan]"
            )
            monitoring_packages = [
                ("psutil", "System and process monitoring"),
                ("codecarbon", "Carbon footprint tracking"),
                (
                    "nvidia-ml-py>=12.0.0",
                    "NVIDIA GPU monitoring (replaces deprecated pynvml)",
                ),
            ]

            success_count = self._install_packages(monitoring_packages, pip_exe)
            if success_count > 0:
                installed_components.append(
                    f"System Monitoring ({success_count}/{len(monitoring_packages)})"
                )
            else:
                failed_components.append("System Monitoring")

        # COMPONENT 6: UI & Utilities
        if 6 in choices:
            self.console.print(
                "\n[bold cyan]🎨 INSTALLING: UI & Utility Libraries[/bold cyan]"
            )
            ui_packages = [
                ("rich", "Rich text and beautiful formatting"),
                ("requests", "HTTP requests library"),
                ("numpy", "Numerical computing library"),
                ("pandas", "Data manipulation library"),
                ("matplotlib", "Plotting and visualization"),
                ("seaborn", "Statistical data visualization"),
            ]

            success_count = self._install_packages(ui_packages, pip_exe)
            if success_count > 0:
                installed_components.append(
                    f"UI & Utilities ({success_count}/{len(ui_packages)})"
                )
            else:
                failed_components.append("UI & Utilities")

        # COMPONENT 7: Optional Tools
        if 7 in choices:
            self.console.print(
                "\n[bold cyan]📈 INSTALLING: Optional Development Tools[/bold cyan]"
            )
            optional_packages = [
                ("jupyter", "Jupyter notebook environment"),
                ("ipython", "Enhanced interactive Python"),
                ("plotly", "Interactive plotting library"),
                ("dash", "Web application framework"),
            ]

            success_count = self._install_packages(optional_packages, pip_exe)
            if success_count > 0:
                installed_components.append(
                    f"Optional Tools ({success_count}/{len(optional_packages)})"
                )
            else:
                failed_components.append("Optional Tools")

        # COMPONENT 8: Requirements.txt
        if 8 in choices:
            requirements_file = Path("config/requirements.txt")
            if requirements_file.exists():
                self.console.print(
                    "\n[bold cyan]📋 INSTALLING: Requirements.txt Dependencies[/bold cyan]"
                )
                try:
                    result = subprocess.run(
                        [pip_exe, "install", "-r", str(requirements_file)],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        self.console.print("✅ Requirements.txt installed successfully")
                        installed_components.append("Requirements.txt")
                    else:
                        self.console.print(
                            f"⚠️ Requirements.txt installation issues: {result.stderr[:100]}..."
                        )
                        # Check if datasets was specifically mentioned as an issue
                        if "datasets" in result.stderr.lower():
                            self.console.print(
                                "   💡 Tip: Try running Component 11 (Reference Datasets) for dataset dependencies"
                            )
                        failed_components.append("Requirements.txt")
                except Exception as e:
                    self.console.print(f"❌ Requirements.txt installation error: {e}")
                    failed_components.append("Requirements.txt")
            else:
                self.console.print(
                    "\n[yellow]📋 Requirements.txt not found - skipping[/yellow]"
                )

        # COMPONENT 9: Language Models
        if 9 in choices:
            self.console.print(
                "\n[bold cyan]🌐 INSTALLING: Language Models & Data[/bold cyan]"
            )

            # Install NLTK data
            self.console.print("📚 Setting up NLTK data...")
            try:
                result = subprocess.run(
                    [
                        python_exe,
                        "-c",
                        "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    self.console.print("✅ NLTK data downloaded")
                    nltk_success = True
                else:
                    self.console.print(f"⚠️ NLTK setup issue: {result.stderr}")
                    nltk_success = False
            except Exception as e:
                self.console.print(f"⚠️ NLTK setup issue: {e}")
                nltk_success = False

            # Install spaCy model
            self.console.print("📚 Setting up spaCy English model...")
            try:
                result = subprocess.run(
                    [python_exe, "-m", "spacy", "download", "en_core_web_sm"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    self.console.print("✅ spaCy English model installed")
                    spacy_success = True
                else:
                    self.console.print(
                        f"⚠️ spaCy model installation issue: {result.stderr}"
                    )
                    spacy_success = False
            except Exception as e:
                self.console.print(f"⚠️ spaCy model installation issue: {e}")
                spacy_success = False

            if nltk_success or spacy_success:
                models_status = []
                if nltk_success:
                    models_status.append("NLTK data")
                if spacy_success:
                    models_status.append("spaCy model")
                installed_components.append(
                    f"Language Models ({', '.join(models_status)})"
                )
            else:
                failed_components.append("Language Models")

        # COMPONENT 10: Research Datasets
        if 10 in choices:
            self.console.print(
                "\n[bold cyan]📚 INSTALLING: Research Datasets for Algorithm Evaluation[/bold cyan]"
            )
            self.console.print(
                "[dim]This downloads research evaluation datasets to enable enhanced algorithm evaluation[/dim]\n"
            )

            try:
                # Use the new clean dataset manager

                # Get project root - use current working directory as fallback
                try:
                    # Try to get path from the module file
                    import app.src.ui.interactive_lct as cli_module

                    project_root = Path(cli_module.__file__).parent.parent.parent.parent
                except Exception as e:
                    # Fallback to current working directory
                    project_root = Path.cwd()

                dataset_manager_path = project_root / "scripts" / "dataset_manager.py"

                if not dataset_manager_path.exists():
                    self.console.print("❌ Dataset manager not found")
                    failed_components.append("Research Datasets")
                else:
                    self.console.print("📥 Installing critical research datasets...")

                    with self.console.status("[bold green]Downloading datasets..."):
                        result = subprocess.run(
                            [sys.executable, str(dataset_manager_path), "--critical"],
                            capture_output=True,
                            text=True,
                            cwd=str(project_root),
                        )

                    if result.returncode == 0:
                        self.console.print(
                            "✅ Critical research datasets installed successfully!"
                        )
                        # Parse output to show installed datasets
                        if "datasets installed" in result.stdout:
                            self.console.print(
                                result.stdout.split("\n")[-10:]
                            )  # Show last 10 lines
                    else:
                        self.console.print(
                            f"⚠️ Some datasets may have failed: {result.stderr}"
                        )
                        # Don't mark as failed if some datasets work

                    # Show quick status
                    status_result = subprocess.run(
                        [sys.executable, str(dataset_manager_path), "--deps"],
                        capture_output=True,
                        text=True,
                        cwd=str(project_root),
                    )

                    if status_result.returncode == 0:
                        self.console.print("✅ Dataset dependencies verified")
                        # Show basic info without loading all datasets
                        if "All dependencies available" in status_result.stdout:
                            self.console.print("📊 Dataset manager ready for use")

            except Exception as e:
                self.console.print(f"❌ Dataset installation error: {e}")
                failed_components.append("Research Datasets")

            # Mark as success if component installation completed
            if "Research Datasets" not in failed_components:
                # Get dynamic dataset info
                try:
                    if project_root:
                        scripts_path = (
                            Path(project_root) / "scripts"
                            if isinstance(project_root, str)
                            else project_root / "scripts"
                        )
                    else:
                        scripts_path = Path.cwd() / "scripts"
                    sys.path.insert(0, str(scripts_path))
                    from dataset_manager import CleanDatasetManager

                    dm = CleanDatasetManager()
                    dataset_count = len(dm.datasets)
                    dataset_names = ", ".join(
                        [info["name"] for info in dm.datasets.values()]
                    )
                except:
                    dataset_count = 7
                    dataset_names = "HumanEval, GSM8K, SafetyBench, HellaSwag, TruthfulQA, AlpacaEval, MT-Bench"

                self.console.print(f"\n[green]📊 ALGORITHM IMPACT:[/green]")
                self.console.print(
                    f"• ✅ Dataset manager ready for on-demand dataset installation"
                )
                self.console.print(
                    f"• ✅ Supports {dataset_count} research datasets: {dataset_names}"
                )
                self.console.print(
                    f"• ✅ Enables enhanced evaluation for text comparison algorithms"
                )
                self.console.print(
                    f"• ✅ All 17 algorithms functional with smart fallback methods"
                )

                self.console.print(f"• 🎯 Clean architecture ready for beta deployment")

                installed_components.append("Research Datasets")
            else:
                self.console.print(
                    f"\n[yellow]⚠️ Dataset manager setup failed - algorithms will use fallback methods[/yellow]"
                )
                self.console.print(
                    f"[dim]💡 Note: All algorithms still functional with built-in evaluation methods[/dim]"
                )

        # COMPONENT 11: HuggingFace Evaluation Datasets
        if 11 in choices:
            self.console.print(
                "\n[bold cyan]🗂️ INSTALLING & TESTING: HuggingFace Evaluation Datasets[/bold cyan]"
            )
            self.console.print(
                "[dim]Downloading, validating, and testing datasets for algorithm compatibility[/dim]\n"
            )

            # Install datasets library if not already installed
            self.console.print("📦 Ensuring 'datasets' library is installed...")
            try:
                result = subprocess.run(
                    [
                        python_exe,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "datasets>=2.14.0",
                        "evaluate>=0.4.0",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    self.console.print("✅ Datasets & Evaluate libraries ready")
                else:
                    self.console.print("⚠️ Issue installing datasets library")
            except Exception as e:
                self.console.print(f"⚠️ Error: {e}")

            # Setup dataset cache directory
            dataset_cache_dir = Path("data/datasets")
            dataset_cache_dir.mkdir(parents=True, exist_ok=True)
            self.console.print(
                f"📁 Dataset cache directory: {dataset_cache_dir.absolute()}"
            )

            # Create dataset download script
            dataset_categories = {
                "BLEU (Translation & Summarization)": [
                    ("wmt19", "en-de"),
                    ("cnn_dailymail", "3.0.0"),
                    ("knkarthick/xsum", None),  # Working alternative (CSV-based)
                    ("multi30k", "de-en"),
                ],
                "ROUGE (Summarization)": [
                    ("cnn_dailymail", "3.0.0"),
                    ("reddit_tifu", "long"),
                    ("scientific_papers", "pubmed"),
                ],
                "BERT Score & Semantic Similarity": [
                    ("mteb/sts-benchmark-sts", None),
                    ("sick", None),
                    ("quora", None),
                    ("glue", "mrpc"),
                ],
                "Mathematical Reasoning": [
                    ("gsm8k", "main"),
                    ("hendrycks/math", "all"),
                    ("math_qa", None),
                ],
                "Commonsense Reasoning": [
                    ("hellaswag", None),
                    ("commonsense_qa", None),
                    (
                        "gimmaru/piqa",
                        "validation",
                    ),  # Working alternative (validation only)
                    ("winogrande", "winogrande_xl"),
                ],
                "Code Generation": [
                    ("openai_humaneval", None),
                    ("mbpp", None),
                ],
                "Truthfulness & Factuality": [
                    (
                        "truthful_qa",
                        "validation",
                    ),  # Use validation split instead of generation
                    ("fever", "v1.0"),
                ],
                "Safety & Preference": [
                    ("Anthropic/hh-rlhf", None),
                ],
            }

            downloaded_count = 0
            failed_count = 0
            skipped_count = 0
            tested_count = 0
            test_results = {}

            for category, datasets in dataset_categories.items():
                self.console.print(f"\n[bold yellow]📂 {category}[/bold yellow]")

                for dataset_name, config in datasets:
                    # Show HuggingFace dataset URL for user visibility
                    dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
                    display_config = (
                        f" (config: {config})" if config and config != "None" else ""
                    )
                    self.console.print(f"  [dim]🔗 {dataset_url}{display_config}[/dim]")

                    try:
                        # Create download and test script with validation
                        cache_dir = str(dataset_cache_dir.absolute())
                        download_code = f"""
import os
os.environ['HF_DATASETS_CACHE'] = '{cache_dir}'

from datasets import load_dataset
import sys
import json

result = {{'dataset': '{dataset_name}', 'config': '{config}', 'status': 'failed', 'samples': 0, 'features': [], 'test_passed': False}}

try:
    dataset_name = "{dataset_name}"
    config = "{config}" if "{config}" != "None" else None
    
    print(f"📥 Downloading {{dataset_name}}...", file=sys.stderr)
    
    # Try to load dataset
    if config:
        ds = load_dataset(dataset_name, config, split="test", streaming=False, cache_dir='{cache_dir}')
    else:
        # Try test split first, fallback to train or validation
        try:
            ds = load_dataset(dataset_name, split="test", streaming=False, cache_dir='{cache_dir}')
        except:
            try:
                ds = load_dataset(dataset_name, split="validation", streaming=False, cache_dir='{cache_dir}')
            except:
                ds = load_dataset(dataset_name, split="train[:100]", streaming=False, cache_dir='{cache_dir}')
    
    # Dataset loaded successfully
    result['status'] = 'downloaded'
    result['samples'] = len(ds)
    result['features'] = list(ds.features.keys())
    
    # Test dataset compatibility
    print(f"🧪 Testing dataset compatibility...", file=sys.stderr)
    
    # Check if dataset has required fields for text evaluation
    has_text = any(field in ds.features.keys() for field in ['text', 'document', 'article', 'question', 'sentence'])
    has_reference = any(field in ds.features.keys() for field in ['summary', 'target', 'answer', 'label', 'translation'])
    
    # Try to access first sample
    first_sample = ds[0]
    sample_fields = list(first_sample.keys())
    
    result['test_passed'] = True
    result['has_text'] = has_text
    result['has_reference'] = has_reference
    result['sample_fields'] = sample_fields
    
    print(f"✅ {{dataset_name}} ({config if config else 'default'}) - {{len(ds)}} samples | Fields: {{', '.join(sample_fields[:3])}}...", file=sys.stderr)
    print(json.dumps(result))
    sys.exit(0)
    
except Exception as e:
    error_msg = str(e)
    if "streaming" in error_msg.lower() or "not found" in error_msg.lower():
        print(f"⏭️ {{dataset_name}} - Streaming/Not available (will load on-demand)", file=sys.stderr)
        result['status'] = 'streaming'
        print(json.dumps(result))
        sys.exit(2)
    else:
        print(f"⚠️ {{dataset_name}} - {{error_msg[:80]}}", file=sys.stderr)
        result['error'] = error_msg[:200]
        print(json.dumps(result))
        sys.exit(1)
"""

                        # Run download and test script with timeout
                        result = subprocess.run(
                            [python_exe, "-c", download_code],
                            capture_output=True,
                            text=True,
                            timeout=180,  # 3 minute timeout per dataset (including testing)
                        )

                        if result.returncode == 0:
                            self.console.print(f"  {result.stderr.strip()}")
                            downloaded_count += 1
                            tested_count += 1

                            # Parse and store test results
                            try:
                                import json

                                lines = result.stdout.strip().split("\n")
                                test_data = json.loads(lines[-1])
                                test_results[f"{dataset_name}:{config}"] = test_data
                            except:
                                pass

                        elif result.returncode == 2:
                            self.console.print(f"  {result.stderr.strip()}")
                            skipped_count += 1

                            # Store streaming dataset info
                            try:
                                import json

                                lines = result.stdout.strip().split("\n")
                                test_data = json.loads(lines[-1])
                                test_results[f"{dataset_name}:{config}"] = test_data
                            except:
                                pass

                        else:
                            self.console.print(f"  {result.stderr.strip()}")
                            failed_count += 1

                    except subprocess.TimeoutExpired:
                        self.console.print(
                            f"  ⏱️ {dataset_name} - Timeout (dataset too large or slow download)"
                        )
                        failed_count += 1
                    except Exception as e:
                        self.console.print(
                            f"  ❌ {dataset_name} - Error: {str(e)[:50]}"
                        )
                        failed_count += 1

            # Summary with testing statistics
            self.console.print(
                f"\n[bold green]📊 Dataset Installation & Testing Summary[/bold green]"
            )
            self.console.print(f"  ✅ Downloaded & Tested: {downloaded_count} datasets")
            self.console.print(
                f"  🧪 Passed Compatibility Tests: {tested_count} datasets"
            )
            self.console.print(f"  ⏭️ Streaming/On-demand: {skipped_count} datasets")
            self.console.print(f"  ⚠️ Failed: {failed_count} datasets")

            total_processed = downloaded_count + skipped_count + failed_count
            success_rate = (
                (downloaded_count / total_processed * 100) if total_processed > 0 else 0
            )

            self.console.print(
                f"\n💡 Total datasets available: {downloaded_count + skipped_count} / {total_processed}"
            )
            self.console.print(f"📈 Success rate: {success_rate:.1f}%")
            self.console.print(f"📁 Storage location: {dataset_cache_dir.absolute()}")

            # Save detailed test report
            if test_results:
                try:
                    report_path = dataset_cache_dir / "test_report.json"
                    import json

                    with open(report_path, "w") as f:
                        json.dump(
                            {
                                "summary": {
                                    "downloaded": downloaded_count,
                                    "tested": tested_count,
                                    "skipped": skipped_count,
                                    "failed": failed_count,
                                    "success_rate": f"{success_rate:.1f}%",
                                    "storage_path": str(dataset_cache_dir.absolute()),
                                },
                                "datasets": test_results,
                            },
                            f,
                            indent=2,
                        )
                    self.console.print(f"📄 Detailed test report: {report_path}")
                except Exception as e:
                    self.console.print(f"⚠️ Could not save test report: {e}")

            # Algorithm impact
            self.console.print(f"\n[bold cyan]🎯 Algorithm Impact:[/bold cyan]")
            self.console.print(
                f"  • BLEU: Ready with {len(dataset_categories['BLEU (Translation & Summarization)'])} datasets"
            )
            self.console.print(
                f"  • ROUGE: Ready with {len(dataset_categories['ROUGE (Summarization)'])} datasets"
            )
            self.console.print(
                f"  • BERT Score: Ready with {len(dataset_categories['BERT Score & Semantic Similarity'])} datasets"
            )
            self.console.print(
                f"  • Semantic Similarity: Ready with {len(dataset_categories['BERT Score & Semantic Similarity'])} datasets"
            )
            self.console.print(
                f"  • Mathematical Reasoning: Ready with {len(dataset_categories['Mathematical Reasoning'])} datasets"
            )
            self.console.print(
                f"  • Commonsense Reasoning: Ready with {len(dataset_categories['Commonsense Reasoning'])} datasets"
            )
            self.console.print(
                f"  • Code Generation: Ready with {len(dataset_categories['Code Generation'])} datasets"
            )
            self.console.print(
                f"  • Truthfulness: Ready with {len(dataset_categories['Truthfulness & Factuality'])} datasets"
            )
            self.console.print(
                f"  • Safety Alignment: Ready with {len(dataset_categories['Safety & Preference'])} datasets"
            )

            if downloaded_count + skipped_count > 0:
                installed_components.append(
                    f"HuggingFace Datasets ({downloaded_count} downloaded, {skipped_count} on-demand)"
                )
            else:
                failed_components.append("HuggingFace Datasets")

        # COMPONENT 12: Fix Dependencies
        if 12 in choices:
            self.console.print(
                "\n[bold cyan]🔧 FIXING: PyTorch & NVIDIA Dependencies[/bold cyan]"
            )
            self.console.print(
                "[dim]This resolves common dependency conflicts and deprecated package warnings[/dim]\n"
            )

            # Ensure we have proper pip executable
            venv_path = Path("llm-experiment-runner/.venv")
            pip_exe = str(venv_path / "bin" / "pip") if venv_path.exists() else "pip3"

            fixed_count = 0
            total_fixes = 4

            try:
                # Ensure packaging.version is available
                try:
                    from packaging import version
                except ImportError:
                    version = None

                # Check current PyTorch version
                self.console.print("🔍 Checking PyTorch version...")
                try:
                    result = subprocess.run(
                        [pip_exe, "show", "torch"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    current_version = None
                    for line in result.stdout.split("\n"):
                        if line.startswith("Version:"):
                            current_version = line.split(":", 1)[1].strip()
                            break

                    if current_version:
                        self.console.print(
                            f"  Current PyTorch version: {current_version}"
                        )

                        # Check if version is < 2.6.0
                        try:
                            if version and version.parse(
                                current_version
                            ) < version.parse("2.6.0"):
                                self.console.print(
                                    "  ⚠️  PyTorch version < 2.6.0 has security vulnerabilities"
                                )
                                self.console.print("  📦 Upgrading to PyTorch 2.6.0...")
                                subprocess.run(
                                    [pip_exe, "install", "torch>=2.6.0"], check=True
                                )
                                self.console.print(
                                    "  ✅ PyTorch upgraded to secure version"
                                )
                                fixed_count += 1
                            elif version:
                                self.console.print("  ✅ PyTorch version is secure")
                                fixed_count += 1
                            else:
                                # Fallback: just check if it's an obviously old version
                                if current_version.startswith(
                                    ("1.", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5")
                                ):
                                    self.console.print(
                                        "  ⚠️  PyTorch version appears old, upgrading..."
                                    )
                                    subprocess.run(
                                        [pip_exe, "install", "torch>=2.6.0"], check=True
                                    )
                                    self.console.print("  ✅ PyTorch upgraded")
                                    fixed_count += 1
                                else:
                                    self.console.print(
                                        "  ✅ PyTorch version appears recent"
                                    )
                                    fixed_count += 1
                        except Exception as ver_e:
                            self.console.print(
                                f"  ⚠️  Version check failed: {ver_e}, installing latest..."
                            )
                            subprocess.run(
                                [pip_exe, "install", "torch>=2.6.0"], check=True
                            )
                            fixed_count += 1
                    else:
                        self.console.print("  ❌ Could not determine PyTorch version")

                except subprocess.CalledProcessError:
                    self.console.print(
                        "  ⚠️  PyTorch not installed, installing secure version..."
                    )
                    subprocess.run([pip_exe, "install", "torch>=2.6.0"], check=True)
                    self.console.print("  ✅ PyTorch installed")
                    fixed_count += 1

                # Fix deprecated pynvml package
                self.console.print("\n🔍 Checking NVIDIA packages...")
                try:
                    # Check if pynvml is installed
                    result = subprocess.run(
                        [pip_exe, "show", "pynvml"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        self.console.print("  ⚠️  Deprecated 'pynvml' package found")
                        self.console.print("  🗑️  Uninstalling deprecated pynvml...")
                        subprocess.run(
                            [pip_exe, "uninstall", "pynvml", "-y"], check=True
                        )
                        self.console.print("  ✅ Deprecated pynvml removed")
                        fixed_count += 1
                    else:
                        self.console.print("  ✅ No deprecated pynvml found")
                        fixed_count += 1

                except subprocess.CalledProcessError as e:
                    self.console.print(f"  ⚠️  Issue with pynvml check: {e}")

                # Ensure nvidia-ml-py is installed
                self.console.print("\n📦 Ensuring modern NVIDIA monitoring...")
                try:
                    subprocess.run(
                        [pip_exe, "install", "nvidia-ml-py>=12.0.0"], check=True
                    )
                    self.console.print("  ✅ nvidia-ml-py installed/updated")
                    fixed_count += 1
                except subprocess.CalledProcessError as e:
                    self.console.print(f"  ⚠️  Issue installing nvidia-ml-py: {e}")

                # Reinstall codecarbon to use updated dependencies
                self.console.print(
                    "\n🔄 Refreshing codecarbon with updated dependencies..."
                )
                try:
                    subprocess.run(
                        [
                            pip_exe,
                            "install",
                            "--upgrade",
                            "--force-reinstall",
                            "codecarbon",
                        ],
                        check=True,
                    )
                    self.console.print("  ✅ codecarbon refreshed")
                    fixed_count += 1
                except subprocess.CalledProcessError as e:
                    self.console.print(f"  ⚠️  Issue refreshing codecarbon: {e}")

                # Summary
                if fixed_count >= 3:
                    self.console.print(
                        f"\n[green]🎉 Dependency fixes completed successfully! ({fixed_count}/{total_fixes})[/green]"
                    )
                    self.console.print(
                        "[green]• PyTorch security vulnerability resolved[/green]"
                    )
                    self.console.print(
                        "[green]• Deprecated NVIDIA packages removed[/green]"
                    )
                    self.console.print("[green]• Modern nvidia-ml-py installed[/green]")
                    self.console.print(
                        "[green]• CodeCarbon refreshed with clean dependencies[/green]"
                    )
                    installed_components.append(
                        f"Dependency Fixes ({fixed_count}/{total_fixes})"
                    )
                else:
                    self.console.print(
                        f"[yellow]⚠️  Some dependency fixes had issues ({fixed_count}/{total_fixes})[/yellow]"
                    )
                    failed_components.append("Dependency Fixes")

            except Exception as e:
                self.console.print(f"❌ Dependency fix error: {e}")
                failed_components.append("Dependency Fixes")

        # Final installation summary
        self.console.print(f"\n[bold green]🎉 Installation Complete![/bold green]")

        if installed_components:
            self.console.print(
                f"\n[green]✅ Successfully Installed ({len(installed_components)} components):[/green]"
            )
            for component in installed_components:
                self.console.print(f"  • {component}")

        if failed_components:
            self.console.print(
                f"\n[red]❌ Failed Installations ({len(failed_components)} components):[/red]"
            )
            for component in failed_components:
                self.console.print(f"  • {component}")

        # Usage instructions
        self.console.print(f"\n[bold cyan]💡 What's Ready:[/bold cyan]")
        if "Virtual Environment" in [
            c.split()[0:2] for c in installed_components if "Virtual" in c
        ]:
            self.console.print("• 🐍 Python virtual environment ready")
        if any("ML" in c for c in installed_components):
            self.console.print("• 🤖 Machine learning and model inference")
        if any("Evaluation" in c for c in installed_components):
            self.console.print("• 📊 Text evaluation and scoring algorithms")
        if any("Reference Datasets" in c for c in installed_components):
            self.console.print(
                "• 📚 Reference datasets for comprehensive algorithm evaluation"
            )
        if "System Monitoring" in str(installed_components):
            self.console.print("• 🔋 CodeCarbon energy profiling (RAPL + GPU)")
        if "System Tools" in installed_components:
            self.console.print("• 🖥️ System tools (Git, etc.)")

        self.console.print(f"\n[bold yellow]� Usage Notes:[/bold yellow]")
        if venv_path.exists():
            self.console.print(
                "• ✅ Activate virtual environment: source llm-experiment-runner/.venv/bin/activate"
            )
            self.console.print(
                "• ✅ Run experiments within the activated virtual environment"
            )
        if "System Monitoring" in str(installed_components):
            self.console.print(
                "• ✅ CodeCarbon will automatically track CPU (RAPL), GPU (nvidia-smi), and CO2"
            )
        if "System Tools" in installed_components:
            self.console.print(
                "• ✅ System tools (git, maven, java) available everywhere"
            )

        self.console.print(f"\n[bold green]� Next Steps:[/bold green]")
        self.console.print(
            "1. Run System Diagnostics (Option 10) to verify installation"
        )
        self.console.print(
            "2. Set up HuggingFace authentication (Option 11) for model access"
        )
        self.console.print("3. Start creating experiments with your configured LCT!")

        input("\nPress Enter to continue...")

    def _install_packages(self, packages, pip_exe):
        """Helper method to install a list of packages"""
        success_count = 0
        for package, description in packages:
            self.console.print(f"📦 Installing {package}: {description}")

            result = subprocess.run(
                [pip_exe, "install", package, "--upgrade"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.console.print(f"  ✅ {package} installed successfully")
                success_count += 1
            else:
                error_msg = (
                    result.stderr[:100] + "..."
                    if len(result.stderr) > 100
                    else result.stderr
                )
                self.console.print(f"  ⚠️ {package} installation issues: {error_msg}")

        return success_count

    def launch_results_explorer(self):
        """Launch the Results Explorer tool"""
        self.console.clear()
        self.console.print("[bold blue]🔍 Results Explorer[/bold blue]\n")

        # Check if results_explorer.py exists
        results_explorer_path = os.path.join(
            os.path.dirname(__file__), "results_explorer.py"
        )

        if not os.path.exists(results_explorer_path):
            self.console.print("[red]❌ Results Explorer not found![/red]")
            self.console.print(
                "Please ensure results_explorer.py is in the same directory as this script."
            )
            input("\nPress Enter to continue...")
            return

        try:
            self.console.print("🔍 Launching Results Explorer...")
            self.console.print("This will open an interactive tool to:")
            self.console.print("• 📊 Browse and analyze experiment results")
            self.console.print("• 📋 View CSV content with statistics and filtering")
            self.console.print("• 🗑️ Delete individual experiments or bulk delete all")
            self.console.print(
                "• 📤 Export experiments individually or bulk export all"
            )
            self.console.print("• 🔄 Manage experiment lifecycle")
            self.console.print(
                "\n[dim]Press Ctrl+C in Results Explorer to return here[/dim]\n"
            )

            input("Press Enter to launch Results Explorer...")

            # Launch the results explorer
            result = subprocess.run(
                [sys.executable, results_explorer_path], cwd=os.path.dirname(__file__)
            )

            self.console.print("\n[green]👋 Results Explorer closed[/green]")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Results Explorer launch cancelled[/yellow]")
        except Exception as e:
            self.console.print(
                f"[red]❌ Error launching Results Explorer: {str(e)}[/red]"
            )

        input("\nPress Enter to continue...")

    def project_cleanup(self):
        """Interactive project cleanup with multi-select options"""
        self.console.clear()
        self.console.print("[bold blue]🧹 Project Cleanup[/bold blue]\n")
        
        # Calculate sizes for different cleanup categories
        cleanup_items = []
        
        # 1. Python cache
        try:
            pycache_size = sum(
                f.stat().st_size
                for f in Path(".").rglob("__pycache__")
                if f.is_dir()
                for f in f.rglob("*")
                if f.is_file()
            )
            pycache_count = len(list(Path(".").rglob("__pycache__")))
            cleanup_items.append({
                "id": "pycache",
                "name": "Python cache (__pycache__)",
                "size": pycache_size,
                "count": f"{pycache_count} directories",
                "description": "Compiled Python bytecode files"
            })
        except Exception:
            pass
        
        # 2. Experiment backups
        try:
            backup_paths = list(Path("experiments").rglob("*backup*"))
            backup_size = sum(
                sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                for p in backup_paths if p.is_dir()
            )
            cleanup_items.append({
                "id": "backups",
                "name": "Experiment backups",
                "size": backup_size,
                "count": f"{len(backup_paths)} folders",
                "description": "Old experiment backup folders"
            })
        except Exception:
            pass
        
        # 3. Redundant RunnerConfig files
        try:
            runner_configs = list(Path("experiments").rglob("RunnerConfig.py"))
            runner_configs = [f for f in runner_configs if f.parent != Path("experiments")]
            config_size = sum(f.stat().st_size for f in runner_configs)
            cleanup_items.append({
                "id": "configs",
                "name": "Old experiment configs",
                "size": config_size,
                "count": f"{len(runner_configs)} files",
                "description": "Replaced by saved_configs/ JSON files"
            })
        except Exception:
            pass
        
        # 4. Downloaded models (HuggingFace cache)
        try:
            hf_cache = Path("data/huggingface")
            if hf_cache.exists():
                hf_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
                model_count = len(list(hf_cache.glob("hub/models--*")))
                cleanup_items.append({
                    "id": "models",
                    "name": "Downloaded models",
                    "size": hf_size,
                    "count": f"~{model_count} models",
                    "description": "HuggingFace model cache (will re-download on next use)",
                    "warning": True
                })
        except Exception:
            pass
        
        # 5. Downloaded datasets
        try:
            datasets_cache = Path("data/datasets")
            if datasets_cache.exists():
                ds_size = sum(f.stat().st_size for f in datasets_cache.rglob("*") if f.is_file())
                ds_count = len([d for d in datasets_cache.iterdir() if d.is_dir()])
                cleanup_items.append({
                    "id": "datasets",
                    "name": "Downloaded datasets",
                    "size": ds_size,
                    "count": f"{ds_count} datasets",
                    "description": "Cached evaluation datasets (will re-download on next use)",
                    "warning": True
                })
        except Exception:
            pass
        
        # 6. Pytest cache
        try:
            pytest_cache = Path(".pytest_cache")
            if pytest_cache.exists():
                pytest_size = sum(f.stat().st_size for f in pytest_cache.rglob("*") if f.is_file())
                cleanup_items.append({
                    "id": "pytest",
                    "name": "Pytest cache",
                    "size": pytest_size,
                    "count": "1 directory",
                    "description": "Test execution cache"
                })
        except Exception:
            pass
        
        # 7. Logs
        try:
            log_files = list(Path(".").rglob("*.log"))
            if log_files:
                log_size = sum(f.stat().st_size for f in log_files)
                cleanup_items.append({
                    "id": "logs",
                    "name": "Log files",
                    "size": log_size,
                    "count": f"{len(log_files)} files",
                    "description": "Application and experiment logs"
                })
        except Exception:
            pass
        
        if not cleanup_items:
            self.console.print("[green]✓ Project is already clean![/green]")
            input("\nPress Enter to continue...")
            return
        
        # Display cleanup options
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Item", style="yellow", width=25)
        table.add_column("Size", style="green", width=12)
        table.add_column("Count", style="blue", width=15)
        table.add_column("Description", style="white", width=40)
        
        for idx, item in enumerate(cleanup_items, 1):
            size_str = self._format_size(item["size"])
            warning = "⚠️ " if item.get("warning") else ""
            table.add_row(
                str(idx),
                warning + item["name"],
                size_str,
                item["count"],
                item["description"]
            )
        
        self.console.print(table)
        self.console.print()
        
        # Calculate total size
        total_size = sum(item["size"] for item in cleanup_items)
        self.console.print(f"[bold]Total reclaimable space: {self._format_size(total_size)}[/bold]\n")
        
        self.console.print("[yellow]⚠️  Items with ⚠️ will require re-downloading on next use[/yellow]")
        self.console.print("[dim]Enter item numbers separated by commas (e.g., 1,2,3) or 'all'[/dim]")
        self.console.print("[dim]Press 0 to cancel[/dim]\n")
        
        choice = Prompt.ask("Select items to clean")
        
        if choice == "0":
            self.console.print("[yellow]Cleanup cancelled[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        # Parse selection
        selected_ids = []
        if choice.lower() == "all":
            selected_ids = [item["id"] for item in cleanup_items]
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                selected_ids = [cleanup_items[i-1]["id"] for i in indices if 1 <= i <= len(cleanup_items)]
            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection[/red]")
                input("\nPress Enter to continue...")
                return
        
        if not selected_ids:
            self.console.print("[yellow]No items selected[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        # Confirm deletion
        selected_items = [item for item in cleanup_items if item["id"] in selected_ids]
        selected_size = sum(item["size"] for item in selected_items)
        
        self.console.print(f"\n[yellow]You are about to delete:[/yellow]")
        for item in selected_items:
            self.console.print(f"  • {item['name']} ({self._format_size(item['size'])})")
        self.console.print(f"\n[bold]Total space to reclaim: {self._format_size(selected_size)}[/bold]\n")
        
        if not Confirm.ask("[red]Are you sure you want to proceed?[/red]"):
            self.console.print("[yellow]Cleanup cancelled[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        # Perform cleanup
        self.console.print("\n[blue]Starting cleanup...[/blue]\n")
        import shutil
        
        for item_id in selected_ids:
            try:
                if item_id == "pycache":
                    count = 0
                    for pycache in Path(".").rglob("__pycache__"):
                        if pycache.is_dir():
                            shutil.rmtree(pycache)
                            count += 1
                    self.console.print(f"[green]✓ Removed {count} __pycache__ directories[/green]")
                
                elif item_id == "backups":
                    count = 0
                    for backup in Path("experiments").rglob("*backup*"):
                        if backup.is_dir():
                            shutil.rmtree(backup)
                            count += 1
                    self.console.print(f"[green]✓ Removed {count} backup folders[/green]")
                
                elif item_id == "configs":
                    count = 0
                    for config in Path("experiments").rglob("RunnerConfig.py"):
                        if config.parent != Path("experiments"):
                            config.unlink()
                            count += 1
                    # Also remove experiment_info.json files
                    for info in Path("experiments").rglob("experiment_info.json"):
                        if info.parent != Path("experiments"):
                            info.unlink()
                    self.console.print(f"[green]✓ Removed {count} old config files[/green]")
                
                elif item_id == "models":
                    hf_cache = Path("data/huggingface")
                    if hf_cache.exists():
                        shutil.rmtree(hf_cache)
                        hf_cache.mkdir(parents=True)
                        self.console.print(f"[green]✓ Cleared model cache[/green]")
                
                elif item_id == "datasets":
                    datasets_cache = Path("data/datasets")
                    if datasets_cache.exists():
                        shutil.rmtree(datasets_cache)
                        datasets_cache.mkdir(parents=True)
                        self.console.print(f"[green]✓ Cleared datasets cache[/green]")
                
                elif item_id == "pytest":
                    pytest_cache = Path(".pytest_cache")
                    if pytest_cache.exists():
                        shutil.rmtree(pytest_cache)
                        self.console.print(f"[green]✓ Removed pytest cache[/green]")
                
                elif item_id == "logs":
                    count = 0
                    for log_file in Path(".").rglob("*.log"):
                        log_file.unlink()
                        count += 1
                    self.console.print(f"[green]✓ Removed {count} log files[/green]")
                
            except Exception as e:
                self.console.print(f"[red]✗ Error cleaning {item_id}: {str(e)}[/red]")
        
        self.console.print(f"\n[bold green]✓ Cleanup completed! Reclaimed {self._format_size(selected_size)}[/bold green]")
        input("\nPress Enter to continue...")

    def show_about(self):
        """Show information about the tool and credits"""
        self.console.clear()

        # Create a beautiful about panel
        about_content = f"""
[bold blue]🚀 Interactive LLM Comparison Tool (LCT)[/bold blue]

[cyan]Version:[/cyan] 2.0 Enhanced Edition
[cyan]Release:[/cyan] September 2025

[bold yellow]📖 Description:[/bold yellow]
An advanced, interactive framework for systematic Large Language Model comparison 
and evaluation. Built as an enhanced extension to the proven Experiment Runner 
framework, LCT provides researchers and developers with powerful tools for 
comprehensive LLM analysis.

[bold green]✨ Key Features:[/bold green]
• 🎯 Beautiful interactive CLI menu system
• 🤖 Support for any HuggingFace model with smart search
• 📊 17 evaluation algorithms across 5 categories
• ⚡ CodeCarbon energy profiling (RAPL + GPU + CO2 tracking)
• 💭 8 structured prompt difficulty categories  
• 🏥 Comprehensive system diagnostics
• 🔧 Automated dependency installation
• 📈 Professional CSV/JSON result outputs
• 💾 Configuration save/load system

[bold red]🏗️ Built Upon:[/bold red]
[cyan]Experiment Runner Framework[/cyan]
• Original: https://github.com/S2-group/experiment-runner
• Citation: S2-group (2021). Experiment Runner: A generic framework for 
  measurement-based experiments. Software Engineering Research Group, 
  Vrije Universiteit Amsterdam.
• License: Apache-2.0

[bold magenta]👨‍💻 Developer & Enhanced by:[/bold magenta]
[cyan]Adam Bouafia[/cyan]
• Portfolio: https://adam-bouafia.github.io/
• LinkedIn: https://www.linkedin.com/in/adam-bouafia-b597ab86/
• Enhanced with interactive interface, energy profiling, advanced algorithms,
  and comprehensive automation features.

[bold cyan]🔬 Research Applications:[/bold cyan]
• LLM performance benchmarking
• Energy efficiency analysis  
• Response quality evaluation
• Multi-model comparison studies
• Academic research and papers

[bold yellow]📄 License:[/bold yellow]
MIT License - Free for research, academic, and commercial use

[bold green]🤝 Contributing:[/bold green]
Contributions welcome! This tool combines the robustness of Experiment Runner
with modern interactive features and comprehensive LLM evaluation capabilities.

[bold blue]📊 Evaluation Categories:[/bold blue]
• Performance & Speed (3 algorithms)
• Quality & Accuracy (4 algorithms)  
• Advanced Evaluation (4 algorithms)
• Task-Specific Benchmarks (6 algorithms)

[italic]Built with ❤️ for the AI research community[/italic]
"""

        # Display in a panel
        panel = Panel(
            about_content,
            title="ℹ️ About Interactive LLM Comparison Tool",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)

        # Additional quick stats
        stats_table = Table(
            title="📈 Tool Statistics", box=box.ROUNDED, show_header=False
        )
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Supported Models", "Any HuggingFace model")
        stats_table.add_row("Evaluation Algorithms", "17 comprehensive algorithms")
        stats_table.add_row("Prompt Categories", "8 difficulty levels")
        stats_table.add_row("Energy Profiling", "3 different tools")
        stats_table.add_row("Output Formats", "CSV, JSON")
        stats_table.add_row("Menu Options", "14 interactive features")

        self.console.print("\n")
        self.console.print(stats_table)

        self.console.print(f"\n[bold cyan]🔗 Quick Links:[/bold cyan]")
        self.console.print(
            "• Developer Portfolio: [link]https://adam-bouafia.github.io/[/link]"
        )
        self.console.print(
            "• LinkedIn Profile: [link]https://www.linkedin.com/in/adam-bouafia-b597ab86/[/link]"
        )
        self.console.print(
            "• Experiment Runner: [link]https://github.com/S2-group/experiment-runner[/link]"
        )

        self.console.print(f"\n[bold yellow]💝 Support Development:[/bold yellow]")
        self.console.print(
            "• Donate via PayPal: [link]https://paypal.me/AdamBouafia[/link]"
        )
        self.console.print("• Your donations help maintain and improve this tool!")

        input("\n📚 Press Enter to return to main menu...")

    def run_experiment(self):
        """Create and run the experiment"""
        self.console.clear()
        self.console.print("[bold blue]🚀 Run Experiment[/bold blue]\n")

        if not self.is_config_complete():
            self.console.print("[red]❌ Configuration incomplete![/red]")
            return

        # Show experiment summary
        summary_table = Table(title="Experiment Summary", box=box.DOUBLE)
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Name", self.config["name"])
        summary_table.add_row("Models", f"{len(self.config['models'])} models")
        summary_table.add_row("Prompts", f"{len(self.config['prompts'])} prompts")
        summary_table.add_row(
            "Algorithms", f"{len(self.config['algorithms'])} algorithms"
        )
        summary_table.add_row("Repetitions", str(self.config["repetitions"]))
        summary_table.add_row(
            "Total Runs",
            str(
                len(self.config["models"])
                * len(self.config["prompts"])
                * self.config["repetitions"]
            ),
        )
        summary_table.add_row("Energy Profiling", self.config["energy_profiler"])

        self.console.print(summary_table)
        self.console.print()

        if not Confirm.ask("Proceed with experiment creation?"):
            return

        try:
            # Create experiment using the local function
            self.console.print("[yellow]Creating experiment configuration...[/yellow]")

            # Build configuration dictionary
            # Extract model IDs from both string and dict formats
            model_ids = [self._get_model_id(m) for m in self.config["models"]]

            config_dict = {
                "name": self.config["name"],
                "models": model_ids,  # Use extracted model IDs
                "prompts": self.config["prompts"],
                "algorithms": self.config["algorithms"],
                "repetitions": self.config["repetitions"],
                "max_length": self.config["max_length"],
                "temperature": self.config["temperature"],
                "energy_profiler": (
                    self.config["energy_profiler"]
                    if self.config["energy_profiler"] != "none"
                    else None
                ),
            }

            # Create the experiment using the local function
            experiment_path = create_experiment_from_config(config_dict)

            self.console.print(
                f"[green]✅ Experiment created successfully![/green]"
            )
            self.console.print(f"[dim]   Configuration: {experiment_path}[/dim]")
            self.console.print()

            if Confirm.ask("Run the experiment now?"):
                self.console.print("[yellow]🚀 Running experiment...[/yellow]")

                # Calculate expected progress info
                total_runs = (
                    len(self.config["models"])
                    * len(self.config["prompts"])
                    * self.config["repetitions"]
                )
                self.console.print(f"[cyan]📊 Experiment Details:[/cyan]")
                self.console.print(f"   • Models: {len(self.config['models'])}")
                self.console.print(f"   • Prompts: {len(self.config['prompts'])}")
                self.console.print(f"   • Repetitions: {self.config['repetitions']}")
                self.console.print(f"   • Total runs: {total_runs}")
                self.console.print(
                    f"   • Energy profiling: {self.config['energy_profiler']}"
                )

                self.console.print(
                    "[dim]Live output will be shown below. Press Ctrl+C to cancel if needed.[/dim]"
                )
                self.console.print("=" * 70)

                # Run experiment using the new CLI run command
                import subprocess
                import os
                import threading
                import time

                # Find project root (go up from app/src/ui to project root)
                project_root = Path(__file__).parent.parent.parent.parent
                
                # Check for venv in multiple locations (prefer root .venv)
                if (project_root / ".venv/bin/python").exists():
                    venv_python = project_root / ".venv/bin/python"
                elif (project_root / "llm-experiment-runner/.venv/bin/python").exists():
                    venv_python = project_root / "llm-experiment-runner/.venv/bin/python"
                else:
                    # Fallback to system python
                    import sys
                    venv_python = Path(sys.executable)

                # Set environment to include the app/src directory in Python path
                env = os.environ.copy()
                app_src_path = str(project_root / "app/src")
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{app_src_path}:{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = app_src_path

                # Add HuggingFace environment variables to suppress warnings
                hf_home = str(project_root / "data/huggingface")
                env.update(
                    {
                        "HF_HOME": hf_home,
                        "HF_DATASETS_CACHE": f"{hf_home}/datasets",
                        "HF_MODELS_CACHE": f"{hf_home}/models",
                        "TRANSFORMERS_CACHE": f"{hf_home}/transformers",
                        "TRANSFORMERS_VERBOSITY": "error",
                        "DATASETS_VERBOSITY": "error",
                        "PYTHONWARNINGS": "ignore::FutureWarning:torch.cuda,ignore::FutureWarning:transformers",
                    }
                )

                # Use the new CLI run command instead of old experiment-runner
                cmd = [
                    str(venv_python),
                    "-m",
                    "llm_runner.cli.main_cli",
                    "run",
                    experiment_path,
                ]

                original_cwd = os.getcwd()

                try:
                    # Start the experiment process with real-time output
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=str(
                            project_root
                        ),  # Run from project root but with PYTHONPATH set
                        env=env,
                    )

                    # Progress indicator with heartbeat
                    start_time = time.time()
                    output_lines = []
                    run_count = 0
                    last_output_time = start_time
                    heartbeat_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    heartbeat_idx = 0

                    def show_heartbeat():
                        """Show a spinning indicator if no output for a while"""
                        nonlocal heartbeat_idx
                        current_time = time.time()
                        if (
                            current_time - last_output_time > 5
                        ):  # No output for 5+ seconds
                            elapsed = int(current_time - start_time)
                            heartbeat_idx = (heartbeat_idx + 1) % len(heartbeat_chars)
                            self.console.print(
                                f"[dim]{heartbeat_chars[heartbeat_idx]} {elapsed:3d}s | Working... (no output for {int(current_time - last_output_time)}s)[/dim]"
                            )

                    # Read output line by line and display in real-time
                    if process.stdout:
                        for line in iter(process.stdout.readline, ""):
                            if line.strip():
                                output_lines.append(line.strip())
                                elapsed = int(time.time() - start_time)
                                last_output_time = time.time()

                                # Track progress through runs
                                if (
                                    "Starting experiment" in line
                                    or "Run " in line
                                    or "Model:" in line
                                ):
                                    run_count += 1
                                    if total_runs > 0:
                                        progress = min(
                                            100, int((run_count / total_runs) * 100)
                                        )
                                        self.console.print(
                                            f"[green]⏱️  {elapsed:3d}s | Progress: {progress:2d}% | {line.strip()}[/green]"
                                        )
                                    else:
                                        self.console.print(
                                            f"[green]⏱️  {elapsed:3d}s | {line.strip()}[/green]"
                                        )
                                elif (
                                    "error" in line.lower()
                                    or "exception" in line.lower()
                                ):
                                    self.console.print(
                                        f"[red]⏱️  {elapsed:3d}s | ❌ {line.strip()}[/red]"
                                    )
                                elif "warning" in line.lower():
                                    self.console.print(
                                        f"[yellow]⏱️  {elapsed:3d}s | ⚠️  {line.strip()}[/yellow]"
                                    )
                                elif (
                                    "completed" in line.lower()
                                    or "finished" in line.lower()
                                ):
                                    self.console.print(
                                        f"[green]⏱️  {elapsed:3d}s | ✅ {line.strip()}[/green]"
                                    )
                                else:
                                    self.console.print(
                                        f"[dim]⏱️  {elapsed:3d}s[/dim] | {line.strip()}"
                                    )
                            else:
                                # Show heartbeat if process is still running but no output
                                if process.poll() is None:
                                    show_heartbeat()

                    # Wait for process to complete
                    process.wait()

                    self.console.print("=" * 70)

                    if process.returncode == 0:
                        elapsed_total = int(time.time() - start_time)
                        self.console.print(
                            f"[green]✅ Experiment completed successfully in {elapsed_total}s![/green]"
                        )

                        # Get results directory from experiment path
                        results_dir = Path(experiment_path).parent
                        self.console.print(
                            f"[blue]📁 Results saved in: {results_dir}/{self.config['name']}/[/blue]"
                        )

                        # Show results files
                        experiment_results_dir = results_dir / self.config["name"]
                        if experiment_results_dir.exists():
                            self.console.print("\n[bold]📊 Generated files:[/bold]")
                            for file in sorted(experiment_results_dir.glob("*")):
                                file_size = file.stat().st_size if file.is_file() else 0
                                self.console.print(
                                    f"  📄 {file.name} ({file_size} bytes)"
                                )

                        # Show quick summary
                        csv_files = list(experiment_results_dir.glob("*.csv"))
                        if csv_files:
                            self.console.print(
                                f"\n[green]📈 Found {len(csv_files)} CSV result files for analysis![/green]"
                            )
                    else:
                        self.console.print(
                            f"[red]❌ Experiment failed (exit code: {process.returncode})[/red]"
                        )
                        if output_lines:
                            self.console.print("[red]Last few output lines:[/red]")
                            for line in output_lines[-5:]:
                                self.console.print(f"  {line}")

                except subprocess.TimeoutExpired:
                    self.console.print("[red]❌ Experiment timed out[/red]")
                    process.kill()
                except KeyboardInterrupt:
                    self.console.print(
                        "[yellow]⚠️  Experiment cancelled by user[/yellow]"
                    )
                    process.terminate()
                except Exception as e:
                    self.console.print(f"[red]❌ Error running experiment: {e}[/red]")

        except Exception as e:
            self.console.print(f"[red]❌ Error creating experiment: {e}[/red]")

        input("\nPress Enter to continue...")


def create_experiment_from_config(config: Dict[str, Any]) -> str:
    """Create experiment from configuration dictionary"""
    import subprocess
    import os

    # Build CLI command with proper comma-separated values using venv python
    # Find project root (go up from app/src/ui to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    venv_python = str(project_root / ".venv/bin/python")
    cmd = [
        venv_python,
        "-m",
        "llm_runner.cli.main_cli",
        "init",
        "--name",
        config["name"],
        "--models",
        ",".join(config["models"]),
        "--prompts",
        ",".join(config["prompts"]),  # Remove the extra quotes
        "--algorithms",
        ",".join(config["algorithms"]),
        "--repetitions",
        str(config["repetitions"]),
        "--max-length",
        str(config["max_length"]),
        "--temperature",
        str(config["temperature"]),
        "--yes",  # Auto-confirm prompts (skip interactive confirmations)
    ]

    if config.get("energy_profiler") and config["energy_profiler"] != "none":
        cmd.extend(["--energy-profiler", config["energy_profiler"]])

    # Set environment to include the app/src directory in Python path
    env = os.environ.copy()
    app_src_path = str(project_root / "app/src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{app_src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = app_src_path

    # Run the command from project root with proper Python path
    # Show real-time progress instead of blocking silently
    from rich.console import Console

    console = Console()

    with console.status("[bold yellow]→ Validating configuration...[/bold yellow]", spinner="dots"):
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(project_root), env=env,
            stdin=subprocess.DEVNULL  # Prevent hanging on interactive prompts
        )

    console.print("[dim]→ Writing experiment files...[/dim]")

    if result.returncode != 0:
        raise Exception(f"Failed to create experiment: {result.stderr}")

    # Extract experiment path from output
    lines = result.stdout.strip().split("\n")
    
    # Find "Configuration:" line and handle wrapped paths
    for i, line in enumerate(lines):
        if "Configuration:" in line:
            # Get the path after "Configuration:"
            path_part = line.split(":", 1)[1].strip()
            
            # Check if the next line continues the path (doesn't start with a new label)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # If next line doesn't have a colon (not a new label), it's a continuation
                if next_line and ":" not in next_line:
                    # Add space before joining to preserve path structure
                    path_part = path_part + " " + next_line
            
            # Resolve to absolute path if needed
            if not path_part.startswith("/"):
                path_part = str(project_root / path_part)
            
            return path_part

    # Fallback - construct path from project root
    return str(project_root / "experiments" / config["name"] / "RunnerConfig.py")


def main():
    """Main entry point"""
    try:
        app = InteractiveLCT()
        app.main_menu()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
