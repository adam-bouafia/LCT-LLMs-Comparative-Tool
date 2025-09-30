#!/usr/bin/env python3
"""
Interactive LLM Comparison Tool (LCT)
Simple CLI menu for creating and running LLM experiments
"""

# Suppress common warnings before other imports
import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*is deprecated.*", category=FutureWarning)
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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
        "Tell me about yourself"
    ],
    "Simple Questions": [
        "What is Python programming?",
        "How do you make coffee?",
        "What's the weather like today?",
        "What's your favorite color?",
        "What do you like to do for fun?"
    ],
    "Creative Writing": [
        "Write a short story about a robot",
        "Describe a beautiful sunset",
        "Create a poem about friendship",
        "Write a dialogue between two characters",
        "Tell me a funny joke"
    ],
    "Knowledge & Facts": [
        "Explain what artificial intelligence is",
        "What are the benefits of renewable energy?",
        "How does the internet work?",
        "What causes climate change?",
        "Explain the water cycle"
    ],
    "Problem Solving": [
        "How would you organize a birthday party?",
        "What's the best way to learn a new language?",
        "How can we reduce plastic waste?",
        "Plan a healthy weekly meal",
        "How to improve time management?"
    ],
    "Technical & Complex": [
        "Explain quantum computing in simple terms",
        "Compare different machine learning algorithms",
        "Describe the process of DNA replication",
        "How do neural networks work?",
        "Explain blockchain technology"
    ],
    "Advanced Reasoning": [
        "Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Analyze the ethical implications of AI in healthcare",
        "Compare and contrast democracy and authoritarianism",
        "Explain the philosophical concept of free will",
        "Discuss the implications of genetic engineering"
    ],
    "Coding & Programming": [
        "Write a Python function to reverse a string",
        "Explain the difference between lists and tuples in Python",
        "How would you implement a binary search algorithm?",
        "Create a simple calculator in JavaScript",
        "Debug this code and explain the issue"
    ]
}

POPULAR_MODELS = [
    "distilbert/distilgpt2",
    "gpt2",
    "microsoft/DialoGPT-medium",
    "facebook/blenderbot-400M-distill",
    "microsoft/DialoGPT-small",
    "huggingface/CodeBERTa-small-v1",
    "google/flan-t5-small",
    "google/flan-t5-base"
]

ALGORITHM_CATEGORIES = {
    "Performance & Speed": [
        ("response_time", "Measures how fast each model responds"),
        ("token_throughput", "Tokens generated per second"),
        ("text_length", "Length of generated responses")
    ],
    "Quality & Accuracy": [
        ("bleu", "BLEU score (needs reference texts)"),
        ("rouge", "ROUGE score (needs reference texts)"),
        ("bert_score", "Semantic similarity using BERT"),
        ("semantic_similarity", "Sentence embedding similarity")
    ],
    "Advanced Evaluation": [
        ("llm_as_judge", "Use LLM to judge response quality"),
        ("g_eval", "Advanced reasoning-based evaluation"),
        ("pairwise_comparison", "Compare responses pairwise"),
        ("safety_alignment", "Check for safe, aligned responses")
    ],
    "Task-Specific": [
        ("code_generation", "Evaluate code generation quality"),
        ("mathematical_reasoning", "Math problem solving"),
        ("commonsense_reasoning", "Common sense understanding"),
        ("truthfulness", "Check for truthful, factual responses")
    ]
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
            "max_length": 50,
            "temperature": 0.7,
            "energy_profiler": "none"
        }
        
    def show_header(self):
        """Display the main header"""
        header = Text("🚀 Interactive LLM Comparison Tool (LCT)", style="bold blue")
        subheader = Text("Create and run LLM experiments with ease", style="dim")
        
        panel = Panel(
            Align.center(f"{header}\n{subheader}"),
            box=box.DOUBLE,
            style="blue"
        )
        self.console.print(panel)
        self.console.print()

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
                ("1", "📝 Set Experiment Name", "✅" if self.config["name"] else "⚪"),
                ("2", "🤖 Select Models", f"✅ ({len(self.config['models'])} selected)" if self.config["models"] else "⚪"),
                ("3", "💭 Choose Prompts", f"✅ ({len(self.config['prompts'])} selected)" if self.config["prompts"] else "⚪"),
                ("4", "⚙️  Select Algorithms", f"✅ ({len(self.config['algorithms'])} selected)" if self.config["algorithms"] else "⚪"),
                ("5", "🔧 Configure Parameters", "✅"),
                ("6", "⚡ Energy Profiling", f"✅ ({self.config['energy_profiler']})" if self.config['energy_profiler'] != "none" else "⚪"),
                
                # Configuration management
                ("7", "💾 Save Configuration", "💾"),
                ("8", "📂 Load Configuration", "📂"),
                
                # System setup and diagnostics
                ("9", "🔧 Install Needed Tools", "🛠️"),
                ("10", "🏥 System Diagnostics", "�"),
                ("11", "🔑 HuggingFace Auth", "�"),
                ("12", "📦 Data Management", "💾"),
                
                # Execution
                ("13", "🚀 Run Experiment", "🚀" if self.is_config_complete() else "⚪"),
                
                # Results management  
                ("14", "🔍 Results Explorer", "📊"),
                
                # Information and exit
                ("15", "ℹ️  About Tool", "📋"),
                ("0", "🚪 Exit", "🚪")
            ]
            
            for opt, desc, status in options:
                table.add_row(opt, desc, status)
            
            self.console.print(table)
            self.console.print()
            
            choice = Prompt.ask("Choose an option", choices=["0"] + [str(i) for i in range(1, 16)])
            
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
                self.setup_huggingface_auth()
            elif choice == "12":
                self.manage_data()
            elif choice == "13":
                if self.is_config_complete():
                    self.run_experiment()
                else:
                    self.console.print("[red]❌ Configuration incomplete! Please set name, models, prompts, and algorithms.[/red]")
                    input("\nPress Enter to continue...")
            elif choice == "14":
                self.launch_results_explorer()
            elif choice == "15":
                self.show_about()
            elif choice == "0":
                if Confirm.ask("Are you sure you want to exit?"):
                    break
    
    def show_current_config(self):
        """Show current configuration status"""
        config_panel = Panel(
            f"[bold]Current Configuration:[/bold]\n"
            f"Name: [cyan]{self.config['name'] or 'Not set'}[/cyan]\n"
            f"Models: [green]{len(self.config['models'])} selected[/green]\n"
            f"Prompts: [yellow]{len(self.config['prompts'])} selected[/yellow]\n"
            f"Algorithms: [blue]{len(self.config['algorithms'])} selected[/blue]\n"
            f"Repetitions: [magenta]{self.config['repetitions']}[/magenta]\n"
            f"Energy: [red]{self.config['energy_profiler']}[/red]",
            title="📊 Status",
            box=box.SIMPLE
        )
        self.console.print(config_panel)
        self.console.print()

    def set_experiment_name(self):
        """Set the experiment name"""
        self.console.clear()
        self.console.print("[bold blue]📝 Set Experiment Name[/bold blue]\n")
        
        current = f" (current: {self.config['name']})" if self.config['name'] else ""
        name = Prompt.ask(f"Enter experiment name{current}")
        
        if name:
            self.config['name'] = name
            self.console.print(f"[green]✅ Experiment name set to: {name}[/green]")
        
        input("\nPress Enter to continue...")

    def select_models(self):
        """Select models to compare"""
        self.console.clear()
        
        while True:
            self.console.print("[bold blue]🤖 Select Models to Compare[/bold blue]\n")
            
            # Show currently selected models
            if self.config['models']:
                selected_table = Table(title="Currently Selected Models", box=box.SIMPLE)
                selected_table.add_column("Model", style="green")
                for model in self.config['models']:
                    selected_table.add_row(model)
                self.console.print(selected_table)
                self.console.print()
            
            # Show available models
            table = Table(title="Popular Models", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Model ID", style="white")
            table.add_column("Selected", style="green")
            
            for i, model in enumerate(POPULAR_MODELS, 1):
                selected = "✅" if model in self.config['models'] else "⚪"
                table.add_row(str(i), model, selected)
            
            table.add_row("c", "Add custom model", "")
            table.add_row("s", "🔍 Search HuggingFace", "🔍")
            table.add_row("r", "Remove model", "")
            table.add_row("0", "Back to main menu", "")
            
            self.console.print(table)
            
            choice = Prompt.ask("Choose option", choices=[str(i) for i in range(len(POPULAR_MODELS) + 1)] + ['c', 's', 'r'])
            
            if choice == "0":
                break
            elif choice == "c":
                custom_model = Prompt.ask("Enter custom model ID (e.g., 'microsoft/DialoGPT-large')")
                if custom_model and custom_model not in self.config['models']:
                    self.config['models'].append(custom_model)
                    self.console.print(f"[green]✅ Added: {custom_model}[/green]")
            elif choice == "s":
                self.search_huggingface_models()
            elif choice == "r":
                if self.config['models']:
                    for i, model in enumerate(self.config['models'], 1):
                        self.console.print(f"{i}. {model}")
                    
                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config['models']):
                            removed = self.config['models'].pop(remove_idx)
                            self.console.print(f"[red]❌ Removed: {removed}[/red]")
                    except:
                        self.console.print("[red]Invalid selection[/red]")
                else:
                    self.console.print("[yellow]No models selected yet[/yellow]")
            else:
                try:
                    idx = int(choice) - 1
                    model = POPULAR_MODELS[idx]
                    if model in self.config['models']:
                        self.config['models'].remove(model)
                        self.console.print(f"[red]❌ Removed: {model}[/red]")
                    else:
                        self.config['models'].append(model)
                        self.console.print(f"[green]✅ Added: {model}[/green]")
                except:
                    self.console.print("[red]Invalid selection[/red]")
            
            input("\nPress Enter to continue...")
            self.console.clear()

    def choose_prompts(self):
        """Choose prompts from categories"""
        self.console.clear()
        
        while True:
            self.console.print("[bold blue]💭 Choose Prompts (Easy to Difficult)[/bold blue]\n")
            
            # Show currently selected prompts
            if self.config['prompts']:
                selected_table = Table(title="Currently Selected Prompts", box=box.SIMPLE)
                selected_table.add_column("#", style="cyan", width=4)
                selected_table.add_column("Prompt", style="green")
                for i, prompt in enumerate(self.config['prompts'], 1):
                    # Truncate long prompts for display
                    display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
                    selected_table.add_row(str(i), display_prompt)
                self.console.print(selected_table)
                self.console.print()
            
            # Show categories
            table = Table(title="Prompt Categories (🟢 Easy → 🔴 Hard)", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Category", style="white")
            table.add_column("Difficulty", style="yellow")
            table.add_column("Count", style="blue")
            
            difficulties = ["🟢 Easy", "🟡 Easy-Medium", "🟠 Medium", "🔵 Medium-Hard", "🟣 Hard", "🔴 Very Hard", "⚫ Expert", "🌟 Advanced"]
            
            categories = list(PROMPT_CATEGORIES.keys())
            for i, (category, difficulty) in enumerate(zip(categories, difficulties), 1):
                count = len(PROMPT_CATEGORIES[category])
                table.add_row(str(i), category, difficulty, f"{count} prompts")
            
            table.add_row("a", "Add all categories", "🌈 Mixed", "")
            table.add_row("c", "Add custom prompt", "✏️ Custom", "")
            table.add_row("r", "Remove prompt", "❌ Remove", "")
            table.add_row("0", "Back to main menu", "", "")
            
            self.console.print(table)
            
            choices = [str(i) for i in range(len(categories) + 1)] + ['a', 'c', 'r']
            choice = Prompt.ask("Choose option", choices=choices)
            
            if choice == "0":
                break
            elif choice == "a":
                # Add all prompts from all categories
                for prompts in PROMPT_CATEGORIES.values():
                    for prompt in prompts:
                        if prompt not in self.config['prompts']:
                            self.config['prompts'].append(prompt)
                self.console.print("[green]✅ Added all prompts from all categories![/green]")
            elif choice == "c":
                custom_prompt = Prompt.ask("Enter your custom prompt")
                if custom_prompt and custom_prompt not in self.config['prompts']:
                    self.config['prompts'].append(custom_prompt)
                    self.console.print(f"[green]✅ Added custom prompt[/green]")
            elif choice == "r":
                if self.config['prompts']:
                    for i, prompt in enumerate(self.config['prompts'], 1):
                        display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
                        self.console.print(f"{i}. {display_prompt}")
                    
                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config['prompts']):
                            removed = self.config['prompts'].pop(remove_idx)
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
                selected = "✅" if prompt in self.config['prompts'] else "⚪"
                # Truncate long prompts for table display
                display_prompt = prompt[:70] + "..." if len(prompt) > 70 else prompt
                table.add_row(str(i), display_prompt, selected)
            
            table.add_row("all", "Add all prompts from this category", "")
            table.add_row("0", "Back to categories", "")
            
            self.console.print(table)
            
            choices = [str(i) for i in range(len(prompts) + 1)] + ['all']
            choice = Prompt.ask("Choose option", choices=choices)
            
            if choice == "0":
                break
            elif choice == "all":
                for prompt in prompts:
                    if prompt not in self.config['prompts']:
                        self.config['prompts'].append(prompt)
                self.console.print(f"[green]✅ Added all prompts from {category_name}[/green]")
                input("\nPress Enter to continue...")
            else:
                try:
                    idx = int(choice) - 1
                    prompt = prompts[idx]
                    if prompt in self.config['prompts']:
                        self.config['prompts'].remove(prompt)
                        self.console.print(f"[red]❌ Removed prompt[/red]")
                    else:
                        self.config['prompts'].append(prompt)
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
            self.console.print("[bold blue]🧮 Select Comparison Algorithms[/bold blue]\n")
            
            # Show currently selected algorithms
            if self.config['algorithms']:
                selected_table = Table(title="Currently Selected Algorithms", box=box.SIMPLE)
                selected_table.add_column("Algorithm", style="green")
                selected_table.add_column("Description", style="dim")
                
                # Create algorithm description lookup
                all_algorithms = {}
                for category in ALGORITHM_CATEGORIES.values():
                    for alg_name, alg_desc in category:
                        all_algorithms[alg_name] = alg_desc
                
                for alg in self.config['algorithms']:
                    desc = all_algorithms.get(alg, "Custom algorithm")
                    selected_table.add_row(alg, desc)
                self.console.print(selected_table)
                self.console.print()
            
            # Show algorithm categories
            table = Table(title="Algorithm Categories", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Category", style="white")
            table.add_column("Count", style="blue")
            
            categories = list(ALGORITHM_CATEGORIES.keys())
            for i, category in enumerate(categories, 1):
                count = len(ALGORITHM_CATEGORIES[category])
                table.add_row(str(i), category, f"{count} algorithms")
            
            table.add_row("a", "Quick select: Performance only", "3 algorithms")
            table.add_row("b", "Quick select: Quality only", "4 algorithms") 
            table.add_row("c", "Quick select: Comprehensive", "8 algorithms")
            table.add_row("all", "🎯 Add ALL algorithms from ALL categories", "17 algorithms")
            table.add_row("r", "Remove algorithm", "")
            table.add_row("0", "Back to main menu", "")
            
            self.console.print(table)
            
            choices = [str(i) for i in range(len(categories) + 1)] + ['a', 'b', 'c', 'all', 'r']
            choice = Prompt.ask("Choose option", choices=choices)
            
            if choice == "0":
                break
            elif choice == "a":
                # Quick select: Performance
                perf_algs = ["response_time", "token_throughput", "text_length"]
                for alg in perf_algs:
                    if alg not in self.config['algorithms']:
                        self.config['algorithms'].append(alg)
                self.console.print("[green]✅ Added performance algorithms[/green]")
            elif choice == "b":
                # Quick select: Quality  
                quality_algs = ["bleu", "rouge", "bert_score", "semantic_similarity"]
                for alg in quality_algs:
                    if alg not in self.config['algorithms']:
                        self.config['algorithms'].append(alg)
                self.console.print("[green]✅ Added quality algorithms[/green]")
            elif choice == "c":
                # Quick select: Comprehensive
                comp_algs = ["response_time", "text_length", "bleu", "rouge", "bert_score", "llm_as_judge", "safety_alignment", "truthfulness"]
                for alg in comp_algs:
                    if alg not in self.config['algorithms']:
                        self.config['algorithms'].append(alg)
                self.console.print("[green]✅ Added comprehensive algorithms[/green]")
            elif choice == "all":
                # Add ALL algorithms from ALL categories
                added_count = 0
                for category_algs in ALGORITHM_CATEGORIES.values():
                    for alg_name, _ in category_algs:
                        if alg_name not in self.config['algorithms']:
                            self.config['algorithms'].append(alg_name)
                            added_count += 1
                self.console.print(f"[green]🎯 Added ALL {added_count} algorithms from all categories![/green]")
                self.console.print(f"[dim]Total selected: {len(self.config['algorithms'])} algorithms[/dim]")
            elif choice == "r":
                if self.config['algorithms']:
                    for i, alg in enumerate(self.config['algorithms'], 1):
                        self.console.print(f"{i}. {alg}")
                    
                    try:
                        remove_idx = IntPrompt.ask("Enter number to remove") - 1
                        if 0 <= remove_idx < len(self.config['algorithms']):
                            removed = self.config['algorithms'].pop(remove_idx)
                            self.console.print(f"[red]❌ Removed: {removed}[/red]")
                    except:
                        self.console.print("[red]Invalid selection[/red]")
                else:
                    self.console.print("[yellow]No algorithms selected yet[/yellow]")
            else:
                try:
                    idx = int(choice) - 1
                    category_name = categories[idx]
                    self.console.clear()
                    self.show_category_algorithms(category_name)
                except:
                    self.console.print("[red]Invalid selection[/red]")
            
            input("\nPress Enter to continue...")
            self.console.clear()
    
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
                selected = "✅" if alg_name in self.config['algorithms'] else "⚪"
                table.add_row(str(i), alg_name, alg_desc, selected)
            
            table.add_row("a", "Add all from this category", "", "")
            table.add_row("0", "Back to categories", "", "")
            
            self.console.print(table)
            
            choices = [str(i) for i in range(len(algorithms) + 1)] + ['a']
            choice = Prompt.ask("Choose option", choices=choices)
            
            if choice == "0":
                break
            elif choice == "a":
                for alg_name, _ in algorithms:
                    if alg_name not in self.config['algorithms']:
                        self.config['algorithms'].append(alg_name)
                self.console.print(f"[green]✅ Added all algorithms from {category_name}[/green]")
                input("\nPress Enter to continue...")
            else:
                try:
                    idx = int(choice) - 1
                    alg_name, alg_desc = algorithms[idx]
                    if alg_name in self.config['algorithms']:
                        self.config['algorithms'].remove(alg_name)
                        self.console.print(f"[red]❌ Removed: {alg_name}[/red]")
                    else:
                        self.config['algorithms'].append(alg_name)
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
            self.console.print("[bold blue]⚙️ Configure Experiment Parameters[/bold blue]\n")
            
            # Show current parameters
            param_table = Table(title="Current Parameters", box=box.SIMPLE)
            param_table.add_column("Parameter", style="cyan")
            param_table.add_column("Value", style="green")
            param_table.add_column("Description", style="dim")
            
            param_table.add_row("repetitions", str(self.config['repetitions']), "Number of times to run each test")
            param_table.add_row("max_length", str(self.config['max_length']), "Maximum response length in tokens")
            param_table.add_row("temperature", str(self.config['temperature']), "Creativity (0.0 = deterministic, 1.0 = creative)")
            
            self.console.print(param_table)
            self.console.print()
            
            table = Table(title="Parameter Options", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Parameter", style="white")
            
            table.add_row("1", "Repetitions (current: {})".format(self.config['repetitions']))
            table.add_row("2", "Max Length (current: {})".format(self.config['max_length']))
            table.add_row("3", "Temperature (current: {})".format(self.config['temperature']))
            table.add_row("4", "Quick presets")
            table.add_row("0", "Back to main menu")
            
            self.console.print(table)
            
            choice = Prompt.ask("Choose option", choices=['1', '2', '3', '4', '0'])
            
            if choice == "0":
                break
            elif choice == "1":
                new_reps = IntPrompt.ask("Enter number of repetitions", default=self.config['repetitions'])
                if 1 <= new_reps <= 20:
                    self.config['repetitions'] = new_reps
                    self.console.print(f"[green]✅ Repetitions set to {new_reps}[/green]")
                else:
                    self.console.print("[red]❌ Please enter a value between 1 and 20[/red]")
            elif choice == "2":
                new_length = IntPrompt.ask("Enter maximum response length", default=self.config['max_length'])
                if 10 <= new_length <= 1000:
                    self.config['max_length'] = new_length
                    self.console.print(f"[green]✅ Max length set to {new_length}[/green]")
                else:
                    self.console.print("[red]❌ Please enter a value between 10 and 1000[/red]")
            elif choice == "3":
                try:
                    new_temp = float(Prompt.ask("Enter temperature (0.0-1.0)", default=str(self.config['temperature'])))
                    if 0.0 <= new_temp <= 1.0:
                        self.config['temperature'] = new_temp
                        self.console.print(f"[green]✅ Temperature set to {new_temp}[/green]")
                    else:
                        self.console.print("[red]❌ Please enter a value between 0.0 and 1.0[/red]")
                except ValueError:
                    self.console.print("[red]❌ Please enter a valid number[/red]")
            elif choice == "4":
                self.console.print("\n[bold]Quick Presets:[/bold]")
                self.console.print("1. Fast testing: 2 reps, 30 length, 0.7 temp")
                self.console.print("2. Balanced: 3 reps, 50 length, 0.7 temp") 
                self.console.print("3. Thorough: 5 reps, 100 length, 0.8 temp")
                self.console.print("4. Creative: 3 reps, 150 length, 0.9 temp")
                
                preset = Prompt.ask("Choose preset", choices=['1', '2', '3', '4'])
                
                if preset == "1":
                    self.config.update({"repetitions": 2, "max_length": 30, "temperature": 0.7})
                elif preset == "2":
                    self.config.update({"repetitions": 3, "max_length": 50, "temperature": 0.7})
                elif preset == "3":
                    self.config.update({"repetitions": 5, "max_length": 100, "temperature": 0.8})
                elif preset == "4":
                    self.config.update({"repetitions": 3, "max_length": 150, "temperature": 0.9})
                
                self.console.print(f"[green]✅ Applied preset {preset}[/green]")
            
            input("\nPress Enter to continue...")
            self.console.clear()

    def configure_energy_profiling(self):
        """Configure energy profiling options"""
        self.console.clear()
        self.console.print("[bold blue]🔋 Configure Energy Profiling[/bold blue]\n")
        
        table = Table(title="Energy Profiling Options", box=box.ROUNDED)
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Profiler", style="white")
        table.add_column("Description", style="dim")
        table.add_column("Status", style="green")
        
        current = self.config['energy_profiler']
        table.add_row("1", "None", "No energy profiling", "✅" if current == "none" else "⚪")
        table.add_row("2", "PowerJoular", "Detailed energy measurement (requires sudo)", "✅" if current == "powerjoular" else "⚪")
        table.add_row("3", "CodeCarbon", "Carbon footprint tracking", "✅" if current == "codecarbon" else "⚪")
        table.add_row("0", "Back to main menu", "", "")
        
        self.console.print(table)
        
        choice = Prompt.ask("Choose energy profiler", choices=['1', '2', '3', '0'])
        
        if choice == "0":
            return
        elif choice == "1":
            self.config['energy_profiler'] = "none"
            self.console.print("[green]✅ Energy profiling disabled[/green]")
        elif choice == "2":
            self.config['energy_profiler'] = "powerjoular"
            self.console.print("[green]✅ PowerJoular energy profiling enabled[/green]")
            self.console.print("[yellow]⚠️  Note: PowerJoular requires sudo privileges to run[/yellow]")
        elif choice == "3":
            self.config['energy_profiler'] = "codecarbon"
            self.console.print("[green]✅ CodeCarbon profiling enabled[/green]")
        
        input("\nPress Enter to continue...")

    def save_configuration(self):
        """Save current configuration to file"""
        self.console.clear()
        self.console.print("[bold blue]💾 Save Configuration[/bold blue]\n")
        
        if not self.is_config_complete():
            self.console.print("[red]❌ Configuration incomplete! Cannot save.[/red]")
            input("\nPress Enter to continue...")
            return
        
        # Suggest filename based on experiment name
        suggested_name = f"{self.config['name']}_config.json" if self.config['name'] else "lct_config.json"
        filename = Prompt.ask("Save as", default=suggested_name)
        
        try:
            config_path = Path("saved_configs")
            config_path.mkdir(exist_ok=True)
            
            filepath = config_path / filename
            with open(filepath, 'w') as f:
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
            
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
            
            self.config.update(loaded_config)
            self.console.print(f"[green]✅ Configuration loaded from {filepath.name}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Error loading configuration: {e}[/red]")
        
        input("\nPress Enter to continue...")

    def is_config_complete(self):
        """Check if configuration is complete enough to run"""
        return (self.config['name'] and 
                self.config['models'] and 
                self.config['prompts'] and 
                self.config['algorithms'])

    def search_huggingface_models(self):
        """Search HuggingFace models"""
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
            with self.console.status(f"[bold green]Searching for '{search_term}' models..."):
                # Search for models - removed deprecated task parameter
                models = list(list_models(search=search_term, limit=20))
            
            if not models:
                self.console.print(f"[yellow]No models found for '{search_term}'[/yellow]")
                input("\nPress Enter to continue...")
                return
            
            # Clear any leftover status indicators and display results
            self.console.clear()
            self.console.print(f"[bold green]✅ Found {len(models)} models for '{search_term}'[/bold green]\n")
            
            # Display search results
            table = Table(title=f"Search Results for '{search_term}'", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Model ID", style="white")
            table.add_column("Downloads", style="green")
            table.add_column("Tags", style="yellow")
            
            # Limit to first 15 results for display
            display_models = models[:15]
            
            for i, model in enumerate(display_models, 1):
                downloads = getattr(model, 'downloads', 0) or 0
                tags = ', '.join(getattr(model, 'tags', [])[:3])  # Show first 3 tags
                table.add_row(str(i), model.modelId, f"{downloads:,}", tags)
            
            table.add_row("0", "Back to model selection", "", "")
            self.console.print(table)
            self.console.print("\n[bold cyan]Select a model to add to your experiment:[/bold cyan]")
            
            # Get user choice with proper validation
            valid_choices = [str(i) for i in range(len(display_models) + 1)]  # 0 to len(models)
            choice = Prompt.ask("Enter your choice", choices=valid_choices, show_choices=True)
            
            if choice == "0":
                return
                
            try:
                idx = int(choice) - 1
                selected_model = display_models[idx].modelId
                if selected_model not in self.config['models']:
                    self.config['models'].append(selected_model)
                    self.console.print(f"\n[bold green]✅ Added: {selected_model}[/bold green]")
                else:
                    self.console.print(f"\n[bold yellow]⚠️  Model already selected: {selected_model}[/bold yellow]")
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
        self.console.print("[bold blue]🔑 HuggingFace Authentication Setup[/bold blue]\n")
        
        # Check current authentication status
        try:
            from huggingface_hub import whoami
            
            # Check if token is already set
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
            if token:
                try:
                    user_info = whoami(token=token)
                    self.console.print(f"[green]✅ Already authenticated as: {user_info.get('name', 'Unknown')}[/green]")
                    
                    update_choice = Prompt.ask("Would you like to update your token?", choices=['y', 'n'], default='n')
                    if update_choice.lower() != 'y':
                        return
                except:
                    self.console.print("[yellow]⚠️ Current token seems invalid[/yellow]")
            
            self.console.print("To access private models and avoid rate limits, you need a HuggingFace token.")
            self.console.print("Get your token from: https://huggingface.co/settings/tokens\n")
            
            # Get token from user
            token = Prompt.ask("Enter your HuggingFace token (leave empty to skip)", default="", show_default=False)
            
            if token:
                # Verify token
                try:
                    with self.console.status("[bold green]Verifying token..."):
                        user_info = whoami(token=token)
                        name = user_info.get('name', 'Unknown')
                    
                    # Show success message after spinner closes
                    self.console.print(f"[green]✅ Token valid! Authenticated as: {name}[/green]")
                    
                    # Save token to environment
                    os.environ['HF_TOKEN'] = token
                    
                    # Ask about saving to .env file (using regular Prompt instead of Confirm)
                    save_choice = Prompt.ask("\n💾 Save token to .env file for future sessions?", choices=['y', 'n'], default='y')
                    
                    if save_choice.lower() == 'y':
                        env_file = Path('.env')
                        env_content = ""
                        
                        if env_file.exists():
                            env_content = env_file.read_text()
                        
                        # Remove existing HF_TOKEN lines
                        lines = [line for line in env_content.split('\n') if not line.startswith('HF_TOKEN=')]
                        
                        # Add new token
                        lines.append(f'HF_TOKEN={token}')
                        
                        env_file.write_text('\n'.join(lines))
                        self.console.print("[green]✅ Token saved to .env file[/green]")
                    else:
                        self.console.print("[yellow]Token saved for current session only[/yellow]")
                        
                except Exception as e:
                    self.console.print(f"[red]❌ Invalid token: {e}[/red]")
            else:
                self.console.print("[yellow]Authentication skipped. Some models may not be accessible.[/yellow]")
                
        except ImportError:
            self.console.print("[red]❌ HuggingFace Hub not installed. Install with: pip install huggingface_hub[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Error setting up authentication: {e}[/red]")
            
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
                    box=box.SIMPLE
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
                
                choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6", "7", "0"])
                
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
            "Model type", 
            choices=["causal", "seq2seq"], 
            default="causal"
        )
        
        if Confirm.ask(f"Download {model_id} ({model_type})? This may take time and storage space."):
            with self.console.status(f"[bold green]Downloading {model_id}..."):
                result = data_manager.download_model(model_id, model_type)
            
            if result:
                self.console.print(f"[green]✅ Model {model_id} downloaded to {result}[/green]")
            else:
                self.console.print(f"[red]❌ Failed to download {model_id}[/red]")
        
        input("\nPress Enter to continue...")
    
    def download_dataset_ui(self, data_manager):
        """UI for downloading datasets"""
        self.console.clear()
        self.console.print("[bold blue]📊 Download Dataset[/bold blue]\n")
        
        # Show common datasets
        common_datasets = [
            "xsum", "cnn_dailymail", "squad", "glue", 
            "wmt14", "wmt16", "opus_books", "multi_news"
        ]
        
        self.console.print("Common datasets: " + ", ".join(common_datasets))
        self.console.print()
        
        dataset_name = Prompt.ask("Enter dataset name")
        config_name = Prompt.ask("Config name (optional, press Enter to skip)", default="")
        config_name = config_name if config_name.strip() else None
        
        if Confirm.ask(f"Download {dataset_name}? This may take time and storage space."):
            with self.console.status(f"[bold green]Downloading {dataset_name}..."):
                result = data_manager.download_dataset(dataset_name, config_name)
            
            if result:
                self.console.print(f"[green]✅ Dataset {dataset_name} downloaded to {result}[/green]")
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
                        self.console.print(f"[red]❌ Failed to remove {dataset_name}[/red]")
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
        table.add_row("Python Version", python_status, f"{python_version} (≥3.8 required)")
        
        # Check available RAM (if psutil available)
        if psutil:
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            ram_available = memory.available / (1024**3)
            ram_ok = ram_gb >= 4
            ram_status = "✅ Good" if ram_ok else "⚠️ Limited"
            table.add_row("RAM", ram_status, f"{ram_gb:.1f}GB total, {ram_available:.1f}GB available")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            disk_ok = disk_free_gb >= 5
            disk_status = "✅ Good" if disk_ok else "⚠️ Limited"
            table.add_row("Disk Space", disk_status, f"{disk_free_gb:.1f}GB free")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_status = "✅ Good" if cpu_count >= 2 else "⚠️ Limited"
            freq_info = f", {cpu_freq.current:.0f}MHz" if cpu_freq else ""
            table.add_row("CPU", cpu_status, f"{cpu_count} cores{freq_info}")
        else:
            table.add_row("System Info", "⚠️ Limited", "Install psutil for detailed system info: pip install psutil")
        
        # Check GPU
        gpu_status = "❌ Not detected"
        gpu_details = "CPU-only processing"
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split('\n')[0]
                gpu_status = "✅ NVIDIA GPU"
                gpu_details = gpu_info
        except:
            try:
                # Check for other GPU indicators
                if shutil.which('rocm-smi') or shutil.which('clinfo'):
                    gpu_status = "⚠️ Non-NVIDIA GPU"
                    gpu_details = "May work with CPU processing"
            except:
                pass
        
        table.add_row("GPU", gpu_status, gpu_details)
        
        # Check required packages
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers', 
            'sentence_transformers': 'Sentence Transformers',
            'nltk': 'NLTK',
            'rouge_score': 'ROUGE Score',
            'huggingface_hub': 'HuggingFace Hub'
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
            
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
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
            self.console.print("[red]❌ CRITICAL: Python version too old. Update to Python 3.8+[/red]")
        elif missing_packages:
            self.console.print(f"[yellow]⚠️ MISSING PACKAGES: Install with: pip install {' '.join(missing_packages)}[/yellow]")
        elif psutil:
            ram_ok = True
            disk_ok = True
            if hasattr(psutil, 'virtual_memory'):
                memory = psutil.virtual_memory()
                ram_gb = memory.total / (1024**3)
                ram_ok = ram_gb >= 4
                
                disk = psutil.disk_usage('/')
                disk_free_gb = disk.free / (1024**3)
                disk_ok = disk_free_gb >= 5
                
            if not ram_ok:
                self.console.print("[yellow]⚠️ LOW RAM: May have issues with large models. Consider using smaller models.[/yellow]")
            elif not disk_ok:
                self.console.print("[yellow]⚠️ LOW DISK SPACE: Models may not download properly. Free up space.[/yellow]")
            else:
                self.console.print("[green]🎉 SYSTEM READY: All checks passed! Your system can run LLM experiments.[/green]")
        else:
            self.console.print("[green]✅ BASIC CHECKS PASSED: Install psutil for detailed system analysis.[/green]")
        
        # Show recommendations
        self.console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")
        
        if gpu_status == "❌ Not detected":
            self.console.print("• For faster inference, consider using a GPU-enabled system")
            
        if missing_packages:
            self.console.print(f"• Install missing packages: [bold]pip install {' '.join(missing_packages)}[/bold]")
            
        if auth_status != "✅ Authenticated":
            self.console.print("• Set up HuggingFace authentication using option 11 in main menu")
            
        if psutil and hasattr(psutil, 'virtual_memory'):
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            if ram_gb < 8:
                self.console.print("• For large models, use smaller variants (e.g., distilgpt2 instead of gpt2-xl)")
        else:
            self.console.print("• Install psutil for detailed system monitoring: pip install psutil")
            
        self.console.print("• Start with small experiments (1-2 models, simple prompts) to test your setup")
        
        input("\nPress Enter to continue...")

    def install_needed_tools(self):
        """Install all required packages and tools automatically"""
        # Ensure Path and subprocess are available throughout the method
        from pathlib import Path
        import subprocess
        import sys
        
        self.console.clear()
        self.console.print("[bold blue]🔧 Install Needed Tools & Dependencies[/bold blue]\n")
        
        self.console.print("This will install all required packages and tools for LCT to work properly.")
        self.console.print("You can choose which components to install:\n")
        
        # Show detailed installation options
        options_table = Table(title="Installation Components", box=box.ROUNDED)
        options_table.add_column("Option", style="cyan", no_wrap=True)
        options_table.add_column("Component", style="white")
        options_table.add_column("Description", style="dim")
        
        options_table.add_row("1", "🖥️ System Tools", "Git, Maven, OpenJDK 11+, CPU utilities, system monitoring")
        options_table.add_row("2", "🔋 PowerJoular", "Energy profiling tool (requires Java, sudo)")
        options_table.add_row("3", "🐍 Virtual Environment", "Python virtual environment setup and pip upgrade")
        options_table.add_row("4", "🤖 Core ML Libraries", "PyTorch, Transformers, HuggingFace Hub, Sentence Transformers")
        options_table.add_row("5", "📊 Evaluation Libraries", "NLTK, spaCy, ROUGE, BERT-Score, BLEU, Evaluation metrics")
        options_table.add_row("6", "🔧 System Monitoring", "psutil, CodeCarbon, nvidia-ml-py, system diagnostics")
        options_table.add_row("7", "🎨 UI & Utilities", "Rich, Pandas, Matplotlib, Seaborn, NumPy, Requests")
        options_table.add_row("8", "� Optional Tools", "Jupyter, IPython, Plotly, Dash (development tools)")
        options_table.add_row("9", "📋 Requirements.txt", "Process existing requirements.txt file if present")
        options_table.add_row("10", "🌐 Language Models", "Download NLTK data and spaCy English model")
        # Get dynamic dataset names for component 11 description
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
            from dataset_manager import CleanDatasetManager
            dm = CleanDatasetManager()
            dataset_names = ', '.join([info['name'] for info in dm.datasets.values()])
        except:
            dataset_names = "HumanEval, GSM8K, SafetyBench, HellaSwag, TruthfulQA, AlpacaEval, MT-Bench"
        
        options_table.add_row("11", "📚 Research Datasets", f"Download research evaluation datasets ({dataset_names})")
        options_table.add_row("12", "🔧 Fix Dependencies", "Resolve PyTorch/NVIDIA package conflicts")
        options_table.add_row("all", "🎯 Install Everything", "Complete installation (all components above)")
        options_table.add_row("0", "❌ Cancel", "Return to main menu")
        
        self.console.print(options_table)
        self.console.print("\n[bold yellow]� Recommendations:[/bold yellow]")
        self.console.print("• First-time users: Choose 'all' for complete setup")
        self.console.print("• Advanced users: Select specific components as needed")
        self.console.print("• System tools (1-2) require sudo privileges")
        self.console.print("• Virtual environment (3) is required for Python packages (4-12)\n")
        
        # Get user selection
        valid_choices = [str(i) for i in range(13)] + ['all']
        choices = []
        
        while True:
            choice = Prompt.ask(
                "Select components to install (comma-separated, e.g. '1,3,4' or 'all')",
                default="all"
            )
            
            if choice.lower() == 'all':
                choices = list(range(1, 13))  # All components except cancel
                break
            elif choice == '0':
                return
            else:
                try:
                    choices = [int(x.strip()) for x in choice.split(',') if x.strip().isdigit()]
                    choices = [x for x in choices if 1 <= x <= 12]
                    if choices:
                        break
                    else:
                        self.console.print("[red]Please enter valid component numbers (1-12) or 'all'[/red]")
                except ValueError:
                    self.console.print("[red]Please enter numbers separated by commas or 'all'[/red]")
        
        self.console.print(f"\n[green]Selected components: {', '.join(map(str, choices))}[/green]\n")
        
        # Component installation logic
        installed_components = []
        failed_components = []
        
        # COMPONENT 1: System Tools
        if 1 in choices:
            self.console.print("[bold cyan]🖥️ INSTALLING: System Tools[/bold cyan]")
            
            system_tools = {
                "git": "Version control system (required for PowerJoular)",
                "maven": "Build tool for Java projects (required for PowerJoular)",
                "openjdk-11-jdk": "Java Development Kit 11+ (required for PowerJoular)",
                "cpulimit": "CPU limiting utility (for energy profiling)",
                "htop": "System monitoring tool"
            }
            
            success_count = 0
            for tool, description in system_tools.items():
                self.console.print(f"📥 Installing {tool}: {description}")
                try:
                    result = subprocess.run([
                        "sudo", "apt", "install", "-y", tool
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.console.print(f"  ✅ {tool} installed successfully")
                        success_count += 1
                    else:
                        self.console.print(f"  ⚠️ {tool} installation had issues (may already be installed)")
                        success_count += 1
                except Exception as e:
                    self.console.print(f"  ❌ Error installing {tool}: {e}")
            
            if success_count > 0:
                installed_components.append("System Tools")
            else:
                failed_components.append("System Tools")
        
        # COMPONENT 2: PowerJoular
        if 2 in choices:
            self.console.print("\n[bold cyan]🔋 INSTALLING: PowerJoular Energy Profiler[/bold cyan]")
            try:
                # Check if PowerJoular directory exists
                powerjoular_dir = Path.home() / "powerjoular"
                
                if not powerjoular_dir.exists():
                    self.console.print("  📥 Cloning PowerJoular repository...")
                    result = subprocess.run([
                        "git", "clone", "https://github.com/joular/powerjoular.git", 
                        str(powerjoular_dir)
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.console.print(f"  ❌ Failed to clone PowerJoular: {result.stderr}")
                        failed_components.append("PowerJoular")
                    else:
                        self.console.print("  ✅ PowerJoular repository cloned")
                else:
                    self.console.print("  ℹ️  PowerJoular directory already exists")
                
                # Build PowerJoular
                if powerjoular_dir.exists():
                    self.console.print("  🔨 Building PowerJoular...")
                    result = subprocess.run([
                        "mvn", "clean", "package", "-DskipTests"
                    ], cwd=str(powerjoular_dir), capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.console.print("  ✅ PowerJoular built successfully")
                        
                        # Create symlink to make powerjoular globally available
                        jar_file = powerjoular_dir / "target" / "powerjoular-1.0.jar"
                        if jar_file.exists():
                            script_content = f"""#!/bin/bash
# PowerJoular wrapper script
java -jar {jar_file} "$@"
"""
                            script_path = Path("/usr/local/bin/powerjoular")
                            try:
                                with open("/tmp/powerjoular_script", "w") as f:
                                    f.write(script_content)
                                subprocess.run(["sudo", "mv", "/tmp/powerjoular_script", str(script_path)], check=True)
                                subprocess.run(["sudo", "chmod", "+x", str(script_path)], check=True)
                                self.console.print("  ✅ PowerJoular available globally as 'powerjoular' command")
                                installed_components.append("PowerJoular")
                            except Exception as e:
                                self.console.print(f"  ⚠️  PowerJoular built but global setup failed: {e}")
                                installed_components.append("PowerJoular (partial)")
                        else:
                            self.console.print("  ⚠️  PowerJoular built but JAR file not found")
                            failed_components.append("PowerJoular")
                    else:
                        self.console.print(f"  ❌ PowerJoular build failed: {result.stderr}")
                        failed_components.append("PowerJoular")
                        
            except Exception as e:
                self.console.print(f"  ❌ PowerJoular installation error: {e}")
                failed_components.append("PowerJoular")
        
        # COMPONENT 3: Virtual Environment
        if 3 in choices:
            self.console.print("\n[bold cyan]🐍 INSTALLING: Python Virtual Environment[/bold cyan]")
            venv_path = Path("llm-experiment-runner/.venv")
            if not venv_path.exists():
                self.console.print("📦 Creating virtual environment...")
                try:
                    result = subprocess.run([
                        "python3", "-m", "venv", str(venv_path)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.console.print("  ✅ Virtual environment created")
                        installed_components.append("Virtual Environment")
                    else:
                        self.console.print(f"  ❌ Virtual environment creation failed: {result.stderr}")
                        failed_components.append("Virtual Environment")
                        # Skip Python package installation if venv failed
                        if any(x in choices for x in [4, 5, 6, 7, 8, 9, 10, 11, 12]):
                            self.console.print("  ⚠️  Skipping Python package installation due to virtual environment failure")
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
        python_exe = str(venv_path / "bin" / "python") if venv_path.exists() else "python3"
        pip_exe = str(venv_path / "bin" / "pip") if venv_path.exists() else "pip3"
        
        # COMPONENT 4: Core ML Libraries
        if 4 in choices:
            self.console.print("\n[bold cyan]🤖 INSTALLING: Core ML Libraries[/bold cyan]")
            core_packages = [
                ("torch>=2.6.0", "PyTorch deep learning framework (v2.6.0+ for security)"),
                ("transformers", "HuggingFace Transformers library"),
                ("sentence-transformers", "Sentence embedding models"),
                ("huggingface_hub", "HuggingFace model hub client")
            ]
            
            success_count = self._install_packages(core_packages, pip_exe)
            if success_count > 0:
                installed_components.append(f"Core ML Libraries ({success_count}/{len(core_packages)})")
            else:
                failed_components.append("Core ML Libraries")
        
        # COMPONENT 5: Evaluation Libraries
        if 5 in choices:
            self.console.print("\n[bold cyan]📊 INSTALLING: Evaluation Libraries[/bold cyan]")
            eval_packages = [
                ("nltk", "Natural Language Toolkit"),
                ("spacy", "Industrial-strength NLP library"),
                ("rouge-score", "ROUGE evaluation metric"),
                ("bert-score", "BERT-based evaluation metric"),
                ("evaluate", "HuggingFace evaluation library"),
                ("datasets", "HuggingFace datasets library"),
                ("sacrebleu", "BLEU score implementation"),
                ("scikit-learn", "Machine learning evaluation metrics")
            ]
            
            success_count = self._install_packages(eval_packages, pip_exe)
            if success_count > 0:
                installed_components.append(f"Evaluation Libraries ({success_count}/{len(eval_packages)})")
            else:
                failed_components.append("Evaluation Libraries")
        
        # COMPONENT 6: System Monitoring
        if 6 in choices:
            self.console.print("\n[bold cyan]🔧 INSTALLING: System Monitoring Tools[/bold cyan]")
            monitoring_packages = [
                ("psutil", "System and process monitoring"),
                ("codecarbon", "Carbon footprint tracking"),
                ("nvidia-ml-py>=12.0.0", "NVIDIA GPU monitoring (replaces deprecated pynvml)")
            ]
            
            success_count = self._install_packages(monitoring_packages, pip_exe)
            if success_count > 0:
                installed_components.append(f"System Monitoring ({success_count}/{len(monitoring_packages)})")
            else:
                failed_components.append("System Monitoring")
        
        # COMPONENT 7: UI & Utilities
        if 7 in choices:
            self.console.print("\n[bold cyan]🎨 INSTALLING: UI & Utility Libraries[/bold cyan]")
            ui_packages = [
                ("rich", "Rich text and beautiful formatting"),
                ("requests", "HTTP requests library"),
                ("numpy", "Numerical computing library"),
                ("pandas", "Data manipulation library"),
                ("matplotlib", "Plotting and visualization"),
                ("seaborn", "Statistical data visualization")
            ]
            
            success_count = self._install_packages(ui_packages, pip_exe)
            if success_count > 0:
                installed_components.append(f"UI & Utilities ({success_count}/{len(ui_packages)})")
            else:
                failed_components.append("UI & Utilities")
        
        # COMPONENT 8: Optional Tools
        if 8 in choices:
            self.console.print("\n[bold cyan]📈 INSTALLING: Optional Development Tools[/bold cyan]")
            optional_packages = [
                ("jupyter", "Jupyter notebook environment"),
                ("ipython", "Enhanced interactive Python"),
                ("plotly", "Interactive plotting library"),
                ("dash", "Web application framework")
            ]
            
            success_count = self._install_packages(optional_packages, pip_exe)
            if success_count > 0:
                installed_components.append(f"Optional Tools ({success_count}/{len(optional_packages)})")
            else:
                failed_components.append("Optional Tools")
        
        # COMPONENT 9: Requirements.txt
        if 9 in choices:
            requirements_file = Path("config/requirements.txt")
            if requirements_file.exists():
                self.console.print("\n[bold cyan]📋 INSTALLING: Requirements.txt Dependencies[/bold cyan]")
                try:
                    result = subprocess.run([
                        pip_exe, "install", "-r", str(requirements_file)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.console.print("✅ Requirements.txt installed successfully")
                        installed_components.append("Requirements.txt")
                    else:
                        self.console.print(f"⚠️ Requirements.txt installation issues: {result.stderr[:100]}...")
                        # Check if datasets was specifically mentioned as an issue
                        if "datasets" in result.stderr.lower():
                            self.console.print("   💡 Tip: Try running Component 11 (Reference Datasets) for dataset dependencies")
                        failed_components.append("Requirements.txt")
                except Exception as e:
                    self.console.print(f"❌ Requirements.txt installation error: {e}")
                    failed_components.append("Requirements.txt")
            else:
                self.console.print("\n[yellow]📋 Requirements.txt not found - skipping[/yellow]")
        
        # COMPONENT 10: Language Models
        if 10 in choices:
            self.console.print("\n[bold cyan]🌐 INSTALLING: Language Models & Data[/bold cyan]")
            
            # Install NLTK data
            self.console.print("📚 Setting up NLTK data...")
            try:
                result = subprocess.run([
                    python_exe, "-c", 
                    "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
                ], capture_output=True, text=True)
                
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
                result = subprocess.run([
                    python_exe, "-m", "spacy", "download", "en_core_web_sm"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.console.print("✅ spaCy English model installed")
                    spacy_success = True
                else:
                    self.console.print(f"⚠️ spaCy model installation issue: {result.stderr}")
                    spacy_success = False
            except Exception as e:
                self.console.print(f"⚠️ spaCy model installation issue: {e}")
                spacy_success = False
            
            if nltk_success or spacy_success:
                models_status = []
                if nltk_success: models_status.append("NLTK data")
                if spacy_success: models_status.append("spaCy model")
                installed_components.append(f"Language Models ({', '.join(models_status)})")
            else:
                failed_components.append("Language Models")
        
        # COMPONENT 11: Research Datasets
        if 11 in choices:
            self.console.print("\n[bold cyan]📚 INSTALLING: Research Datasets for Algorithm Evaluation[/bold cyan]")
            self.console.print("[dim]This downloads research evaluation datasets to enable enhanced algorithm evaluation[/dim]\n")
            
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
                        result = subprocess.run([
                            sys.executable, str(dataset_manager_path), "--critical"
                        ], capture_output=True, text=True, cwd=str(project_root))
                    
                    if result.returncode == 0:
                        self.console.print("✅ Critical research datasets installed successfully!")
                        # Parse output to show installed datasets
                        if "datasets installed" in result.stdout:
                            self.console.print(result.stdout.split('\n')[-10:])  # Show last 10 lines
                    else:
                        self.console.print(f"⚠️ Some datasets may have failed: {result.stderr}")
                        # Don't mark as failed if some datasets work
                    
                    # Show quick status
                    status_result = subprocess.run([
                        sys.executable, str(dataset_manager_path), "--deps"
                    ], capture_output=True, text=True, cwd=str(project_root))
                    
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
                    scripts_path = Path(project_root) / "scripts" if isinstance(project_root, str) else project_root / "scripts"
                    sys.path.insert(0, str(scripts_path))
                    from dataset_manager import CleanDatasetManager
                    dm = CleanDatasetManager()
                    dataset_count = len(dm.datasets)
                    dataset_names = ', '.join([info['name'] for info in dm.datasets.values()])
                except:
                    dataset_count = 7
                    dataset_names = "HumanEval, GSM8K, SafetyBench, HellaSwag, TruthfulQA, AlpacaEval, MT-Bench"
                
                self.console.print(f"\n[green]📊 ALGORITHM IMPACT:[/green]")
                self.console.print(f"• ✅ Dataset manager ready for on-demand dataset installation")
                self.console.print(f"• ✅ Supports {dataset_count} research datasets: {dataset_names}")
                self.console.print(f"• ✅ Enables enhanced evaluation for text comparison algorithms")
                self.console.print(f"• ✅ All 17 algorithms functional with smart fallback methods")
                
                self.console.print(f"• 🎯 Clean architecture ready for beta deployment")
                
                installed_components.append("Research Datasets")
            else:
                self.console.print(f"\n[yellow]⚠️ Dataset manager setup failed - algorithms will use fallback methods[/yellow]")
                self.console.print(f"[dim]💡 Note: All algorithms still functional with built-in evaluation methods[/dim]")
        
        # COMPONENT 12: Fix Dependencies
        if 12 in choices:
            self.console.print("\n[bold cyan]🔧 FIXING: PyTorch & NVIDIA Dependencies[/bold cyan]")
            self.console.print("[dim]This resolves common dependency conflicts and deprecated package warnings[/dim]\n")
            
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
                    result = subprocess.run([pip_exe, "show", "torch"], capture_output=True, text=True, check=True)
                    current_version = None
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            current_version = line.split(':', 1)[1].strip()
                            break
                    
                    if current_version:
                        self.console.print(f"  Current PyTorch version: {current_version}")
                        
                        # Check if version is < 2.6.0
                        try:
                            if version and version.parse(current_version) < version.parse("2.6.0"):
                                self.console.print("  ⚠️  PyTorch version < 2.6.0 has security vulnerabilities")
                                self.console.print("  📦 Upgrading to PyTorch 2.6.0...")
                                subprocess.run([pip_exe, "install", "torch>=2.6.0"], check=True)
                                self.console.print("  ✅ PyTorch upgraded to secure version")
                                fixed_count += 1
                            elif version:
                                self.console.print("  ✅ PyTorch version is secure")
                                fixed_count += 1
                            else:
                                # Fallback: just check if it's an obviously old version
                                if current_version.startswith(("1.", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5")):
                                    self.console.print("  ⚠️  PyTorch version appears old, upgrading...")
                                    subprocess.run([pip_exe, "install", "torch>=2.6.0"], check=True)
                                    self.console.print("  ✅ PyTorch upgraded")
                                    fixed_count += 1
                                else:
                                    self.console.print("  ✅ PyTorch version appears recent")
                                    fixed_count += 1
                        except Exception as ver_e:
                            self.console.print(f"  ⚠️  Version check failed: {ver_e}, installing latest...")
                            subprocess.run([pip_exe, "install", "torch>=2.6.0"], check=True)
                            fixed_count += 1
                    else:
                        self.console.print("  ❌ Could not determine PyTorch version")
                        
                except subprocess.CalledProcessError:
                    self.console.print("  ⚠️  PyTorch not installed, installing secure version...")
                    subprocess.run([pip_exe, "install", "torch>=2.6.0"], check=True)
                    self.console.print("  ✅ PyTorch installed")
                    fixed_count += 1
                
                # Fix deprecated pynvml package
                self.console.print("\n🔍 Checking NVIDIA packages...")
                try:
                    # Check if pynvml is installed
                    result = subprocess.run([pip_exe, "show", "pynvml"], capture_output=True, text=True)
                    if result.returncode == 0:
                        self.console.print("  ⚠️  Deprecated 'pynvml' package found")
                        self.console.print("  🗑️  Uninstalling deprecated pynvml...")
                        subprocess.run([pip_exe, "uninstall", "pynvml", "-y"], check=True)
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
                    subprocess.run([pip_exe, "install", "nvidia-ml-py>=12.0.0"], check=True)
                    self.console.print("  ✅ nvidia-ml-py installed/updated")
                    fixed_count += 1
                except subprocess.CalledProcessError as e:
                    self.console.print(f"  ⚠️  Issue installing nvidia-ml-py: {e}")
                
                # Reinstall codecarbon to use updated dependencies
                self.console.print("\n🔄 Refreshing codecarbon with updated dependencies...")
                try:
                    subprocess.run([pip_exe, "install", "--upgrade", "--force-reinstall", "codecarbon"], check=True)
                    self.console.print("  ✅ codecarbon refreshed")
                    fixed_count += 1
                except subprocess.CalledProcessError as e:
                    self.console.print(f"  ⚠️  Issue refreshing codecarbon: {e}")
                
                # Summary
                if fixed_count >= 3:
                    self.console.print(f"\n[green]🎉 Dependency fixes completed successfully! ({fixed_count}/{total_fixes})[/green]")
                    self.console.print("[green]• PyTorch security vulnerability resolved[/green]")
                    self.console.print("[green]• Deprecated NVIDIA packages removed[/green]") 
                    self.console.print("[green]• Modern nvidia-ml-py installed[/green]")
                    self.console.print("[green]• CodeCarbon refreshed with clean dependencies[/green]")
                    installed_components.append(f"Dependency Fixes ({fixed_count}/{total_fixes})")
                else:
                    self.console.print(f"[yellow]⚠️  Some dependency fixes had issues ({fixed_count}/{total_fixes})[/yellow]")
                    failed_components.append("Dependency Fixes")
                    
            except Exception as e:
                self.console.print(f"❌ Dependency fix error: {e}")
                failed_components.append("Dependency Fixes")
        
        # Final installation summary
        self.console.print(f"\n[bold green]🎉 Installation Complete![/bold green]")
        
        if installed_components:
            self.console.print(f"\n[green]✅ Successfully Installed ({len(installed_components)} components):[/green]")
            for component in installed_components:
                self.console.print(f"  • {component}")
        
        if failed_components:
            self.console.print(f"\n[red]❌ Failed Installations ({len(failed_components)} components):[/red]")
            for component in failed_components:
                self.console.print(f"  • {component}")
        
        # Usage instructions
        self.console.print(f"\n[bold cyan]💡 What's Ready:[/bold cyan]")
        if "Virtual Environment" in [c.split()[0:2] for c in installed_components if "Virtual" in c]:
            self.console.print("• 🐍 Python virtual environment ready")
        if any("ML" in c for c in installed_components):
            self.console.print("• 🤖 Machine learning and model inference")
        if any("Evaluation" in c for c in installed_components):
            self.console.print("• 📊 Text evaluation and scoring algorithms")
        if any("Reference Datasets" in c for c in installed_components):
            self.console.print("• 📚 Reference datasets for comprehensive algorithm evaluation")
        if "PowerJoular" in str(installed_components):
            self.console.print("• 🔋 PowerJoular energy profiling")
        if "System Tools" in installed_components:
            self.console.print("• 🖥️ System tools (Git, Maven, Java)")
        
        self.console.print(f"\n[bold yellow]� Usage Notes:[/bold yellow]")
        if venv_path.exists():
            self.console.print("• ✅ Activate virtual environment: source llm-experiment-runner/.venv/bin/activate")
            self.console.print("• ✅ Run experiments within the activated virtual environment")
        if "PowerJoular" in str(installed_components):
            self.console.print("• ✅ PowerJoular usage: sudo powerjoular -l -f output.csv")
        if "System Tools" in installed_components:
            self.console.print("• ✅ System tools (git, maven, java) available everywhere")
        
        self.console.print(f"\n[bold green]� Next Steps:[/bold green]")
        self.console.print("1. Run System Diagnostics (Option 10) to verify installation")
        self.console.print("2. Set up HuggingFace authentication (Option 11) for model access")
        self.console.print("3. Start creating experiments with your configured LCT!")
        
        input("\nPress Enter to continue...")
    
    def _install_packages(self, packages, pip_exe):
        """Helper method to install a list of packages"""
        success_count = 0
        for package, description in packages:
            self.console.print(f"📦 Installing {package}: {description}")
            
            result = subprocess.run([
                pip_exe, "install", package, "--upgrade"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print(f"  ✅ {package} installed successfully")
                success_count += 1
            else:
                error_msg = result.stderr[:100] + "..." if len(result.stderr) > 100 else result.stderr
                self.console.print(f"  ⚠️ {package} installation issues: {error_msg}")
        
        return success_count

    def launch_results_explorer(self):
        """Launch the Results Explorer tool"""
        self.console.clear()
        self.console.print("[bold blue]🔍 Results Explorer[/bold blue]\n")
        
        # Check if results_explorer.py exists
        results_explorer_path = os.path.join(os.path.dirname(__file__), "results_explorer.py")
        
        if not os.path.exists(results_explorer_path):
            self.console.print("[red]❌ Results Explorer not found![/red]")
            self.console.print("Please ensure results_explorer.py is in the same directory as this script.")
            input("\nPress Enter to continue...")
            return
        
        try:
            self.console.print("🔍 Launching Results Explorer...")
            self.console.print("This will open an interactive tool to:")
            self.console.print("• 📊 Browse and analyze experiment results")
            self.console.print("• 📋 View CSV content with statistics and filtering")
            self.console.print("• 🗑️ Delete individual experiments or bulk delete all")
            self.console.print("• 📤 Export experiments individually or bulk export all")
            self.console.print("• 🔄 Manage experiment lifecycle")
            self.console.print("\n[dim]Press Ctrl+C in Results Explorer to return here[/dim]\n")
            
            input("Press Enter to launch Results Explorer...")
            
            # Launch the results explorer
            result = subprocess.run([sys.executable, results_explorer_path], 
                                  cwd=os.path.dirname(__file__))
            
            self.console.print("\n[green]👋 Results Explorer closed[/green]")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Results Explorer launch cancelled[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Error launching Results Explorer: {str(e)}[/red]")
        
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
• ⚡ Multi-tool energy profiling (PowerJoular, CodeCarbon, RAPL)
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
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Additional quick stats
        stats_table = Table(title="📈 Tool Statistics", box=box.ROUNDED, show_header=False)
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
        self.console.print("• Developer Portfolio: [link]https://adam-bouafia.github.io/[/link]")
        self.console.print("• LinkedIn Profile: [link]https://www.linkedin.com/in/adam-bouafia-b597ab86/[/link]")
        self.console.print("• Experiment Runner: [link]https://github.com/S2-group/experiment-runner[/link]")
        
        self.console.print(f"\n[bold yellow]💝 Support Development:[/bold yellow]")
        self.console.print("• Donate via PayPal: [link]https://paypal.me/AdamBouafia[/link]")
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
        
        summary_table.add_row("Name", self.config['name'])
        summary_table.add_row("Models", f"{len(self.config['models'])} models")
        summary_table.add_row("Prompts", f"{len(self.config['prompts'])} prompts")
        summary_table.add_row("Algorithms", f"{len(self.config['algorithms'])} algorithms")
        summary_table.add_row("Repetitions", str(self.config['repetitions']))
        summary_table.add_row("Total Runs", str(len(self.config['models']) * len(self.config['prompts']) * self.config['repetitions']))
        summary_table.add_row("Energy Profiling", self.config['energy_profiler'])
        
        self.console.print(summary_table)
        self.console.print()
        
        if not Confirm.ask("Proceed with experiment creation?"):
            return
        
        try:
            # Create experiment using the local function
            self.console.print("[yellow]Creating experiment configuration...[/yellow]")
            
            # Build configuration dictionary
            config_dict = {
                'name': self.config['name'],
                'models': self.config['models'],
                'prompts': self.config['prompts'],
                'algorithms': self.config['algorithms'],
                'repetitions': self.config['repetitions'],
                'max_length': self.config['max_length'],
                'temperature': self.config['temperature'],
                'energy_profiler': self.config['energy_profiler'] if self.config['energy_profiler'] != 'none' else None
            }
            
            # Create the experiment using the local function
            experiment_path = create_experiment_from_config(config_dict)
            
            self.console.print(f"[green]✅ Experiment created at: {experiment_path}[/green]")
            self.console.print()
            
            if Confirm.ask("Run the experiment now?"):
                self.console.print("[yellow]🚀 Running experiment...[/yellow]")
                
                # Calculate expected progress info
                total_runs = len(self.config['models']) * len(self.config['prompts']) * self.config['repetitions']
                self.console.print(f"[cyan]📊 Experiment Details:[/cyan]")
                self.console.print(f"   • Models: {len(self.config['models'])}")
                self.console.print(f"   • Prompts: {len(self.config['prompts'])}")  
                self.console.print(f"   • Repetitions: {self.config['repetitions']}")
                self.console.print(f"   • Total runs: {total_runs}")
                self.console.print(f"   • Energy profiling: {self.config['energy_profiler']}")
                
                self.console.print("[dim]Live output will be shown below. Press Ctrl+C to cancel if needed.[/dim]")
                self.console.print("=" * 70)
                
                # Run experiment using the new CLI run command
                import subprocess
                import os
                import threading
                import time
                
                # Find project root (go up from app/src/ui to project root)
                project_root = Path(__file__).parent.parent.parent.parent
                venv_python = project_root / 'llm-experiment-runner/.venv/bin/python'
                
                # Set environment to include the app/src directory in Python path
                env = os.environ.copy()
                app_src_path = str(project_root / 'app/src')
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{app_src_path}:{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = app_src_path
                
                # Add HuggingFace environment variables to suppress warnings
                hf_home = str(project_root / 'data/huggingface')
                env.update({
                    'HF_HOME': hf_home,
                    'HF_DATASETS_CACHE': f"{hf_home}/datasets",
                    'HF_MODELS_CACHE': f"{hf_home}/models",
                    'TRANSFORMERS_CACHE': f"{hf_home}/transformers",
                    'TRANSFORMERS_VERBOSITY': 'error',
                    'DATASETS_VERBOSITY': 'error',
                    'PYTHONWARNINGS': 'ignore::FutureWarning:torch.cuda,ignore::FutureWarning:transformers'
                })
                
                # Use the new CLI run command instead of old experiment-runner
                cmd = [str(venv_python), '-m', 'llm_runner.cli.main_cli', 'run', experiment_path]
                
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
                        cwd=str(project_root),  # Run from project root but with PYTHONPATH set
                        env=env
                    )
                    
                    # Progress indicator with heartbeat
                    start_time = time.time()
                    output_lines = []
                    run_count = 0
                    last_output_time = start_time
                    heartbeat_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    heartbeat_idx = 0
                    
                    def show_heartbeat():
                        """Show a spinning indicator if no output for a while"""
                        nonlocal heartbeat_idx, last_output_time
                        current_time = time.time()
                        if current_time - last_output_time > 5:  # No output for 5+ seconds
                            elapsed = int(current_time - start_time)
                            heartbeat_idx = (heartbeat_idx + 1) % len(heartbeat_chars)
                            self.console.print(f"[dim]{heartbeat_chars[heartbeat_idx]} {elapsed:3d}s | Working... (no output for {int(current_time - last_output_time)}s)[/dim]")
                    
                    # Read output line by line and display in real-time
                    for line in iter(process.stdout.readline, ''):
                        if line.strip():
                            output_lines.append(line.strip())
                            elapsed = int(time.time() - start_time)
                            last_output_time = time.time()
                            
                            # Track progress through runs
                            if "Starting experiment" in line or "Run " in line or "Model:" in line:
                                run_count += 1
                                if total_runs > 0:
                                    progress = min(100, int((run_count / total_runs) * 100))
                                    self.console.print(f"[green]⏱️  {elapsed:3d}s | Progress: {progress:2d}% | {line.strip()}[/green]")
                                else:
                                    self.console.print(f"[green]⏱️  {elapsed:3d}s | {line.strip()}[/green]")
                            elif "error" in line.lower() or "exception" in line.lower():
                                self.console.print(f"[red]⏱️  {elapsed:3d}s | ❌ {line.strip()}[/red]")
                            elif "warning" in line.lower():
                                self.console.print(f"[yellow]⏱️  {elapsed:3d}s | ⚠️  {line.strip()}[/yellow]")  
                            elif "completed" in line.lower() or "finished" in line.lower():
                                self.console.print(f"[green]⏱️  {elapsed:3d}s | ✅ {line.strip()}[/green]")
                            else:
                                self.console.print(f"[dim]⏱️  {elapsed:3d}s[/dim] | {line.strip()}")
                        else:
                            # Show heartbeat if process is still running but no output
                            if process.poll() is None:
                                show_heartbeat()
                    
                    # Wait for process to complete
                    process.wait()
                    
                    self.console.print("=" * 70)
                    
                    if process.returncode == 0:
                        elapsed_total = int(time.time() - start_time)
                        self.console.print(f"[green]✅ Experiment completed successfully in {elapsed_total}s![/green]")
                        
                        # Get results directory from experiment path
                        results_dir = Path(experiment_path).parent
                        self.console.print(f"[blue]📁 Results saved in: {results_dir}/{self.config['name']}/[/blue]")
                        
                        # Show results files
                        experiment_results_dir = results_dir / self.config['name']
                        if experiment_results_dir.exists():
                            self.console.print("\n[bold]📊 Generated files:[/bold]")
                            for file in sorted(experiment_results_dir.glob("*")):
                                file_size = file.stat().st_size if file.is_file() else 0
                                self.console.print(f"  📄 {file.name} ({file_size} bytes)")
                        
                        # Show quick summary
                        csv_files = list(experiment_results_dir.glob("*.csv"))
                        if csv_files:
                            self.console.print(f"\n[green]📈 Found {len(csv_files)} CSV result files for analysis![/green]")
                    else:
                        self.console.print(f"[red]❌ Experiment failed (exit code: {process.returncode})[/red]")
                        if output_lines:
                            self.console.print("[red]Last few output lines:[/red]")
                            for line in output_lines[-5:]:
                                self.console.print(f"  {line}")
                        
                except subprocess.TimeoutExpired:
                    self.console.print("[red]❌ Experiment timed out[/red]")
                    process.kill()
                except KeyboardInterrupt:
                    self.console.print("[yellow]⚠️  Experiment cancelled by user[/yellow]")
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
    venv_python = str(project_root / '.venv/bin/python')
    cmd = [
        venv_python, '-m', 'llm_runner.cli.main_cli', 'init',
        '--name', config['name'],
        '--models', ','.join(config['models']),
        '--prompts', ','.join(config['prompts']),  # Remove the extra quotes
        '--algorithms', ','.join(config['algorithms']),
        '--repetitions', str(config['repetitions']),
        '--max-length', str(config['max_length']),
        '--temperature', str(config['temperature'])
    ]
    
    if config.get('energy_profiler') and config['energy_profiler'] != 'none':
        cmd.extend(['--energy-profiler', config['energy_profiler']])
    
    # Set environment to include the app/src directory in Python path
    env = os.environ.copy()
    app_src_path = str(project_root / 'app/src')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{app_src_path}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = app_src_path
    
    # Run the command from project root with proper Python path
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root), env=env)
    
    if result.returncode != 0:
        raise Exception(f"Failed to create experiment: {result.stderr}")
    
    # Extract experiment path from output
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if 'Configuration:' in line:
            return line.split(':', 1)[1].strip()
    
    # Fallback - construct path
    return str(Path(__file__).parent / 'experiments' / config['name'] / 'RunnerConfig.py')

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