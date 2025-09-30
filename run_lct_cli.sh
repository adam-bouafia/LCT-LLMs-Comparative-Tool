#!/bin/bash
# LLM Comparison Tool (LCT) CLI Launcher Script
# This script properly sets up the environment and runs the LCT CLI

# Set the working directory to the experiment runner root
cd "/home/neo/Documents/experiment runner"


# Set HuggingFace environment variables to suppress warnings
export HF_HOME="/home/neo/Documents/experiment runner/data/huggingface"
export HF_DATASETS_CACHE="/home/neo/Documents/experiment runner/data/huggingface/datasets"
export HF_MODELS_CACHE="/home/neo/Documents/experiment runner/data/huggingface/models"
export TRANSFORMERS_CACHE="/home/neo/Documents/experiment runner/data/huggingface/transformers"
export TRANSFORMERS_VERBOSITY=error
export DATASETS_VERBOSITY=error

# Set PYTHONPATH to include the app/src directory for module imports
export PYTHONPATH="/home/neo/Documents/experiment runner/app/src"

# Check if virtual environment exists
if [ ! -d "llm-experiment-runner/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the Install Tools (option 9) first to set up dependencies."
    exit 1
fi

# Run the LCT CLI using the virtual environment
llm-experiment-runner/.venv/bin/python -m llm_runner.cli.main_cli "$@"