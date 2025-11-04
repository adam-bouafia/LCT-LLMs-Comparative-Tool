#!/bin/bash
# LLM Comparison Tool (LCT) CLI Launcher Script
# This script properly sets up the environment and runs the LCT CLI

# Set the working directory to the experiment runner root
cd "$(dirname "$0")"

# Get the absolute path of the project directory
PROJECT_DIR="$(pwd)"

# Set HuggingFace environment variables to suppress warnings
export HF_HOME="${PROJECT_DIR}/data/huggingface"
export HF_DATASETS_CACHE="${PROJECT_DIR}/data/huggingface/datasets"
export HF_MODELS_CACHE="${PROJECT_DIR}/data/huggingface/models"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/data/huggingface/transformers"
export TRANSFORMERS_VERBOSITY=error
export DATASETS_VERBOSITY=error

# Set PYTHONPATH to include the app/src directory for module imports
export PYTHONPATH="${PROJECT_DIR}/app/src"

# Check if virtual environment exists
if [ ! -d "llm-experiment-runner/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the Install Tools (option 9) first to set up dependencies."
    exit 1
fi

# Run the LCT CLI using the virtual environment
llm-experiment-runner/.venv/bin/python -m llm_runner.cli.main_cli "$@"