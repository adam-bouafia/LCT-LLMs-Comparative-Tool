#!/bin/bash
# LLM Comparative Tool Starter
echo "🚀 Starting LLM Comparative Tool..."
cd "$(dirname "$0")"

# Get the absolute path of the project directory
PROJECT_DIR="$(pwd)"

# Set HuggingFace environment variables
export HF_HOME="${PROJECT_DIR}/data/huggingface"
export HF_DATASETS_CACHE="${PROJECT_DIR}/data/huggingface/datasets"
export HF_MODELS_CACHE="${PROJECT_DIR}/data/huggingface/models"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/data/huggingface/transformers"
export TRANSFORMERS_VERBOSITY=error
export DATASETS_VERBOSITY=error

# Set PYTHONPATH to include the app/src directory for module imports
export PYTHONPATH="$(pwd)/app/src"

# Activate virtual environment if it exists
if [ -d "llm-experiment-runner/.venv" ]; then
    echo "🔋 Using virtual environment..."
    PYTHON_CMD="llm-experiment-runner/.venv/bin/python"
elif [ -d ".venv" ]; then
    echo "🔋 Using legacy virtual environment..."
    PYTHON_CMD=".venv/bin/python"
else
    echo "⚠️  No virtual environment found, using system Python..."
    PYTHON_CMD="python3"
fi

# Run the main application
echo "🎯 Launching interactive tool..."
$PYTHON_CMD app/src/ui/interactive_lct.py "$@"
