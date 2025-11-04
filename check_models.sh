#!/bin/bash
# Quick script to check if models are gated before running experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "llm-experiment-runner/.venv" ]; then
    source llm-experiment-runner/.venv/bin/activate
fi

# Check if models.txt exists or use arguments
if [ -f "models_to_check.txt" ]; then
    echo "📋 Checking models from models_to_check.txt..."
    python app/src/tools/gated_model_checker.py --batch-file models_to_check.txt --alternatives
elif [ $# -gt 0 ]; then
    echo "📋 Checking provided models..."
    python app/src/tools/gated_model_checker.py --check-list "$@" --alternatives
else
    echo "Usage: $0 [model1] [model2] ..."
    echo "   or: Create models_to_check.txt with one model ID per line"
    echo ""
    echo "Example:"
    echo "  $0 gpt2 meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1"
    exit 1
fi
