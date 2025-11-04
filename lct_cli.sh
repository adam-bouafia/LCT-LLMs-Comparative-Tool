#!/bin/bash
# LLM Comparative Tool CLI Wrapper
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/app/src"

# Detect Python and virtual environment
if [ -d "llm-experiment-runner/.venv" ]; then
    PYTHON_CMD="llm-experiment-runner/.venv/bin/python"
elif [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# Run CLI with all arguments passed through
$PYTHON_CMD -m llm_runner.cli.main_cli "$@"
