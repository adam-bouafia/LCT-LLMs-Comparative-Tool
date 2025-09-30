#!/usr/bin/env python3
"""
LCT Launcher - Simple entry point for Interactive LLM Comparison Tool
Just run this file to start the interactive menu!
"""

import sys
import os
from pathlib import Path

# Add project directories to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
app_src = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, app_src)
sys.path.insert(0, project_root)

# Now import the LLM modules
from llm_runner.configs.energy_profiled_config import EnergyProfiledConfig

try:
    from ui.interactive_lct import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing interactive_lct: {e}")
    print("Make sure you're in the correct directory and have all dependencies installed.")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Goodbye!")
    sys.exit(0)