#!/usr/bin/env python3
"""
LLM Comparative Tool - Main Entry Point
"""
import sys
import os
from pathlib import Path

# Add project directories to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
app_src = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, app_src)
sys.path.insert(0, project_root)

def main():
    """Main entry point"""
    from ui.interactive_lct import main as lct_main
    lct_main()

if __name__ == "__main__":
    main()
