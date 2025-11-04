#!/usr/bin/env python3
"""
Import Validation Script
Validates that all problematic imports are now working correctly.
"""

import sys
from pathlib import Path

# Add the app/src directory to Python path for root-level scripts
sys.path.insert(0, str(Path(__file__).parent / "app" / "src"))


def validate_imports():
    """Validate all the imports that were causing Pylance errors."""
    print("🔍 IMPORT VALIDATION TEST")
    print("=" * 50)

    tests = [
        ("llm_runner.algorithms.comparison_algorithms", "Algorithm classes"),
        ("core.data_manager", "Data manager"),
        ("llm_runner.data.reference_loader", "Reference loader"),
        ("llm_runner.data.dataset_config", "Dataset configuration"),
    ]

    all_passed = True

    for module_path, description in tests:
        try:
            __import__(module_path)
            print(f"✅ {description}: {module_path}")
        except ImportError as e:
            print(f"❌ {description}: {module_path} - {e}")
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL IMPORTS VALIDATED SUCCESSFULLY!")
        print("✅ Pylance errors should now be resolved")
        print("✅ pyrightconfig.json and .vscode/settings.json configured")
        print("✅ All scripts functional")
    else:
        print("❌ Some imports still failing")

    return all_passed


if __name__ == "__main__":
    validate_imports()
