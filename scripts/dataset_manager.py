#!/usr/bin/env python3
"""
Clean Dataset Manager for LLM Experiment Runner
Manages research evaluation datasets with verified working configurations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class CleanDatasetManager:
    """Clean dataset manager with verified working configurations."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize with clean directory structure."""
        self.project_root = Path(__file__).parent
        self.data_dir = data_dir or self.project_root / "data"
        self.datasets_dir = self.data_dir / "research_datasets"
        self.cache_dir = self.data_dir / "cache"

        # Create clean directory structure
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set cache environment
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)

        # Verified working datasets only
        self.datasets = {
            "humaneval": {
                "name": "HumanEval",
                "url": "openai_humaneval",
                "split": "test",
                "priority": "critical",
                "size_mb": 0.08,  # Very small
                "samples": 164,
                "algorithm": "code_generation",
                "description": "Code generation functional correctness",
            },
            "gsm8k": {
                "name": "GSM8K",
                "url": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "priority": "critical",
                "size_mb": 1,
                "samples": 1319,
                "algorithm": "mathematical_reasoning",
                "description": "Grade school math word problems",
            },
            "hellaswag": {
                "name": "HellaSwag",
                "url": "Rowan/hellaswag",
                "split": "validation",
                "priority": "high",
                "size_mb": 3,
                "samples": 10042,
                "algorithm": "commonsense_reasoning",
                "description": "Commonsense reasoning scenarios",
            },
            "truthfulqa": {
                "name": "TruthfulQA",
                "url": "truthful_qa",
                "config": "generation",
                "split": "validation",
                "priority": "high",
                "size_mb": 0.2,
                "samples": 817,
                "algorithm": "truthfulness",
                "description": "Truthfulness evaluation questions",
            },
            "safetybench": {
                "name": "SafetyBench",
                "url": "thu-coai/SafetyBench",
                "config": "test",
                "split": "en",  # Use 'en' split instead of 'test'
                "priority": "critical",
                "size_mb": 25,
                "samples": 11435,
                "algorithm": "safety_alignment",
                "description": "Comprehensive safety alignment evaluation (English)",
            },
            "mt_bench": {
                "name": "MT-Bench",
                "url": "lmsys/mt_bench_human_judgments",
                "split": "train",
                "priority": "high",
                "size_mb": 5,
                "samples": 3355,
                "algorithm": "multi_turn_dialogue",
                "description": "Multi-turn dialogue human judgment dataset",
            },
            "alpaca_eval": {
                "name": "AlpacaEval",
                "url": "tatsu-lab/alpaca_eval",
                "config": "alpaca_eval_gpt4_baseline",
                "split": "eval",
                "priority": "high",
                "size_mb": 3,
                "samples": 805,
                "algorithm": "instruction_following",
                "description": "Instruction-following evaluation with GPT-4 as judge",
            },
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import datasets
            import transformers
            import torch

            print(f"✅ datasets library: {datasets.__version__}")
            print(f"✅ transformers library: {transformers.__version__}")
            print(f"✅ torch library: {torch.__version__}")
            print("✅ All dependencies available!")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False

    def install_dataset(self, dataset_key: str) -> bool:
        """Install a single dataset with verified configuration."""
        if dataset_key not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False

        dataset_info = self.datasets[dataset_key]

        print(f"\n🚀 Installing {dataset_info['name']}...")
        print(f"📥 Downloading {dataset_info['name']} ({dataset_info['size_mb']}MB)...")

        try:
            from datasets import load_dataset

            # Prepare load arguments
            load_args = {
                "split": dataset_info["split"],
                "cache_dir": str(self.datasets_dir),
            }

            # Add config if specified
            if "config" in dataset_info:
                load_args["name"] = dataset_info["config"]

            # Load dataset
            dataset = load_dataset(dataset_info["url"], **load_args)

            print(
                f"✅ {dataset_info['name']} loaded successfully ({len(dataset)} samples)"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to install {dataset_info['name']}: {e}")
            return False

    def install_critical(self) -> Dict[str, bool]:
        """Install critical priority datasets."""
        critical_datasets = [
            k for k, v in self.datasets.items() if v["priority"] == "critical"
        ]
        results = {}

        for dataset_key in critical_datasets:
            results[dataset_key] = self.install_dataset(dataset_key)

        return results

    def install_high_priority(self) -> Dict[str, bool]:
        """Install high priority datasets."""
        high_datasets = [k for k, v in self.datasets.items() if v["priority"] == "high"]
        results = {}

        for dataset_key in high_datasets:
            results[dataset_key] = self.install_dataset(dataset_key)

        return results

    def install_all(self) -> Dict[str, bool]:
        """Install all available datasets."""
        results = {}

        for dataset_key in self.datasets.keys():
            results[dataset_key] = self.install_dataset(dataset_key)

        return results

    def check_installation_status(self) -> Dict[str, bool]:
        """Check which datasets are already installed."""
        status = {}

        for dataset_key, dataset_info in self.datasets.items():
            try:
                from datasets import load_dataset

                # Try to load without downloading
                load_args = {
                    "split": dataset_info["split"],
                    "cache_dir": str(self.datasets_dir),
                    "download_mode": "reuse_cache_if_exists",
                }

                if "config" in dataset_info:
                    load_args["name"] = dataset_info["config"]

                dataset = load_dataset(dataset_info["url"], **load_args)
                status[dataset_key] = True

            except Exception:
                status[dataset_key] = False

        return status

    def show_status(self):
        """Show current installation status."""
        status = self.check_installation_status()
        installed_count = sum(status.values())
        total_count = len(status)

        print(f"\n📊 DATASET STATUS")
        print("=" * 30)
        print(f"Progress: {installed_count}/{total_count} datasets installed")
        print()

        for dataset_key, is_installed in status.items():
            dataset_info = self.datasets[dataset_key]
            icon = "✅" if is_installed else "⭕"
            priority_text = f"({dataset_info['priority']} priority)"

            print(f"{icon} {dataset_info['name']} {priority_text}")
            print(f"   Algorithm: {dataset_info['algorithm']}")
            print(f"   Description: {dataset_info['description']}")
            print()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Clean Dataset Manager for LLM Experiment Runner"
    )
    parser.add_argument("--data-dir", type=Path, help="Data directory path")
    parser.add_argument(
        "--critical",
        action="store_true",
        help="Install critical priority datasets only",
    )
    parser.add_argument(
        "--high", action="store_true", help="Install high priority datasets"
    )
    parser.add_argument("--all", action="store_true", help="Install all datasets")
    parser.add_argument("--dataset", type=str, help="Install specific dataset")
    parser.add_argument(
        "--status", action="store_true", help="Show installation status"
    )
    parser.add_argument("--deps", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    manager = CleanDatasetManager(args.data_dir)

    # Check dependencies first
    if not manager.check_dependencies():
        sys.exit(1)

    if args.deps:
        sys.exit(0)

    if args.status:
        manager.show_status()
        sys.exit(0)

    # Installation commands
    results = {}

    if args.critical:
        print("\n📦 Installing critical priority datasets...")
        results = manager.install_critical()
    elif args.high:
        print("\n📦 Installing high priority datasets...")
        results = manager.install_high_priority()
    elif args.all:
        print("\n📦 Installing all datasets...")
        results = manager.install_all()
    elif args.dataset:
        print(f"\n📦 Installing {args.dataset}...")
        results[args.dataset] = manager.install_dataset(args.dataset)
    else:
        # Default: install critical datasets
        print("\n📦 Installing critical priority datasets...")
        results = manager.install_critical()

    # Show results
    if results:
        success_count = sum(results.values())
        total_count = len(results)

        if success_count == total_count:
            print(f"\n✅ All {success_count} datasets installed successfully!")
        elif success_count > 0:
            print(f"\n⚠️ {success_count}/{total_count} datasets installed successfully")
        else:
            print("\n❌ No datasets were installed successfully")
            sys.exit(1)

    # Always show final status
    manager.show_status()


if __name__ == "__main__":
    main()
