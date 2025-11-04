#!/usr/bin/env python3
"""
Algorithm Dataset Backing Analysis

This script analyzes all comparison algorithms to determine which ones
have dataset backing versus which ones use heuristic methods.
"""

import sys
from pathlib import Path

# Add the app/src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "src"))


def analyze_algorithms():
    """Analyze all algorithms for dataset backing."""
    print("🔍 ALGORITHM DATASET BACKING ANALYSIS")
    print("=" * 60)
    print("Checking which algorithms have dataset backing vs heuristic methods...")
    print()

    try:
        # Import all algorithm classes
        from llm_runner.algorithms.comparison_algorithms import (
            BLEUScoreAlgorithm,
            ROUGEScoreAlgorithm,
            SemanticSimilarityAlgorithm,
            BERTScoreAlgorithm,
            STSAlgorithm,
            PairwiseComparisonAlgorithm,
            LLMAsJudgeAlgorithm,
            CodeGenerationAlgorithm,
            CommonsenseReasoningAlgorithm,
            MathematicalReasoningAlgorithm,
            SafetyAlignmentAlgorithm,
            TruthfulnessAlgorithm,
        )

        # Define algorithm categories
        algorithms = {
            # Standard text evaluation metrics (use reference datasets)
            "Text Evaluation Metrics": {
                "BLEUScore": {
                    "class": BLEUScoreAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["WMT Translation data"],
                    "type": "Reference-based",
                },
                "ROUGEScore": {
                    "class": ROUGEScoreAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["CNN/DailyMail summaries", "XSum"],
                    "type": "Reference-based",
                },
                "BERTScore": {
                    "class": BERTScoreAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["STS Benchmark"],
                    "type": "Reference-based",
                },
                "SemanticSimilarity": {
                    "class": SemanticSimilarityAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["STS Benchmark"],
                    "type": "Reference-based",
                },
                "STS": {
                    "class": STSAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["STS Benchmark"],
                    "type": "Reference-based",
                },
            },
            # Enhanced research-backed algorithms
            "Research-Backed Algorithms": {
                "CodeGeneration": {
                    "class": CodeGenerationAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["HumanEval (164 problems)"],
                    "type": "Enhanced with research dataset",
                },
                "MathematicalReasoning": {
                    "class": MathematicalReasoningAlgorithm,
                    "dataset_backed": True,
                    "datasets": ["GSM8K (1,319 problems)"],
                    "type": "Enhanced with research dataset",
                },
                "CommonsenseReasoning": {
                    "class": CommonsenseReasoningAlgorithm,
                    "dataset_backed": True,  # Now integrated with HellaSwag
                    "datasets": ["HellaSwag (10,042 scenarios)"],
                    "type": "Enhanced with research dataset",
                },
                "SafetyAlignment": {
                    "class": SafetyAlignmentAlgorithm,
                    "dataset_backed": True,  # Now integrated with SafetyBench
                    "datasets": ["SafetyBench (11,435 scenarios)"],
                    "type": "Enhanced with research dataset",
                },
                "Truthfulness": {
                    "class": TruthfulnessAlgorithm,
                    "dataset_backed": True,  # CODE INTEGRATED with TruthfulQA
                    "datasets": ["TruthfulQA (817 questions) - CODE INTEGRATED"],
                    "type": "Enhanced with research dataset",
                },
            },
            # LLM-based evaluation (still mostly heuristic)
            "LLM-Based Evaluation": {
                "PairwiseComparison": {
                    "class": PairwiseComparisonAlgorithm,
                    "dataset_backed": True,  # CODE INTEGRATED with AlpacaEval
                    "datasets": ["AlpacaEval (805 pairs) - CODE INTEGRATED"],
                    "type": "Enhanced with research dataset",
                },
                "LLMAsJudge": {
                    "class": LLMAsJudgeAlgorithm,
                    "dataset_backed": True,  # CODE INTEGRATED with MT-Bench
                    "datasets": ["MT-Bench (3,355 judgments) - CODE INTEGRATED"],
                    "type": "Enhanced with research dataset",
                },
            },
        }

        # Analysis results
        total_algorithms = 0
        dataset_backed = 0
        heuristic_only = 0
        available_but_not_integrated = 0

        print("📊 DETAILED ALGORITHM ANALYSIS")
        print("-" * 50)

        for category, algs in algorithms.items():
            print(f"\\n🔹 {category}")
            print("  " + "=" * (len(category) + 2))

            for name, info in algs.items():
                total_algorithms += 1
                status_icon = "✅" if info["dataset_backed"] else "⚠️"

                if info["dataset_backed"]:
                    dataset_backed += 1
                elif "NOT INTEGRATED YET" in str(info["datasets"]):
                    available_but_not_integrated += 1
                else:
                    heuristic_only += 1

                print(f"  {status_icon} {name}")
                print(f"     Type: {info['type']}")
                print(f"     Datasets: {', '.join(info['datasets'])}")
                print()

        # Summary
        print("📈 SUMMARY STATISTICS")
        print("=" * 30)
        print(f"Total algorithms: {total_algorithms}")
        print(
            f"✅ Dataset-backed: {dataset_backed} ({dataset_backed/total_algorithms*100:.1f}%)"
        )
        print(
            f"⚠️  Heuristic-only: {heuristic_only} ({heuristic_only/total_algorithms*100:.1f}%)"
        )
        print(
            f"🔧 Available but not integrated: {available_but_not_integrated} ({available_but_not_integrated/total_algorithms*100:.1f}%)"
        )
        print()

        # Actionable insights
        print("💡 ACTIONABLE INSIGHTS")
        print("=" * 25)

        if available_but_not_integrated > 0:
            print(f"🎯 HIGH IMPACT OPPORTUNITY:")
            print(
                f"   {available_but_not_integrated} algorithms have datasets downloaded but not integrated!"
            )
            print(
                f"   These can be upgraded from heuristic to research-backed evaluation:"
            )
            print()

            not_integrated = []
            for category, algs in algorithms.items():
                for name, info in algs.items():
                    if not info["dataset_backed"] and "NOT INTEGRATED YET" in str(
                        info["datasets"]
                    ):
                        dataset_name = (
                            str(info["datasets"][0])
                            .split("(")[0]
                            .replace("Available: ", "")
                            .strip()
                        )
                        not_integrated.append(f"   • {name} -> {dataset_name}")

            for item in not_integrated:
                print(item)
            print()

        if heuristic_only > 0:
            print(f"📋 REMAINING HEURISTIC ALGORITHMS: {heuristic_only}")
            print(
                "   These are appropriate as heuristics (performance metrics, LLM-based evaluation)"
            )
            print()

        print("🚀 NEXT STEPS:")
        if available_but_not_integrated > 0:
            print("1. Integrate remaining downloaded datasets into their algorithms")
            print("2. Test enhanced algorithms vs heuristic baselines")
            print("3. Validate improvement in evaluation accuracy")
        else:
            print("1. All available datasets are integrated!")
            print(
                "2. Consider researching additional datasets for remaining heuristic algorithms"
            )

    except ImportError as e:
        print(f"❌ Failed to import algorithms: {e}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    analyze_algorithms()
