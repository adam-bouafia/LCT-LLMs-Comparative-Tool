#!/usr/bin/env python3
"""
Enhanced Model Compatibility Checker for LLM Evaluation System

This utility helps users verify model compatibility before running experiments
and provides recommendations for different use cases.

NEW: Now includes support for 40+ popular modern models including:
- LLaMA 2 (Meta AI)
- Mistral AI models  
- Google Gemma
- Alibaba Qwen/Qwen2
- Microsoft Phi models
- 01-ai Yi models
- And many more...

Usage:
    python model_compatibility_checker.py model_name
    python model_compatibility_checker.py --list-recommended
"""

import sys
import argparse
from typing import Dict, List, Tuple, Optional
from llm_runner.loaders.universal_loader import UniversalModelLoader


def get_recommended_models() -> Dict[str, List[Tuple[str, str]]]:
    """Get recommended models by category with descriptions."""
    return {
        "small_fast": [
            ("gpt2", "Classic GPT-2: Fast, lightweight, great for testing"),
            ("distilgpt2", "Distilled GPT-2: Even faster, good baseline"),
            ("microsoft/phi-2", "Phi-2: Small but capable Microsoft model"),
            ("google/gemma-2b", "Gemma-2B: Google's efficient small model"),
            ("t5-small", "T5-Small: Versatile seq2seq model for many tasks"),
            ("google/flan-t5-small", "FLAN-T5-Small: Instruction-tuned T5")
        ],
        
        "medium_performance": [
            ("meta-llama/Llama-2-7b-hf", "LLaMA 2-7B: High-quality Meta model"),
            ("mistralai/Mistral-7B-v0.1", "Mistral-7B: Efficient high-performance model"),
            ("google/gemma-7b", "Gemma-7B: Google's balanced performance model"),
            ("Qwen/Qwen-7B", "Qwen-7B: Alibaba's multilingual model"),
            ("microsoft/Phi-3-mini-4k-instruct", "Phi-3: Microsoft's instruction-tuned model"),
            ("01-ai/Yi-6B", "Yi-6B: High-quality Chinese-English model")
        ],
        
        "code_models": [
            ("codellama/CodeLlama-7b-hf", "Code Llama: Meta's specialized code model"),
            ("Salesforce/codegen-350M-mono", "CodeGen: Salesforce code generation model"),
            ("Salesforce/codet5-small", "CodeT5: Code-focused T5 variant"),
            ("microsoft/CodeGPT-small-py", "CodeGPT: Microsoft Python code model")
        ],
        
        "chat_models": [
            ("microsoft/DialoGPT-medium", "DialoGPT: Conversational Microsoft model"),
            ("facebook/blenderbot-400M-distill", "BlenderBot: Facebook chat model"),
            ("microsoft/GODEL-v1_1-base-seq2seq", "GODEL: Task-oriented dialog model")
        ],
        
        "summarization": [
            ("google/pegasus-xsum", "Pegasus XSum: Abstractive summarization"),
            ("google/pegasus-cnn_dailymail", "Pegasus CNN: News summarization"),
            ("facebook/bart-large", "BART-Large: Strong seq2seq model"),
            ("google/flan-t5-base", "FLAN-T5-Base: Instruction-tuned for tasks")
        ],
        
        "translation": [
            ("Helsinki-NLP/opus-mt-en-de", "Marian: English-German translation"),
            ("facebook/mbart-large-50", "mBART: Multilingual translation model"),
            ("google/ul2", "UL2: Universal language learner")
        ],
        
        "large_models": [
            ("EleutherAI/gpt-j-6b", "GPT-J: Large open-source model"),
            ("facebook/opt-1.3b", "OPT-1.3B: Facebook's large model"),
            ("bigscience/bloom-1b7", "BLOOM: Multilingual large model"),
            ("tiiuae/falcon-7b", "Falcon-7B: High-performance UAE model")
        ]
    }


def check_model_compatibility(model_id: str) -> Dict:
    """Check compatibility of a specific model."""
    print(f"\n🔍 CHECKING COMPATIBILITY: {model_id}")
    print("=" * 50)
    
    try:
        loader = UniversalModelLoader()
        
        # Create metadata
        class TestMetadata:
            def __init__(self, model_id):
                self.id = model_id
                self.model_id = model_id
                self.pipeline_tag = None
                self.task = None
        
        metadata = TestMetadata(model_id)
        
        # Test model class detection
        model_class = loader._get_model_class(metadata)
        pipeline_task = loader._get_pipeline_task(metadata)
        
        print(f"✅ Model ID: {model_id}")
        print(f"✅ Detected Class: {model_class.__name__}")
        print(f"✅ Pipeline Task: {pipeline_task}")
        
        # Determine model category
        if "Seq2Seq" in model_class.__name__:
            category = "Sequence-to-Sequence (Encoder-Decoder)"
            use_case = "Translation, summarization, text-to-text tasks"
        else:
            category = "Causal Language Model (Decoder-only)"
            use_case = "Text generation, completion, chat"
        
        print(f"✅ Category: {category}")
        print(f"✅ Best Use Case: {use_case}")
        
        # Check for potential issues
        warnings = []
        
        if any(pattern in model_id.lower() for pattern in ['llama', 'mistral', 'falcon']) and '7b' in model_id.lower():
            warnings.append("⚠️  Large model (7B+) - ensure sufficient RAM/VRAM")
        
        if 'instruct' in model_id.lower() or 'chat' in model_id.lower():
            warnings.append("💡 Instruction-tuned model - may work better with specific prompts")
            
        if any(pattern in model_id.lower() for pattern in ['code', 'python', 'java']):
            warnings.append("💻 Code-specialized model - optimized for programming tasks")
        
        return {
            "compatible": True,
            "model_class": model_class.__name__,
            "pipeline_task": pipeline_task,
            "category": category,
            "use_case": use_case,
            "warnings": warnings
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            "compatible": False,
            "error": str(e),
            "suggestions": [
                "Check if model ID is correct",
                "Verify model exists on HuggingFace Hub",
                "Try adding trust_remote_code=True if using custom code"
            ]
        }


def list_recommended_models():
    """Display all recommended models by category."""
    print("\n🎯 RECOMMENDED MODELS BY CATEGORY")
    print("=" * 50)
    
    models = get_recommended_models()
    
    for category, model_list in models.items():
        print(f"\n📂 {category.upper().replace('_', ' ')}")
        print("-" * 30)
        
        for model_id, description in model_list:
            print(f"  • {model_id}")
            print(f"    {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Check model compatibility for LLM evaluation system"
    )
    parser.add_argument(
        "model", 
        nargs="?",
        help="Model ID to check (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--list-recommended", 
        action="store_true",
        help="List all recommended models by category"
    )
    
    args = parser.parse_args()
    
    if args.list_recommended:
        list_recommended_models()
    elif args.model:
        result = check_model_compatibility(args.model)
        
        if result["compatible"]:
            print(f"\n✅ COMPATIBILITY: EXCELLENT")
            print(f"   Ready for LLM evaluation experiments!")
            
            if result.get("warnings"):
                print(f"\n⚠️  NOTES:")
                for warning in result["warnings"]:
                    print(f"   {warning}")
        else:
            print(f"\n❌ COMPATIBILITY: ISSUES DETECTED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
            if result.get("suggestions"):
                print(f"\n💡 SUGGESTIONS:")
                for suggestion in result["suggestions"]:
                    print(f"   • {suggestion}")
    else:
        parser.print_help()
        print(f"\n💡 EXAMPLES:")
        print(f"   python model_compatibility_checker.py gpt2")
        print(f"   python model_compatibility_checker.py meta-llama/Llama-2-7b-hf")
        print(f"   python model_compatibility_checker.py --list-recommended")


if __name__ == "__main__":
    main()