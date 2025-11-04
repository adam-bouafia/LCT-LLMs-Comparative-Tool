#!/usr/bin/env python3
"""
Gated Model Checker for HuggingFace Models

This utility checks if models are gated (require authentication/approval)
before attempting to use them in experiments.

Usage:
    python gated_model_checker.py model_name
    python gated_model_checker.py --check-list model1 model2 model3
    python gated_model_checker.py --batch-file models.txt
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelAccessInfo:
    """Information about model access requirements."""
    model_id: str
    is_gated: bool
    gated_type: Optional[str] = None  # 'auto', 'manual', None
    requires_approval: bool = False
    has_access: bool = False
    error: Optional[str] = None


def check_model_access(model_id: str, token: Optional[str] = None) -> ModelAccessInfo:
    """
    Check if a model is gated and if we have access to it.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace API token (optional, checks env vars if not provided)
    
    Returns:
        ModelAccessInfo with access details
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        
        # Get token from parameter or environment
        if token is None:
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        
        api = HfApi(token=token)
        
        try:
            # Try to get model info
            model_info = api.model_info(model_id, token=token)
            
            # Check if model is gated
            is_gated = model_info.gated
            
            if not is_gated:
                return ModelAccessInfo(
                    model_id=model_id,
                    is_gated=False,
                    has_access=True
                )
            
            # Model is gated - check what type
            gated_type = getattr(model_info, 'gated', None)
            
            # Determine if it requires manual approval
            # 'auto' = automatic approval after accepting terms
            # 'manual' = requires human review
            requires_approval = gated_type == 'manual' if isinstance(gated_type, str) else True
            
            # Try to access a file to see if we have permission
            has_access = False
            try:
                # Try to download config.json (small file) to test access
                hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    token=token,
                    cache_dir=None,
                    local_files_only=False
                )
                has_access = True
            except GatedRepoError:
                has_access = False
            except Exception:
                # If file doesn't exist but no gated error, we likely have access
                has_access = True
            
            return ModelAccessInfo(
                model_id=model_id,
                is_gated=True,
                gated_type='manual' if requires_approval else 'auto',
                requires_approval=requires_approval,
                has_access=has_access
            )
            
        except GatedRepoError as e:
            # Definitely gated and we don't have access
            return ModelAccessInfo(
                model_id=model_id,
                is_gated=True,
                gated_type='manual',  # Assume manual if we hit this error
                requires_approval=True,
                has_access=False,
                error="Access denied - model requires approval"
            )
            
        except RepositoryNotFoundError:
            return ModelAccessInfo(
                model_id=model_id,
                is_gated=False,
                has_access=False,
                error="Model not found on HuggingFace Hub"
            )
            
    except ImportError:
        return ModelAccessInfo(
            model_id=model_id,
            is_gated=False,
            has_access=False,
            error="huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    except Exception as e:
        return ModelAccessInfo(
            model_id=model_id,
            is_gated=False,
            has_access=False,
            error=f"Error checking model: {str(e)}"
        )


def check_multiple_models(model_ids: List[str], token: Optional[str] = None) -> List[ModelAccessInfo]:
    """Check access for multiple models."""
    results = []
    for model_id in model_ids:
        print(f"Checking {model_id}...", end=" ", flush=True)
        info = check_model_access(model_id, token)
        
        if info.error:
            print(f"❌ {info.error}")
        elif info.is_gated and not info.has_access:
            print(f"🔒 GATED - Needs approval")
        elif info.is_gated and info.has_access:
            print(f"🔓 GATED - You have access")
        else:
            print(f"✅ Open access")
        
        results.append(info)
    
    return results


def get_open_alternatives(model_category: str) -> List[str]:
    """Get open (non-gated) alternatives for common model types."""
    alternatives = {
        "llama": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "openlm-research/open_llama_3b_v2",
            "openlm-research/open_llama_7b_v2",
        ],
        "mistral": [
            "HuggingFaceH4/zephyr-7b-beta",  # Based on Mistral
            "teknium/OpenHermes-2.5-Mistral-7B",
        ],
        "gemma": [
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        "code": [
            "Salesforce/codegen-350M-mono",
            "Salesforce/codegen-2B-mono",
            "microsoft/CodeGPT-small-py",
        ],
        "general": [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "bigscience/bloomz-560m",
            "bigscience/bloom-1b7",
        ]
    }
    return alternatives.get(model_category, alternatives["general"])


def print_access_report(results: List[ModelAccessInfo]):
    """Print a detailed report of model access status."""
    print("\n" + "="*70)
    print("MODEL ACCESS REPORT")
    print("="*70)
    
    open_models = [r for r in results if not r.is_gated and not r.error]
    gated_with_access = [r for r in results if r.is_gated and r.has_access]
    gated_no_access = [r for r in results if r.is_gated and not r.has_access]
    errors = [r for r in results if r.error]
    
    if open_models:
        print(f"\n✅ OPEN ACCESS ({len(open_models)} models):")
        for info in open_models:
            print(f"   • {info.model_id}")
    
    if gated_with_access:
        print(f"\n🔓 GATED - YOU HAVE ACCESS ({len(gated_with_access)} models):")
        for info in gated_with_access:
            print(f"   • {info.model_id} (Type: {info.gated_type})")
    
    if gated_no_access:
        print(f"\n🔒 GATED - ACCESS REQUIRED ({len(gated_no_access)} models):")
        for info in gated_no_access:
            approval_type = "Manual approval" if info.requires_approval else "Accept terms"
            print(f"   ❌ {info.model_id}")
            print(f"      → {approval_type} needed")
            print(f"      → Visit: https://huggingface.co/{info.model_id}")
    
    if errors:
        print(f"\n⚠️  ERRORS ({len(errors)} models):")
        for info in errors:
            print(f"   • {info.model_id}: {info.error}")
    
    # Summary
    print(f"\n" + "-"*70)
    print(f"SUMMARY:")
    print(f"   Total models checked: {len(results)}")
    print(f"   ✅ Ready to use: {len(open_models) + len(gated_with_access)}")
    print(f"   🔒 Need access: {len(gated_no_access)}")
    print(f"   ⚠️  Errors: {len(errors)}")
    
    if gated_no_access:
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Go to each model's HuggingFace page")
        print(f"   2. Sign in to your HuggingFace account")
        print(f"   3. Accept terms of use (and wait for approval if manual)")
        print(f"   4. Set HF_TOKEN environment variable with your token")
        print(f"\n   Get your token at: https://huggingface.co/settings/tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Check if HuggingFace models are gated and require approval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gated_model_checker.py meta-llama/Llama-2-7b-hf
  python gated_model_checker.py --check-list gpt2 meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1
  python gated_model_checker.py --batch-file models.txt
  
Note: Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable for authentication
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        help="Model ID to check (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    
    parser.add_argument(
        "--check-list",
        nargs="+",
        metavar="MODEL",
        help="Check multiple models"
    )
    
    parser.add_argument(
        "--batch-file",
        type=str,
        metavar="FILE",
        help="File containing list of model IDs (one per line)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--alternatives",
        action="store_true",
        help="Show open alternatives for gated models"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    
    if not token:
        print("⚠️  No HuggingFace token found. Set HF_TOKEN environment variable for full access check.")
        print("   Get your token at: https://huggingface.co/settings/tokens\n")
    
    # Determine which models to check
    models_to_check = []
    
    if args.batch_file:
        try:
            with open(args.batch_file, 'r') as f:
                models_to_check = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            print(f"❌ Error: File '{args.batch_file}' not found")
            sys.exit(1)
    elif args.check_list:
        models_to_check = args.check_list
    elif args.model:
        models_to_check = [args.model]
    else:
        parser.print_help()
        sys.exit(0)
    
    # Check models
    results = check_multiple_models(models_to_check, token)
    
    # Print report
    print_access_report(results)
    
    # Show alternatives if requested
    if args.alternatives:
        gated_models = [r for r in results if r.is_gated and not r.has_access]
        if gated_models:
            print(f"\n" + "="*70)
            print("OPEN ALTERNATIVES")
            print("="*70)
            print("\n📂 ALWAYS OPEN - NO APPROVAL NEEDED:")
            for model in get_open_alternatives("general"):
                print(f"   • {model}")
    
    # Exit with error code if any models need access
    gated_no_access = [r for r in results if r.is_gated and not r.has_access]
    if gated_no_access:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
