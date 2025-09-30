"""
Model Compatibility Checker Utility

This utility helps users check if their desired models are compatible
with the LLM Comparison Tool before running experiments.
"""

from llm_runner.loaders.universal_loader import UniversalModelLoader
from llm_runner.discovery.hf_model_discovery import HuggingFaceModelDiscovery
from typing import List, Dict, Any

def check_model_compatibility(model_ids: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Check compatibility of a list of model IDs.
    
    Args:
        model_ids: List of HuggingFace model IDs to check
        verbose: Print detailed information
    
    Returns:
        Dictionary with compatibility results
    """
    loader = UniversalModelLoader()
    discovery = HuggingFaceModelDiscovery()
    
    results = {
        'compatible': [],
        'incompatible': [],
        'warnings': [],
        'total': len(model_ids),
        'success_rate': 0
    }
    
    if verbose:
        print("🔍 MODEL COMPATIBILITY CHECK")
        print("=" * 40)
    
    for i, model_id in enumerate(model_ids, 1):
        try:
            if verbose:
                print(f"{i}. Checking {model_id}...")
            
            # Get metadata
            try:
                metadata = discovery.get_model_info(model_id)
                if not metadata:
                    # Create minimal metadata for testing
                    class SimpleMetadata:
                        def __init__(self, model_id):
                            self.id = model_id
                            self.model_id = model_id
                            self.pipeline_tag = None
                            self.task = None
                    metadata = SimpleMetadata(model_id)
            except Exception:
                # Fallback metadata
                class SimpleMetadata:
                    def __init__(self, model_id):
                        self.id = model_id
                        self.model_id = model_id
                        self.pipeline_tag = None
                        self.task = None
                metadata = SimpleMetadata(model_id)
            
            # Test detection
            detected_class = loader._get_model_class(metadata)
            detected_task = loader._get_pipeline_task(metadata)
            
            # Check if it's a supported pattern
            model_lower = model_id.lower()
            
            # Known problematic patterns
            problematic_patterns = ['bert', 'roberta', 'distilbert', 'electra']
            is_problematic = any(pattern in model_lower for pattern in problematic_patterns if not any(ok in model_lower for ok in ['code', 'gpt']))
            
            compatibility = {
                'model_id': model_id,
                'detected_class': detected_class.__name__,
                'detected_task': detected_task,
                'compatible': True,
                'warnings': []
            }
            
            if is_problematic:
                compatibility['warnings'].append("This model may be designed for classification/embedding tasks rather than text generation")
            
            results['compatible'].append(compatibility)
            
            if verbose:
                print(f"   ✅ Compatible - {detected_class.__name__} / {detected_task}")
                if compatibility['warnings']:
                    for warning in compatibility['warnings']:
                        print(f"   ⚠️  Warning: {warning}")
                        
        except Exception as e:
            incompatible = {
                'model_id': model_id,
                'error': str(e),
                'compatible': False
            }
            results['incompatible'].append(incompatible)
            
            if verbose:
                print(f"   ❌ Incompatible - {e}")
    
    results['success_rate'] = len(results['compatible']) / results['total'] * 100
    
    if verbose:
        print()
        print("📊 COMPATIBILITY SUMMARY:")
        print(f"   ✅ Compatible: {len(results['compatible'])}/{results['total']} ({results['success_rate']:.1f}%)")
        if results['incompatible']:
            print(f"   ❌ Incompatible: {len(results['incompatible'])}")
        
        print()
        if results['success_rate'] == 100:
            print("🎊 All models are compatible!")
        elif results['success_rate'] >= 90:
            print("👍 Excellent compatibility!")
        else:
            print("⚠️  Some models may need manual attention")
    
    return results

def get_recommended_models() -> Dict[str, List[str]]:
    """Get lists of recommended models by category."""
    return {
        'small_fast': [
            'distilgpt2',
            'microsoft/DialoGPT-small',
            'google/flan-t5-small',
            't5-small'
        ],
        'medium': [
            'gpt2',
            'microsoft/DialoGPT-medium', 
            'google/flan-t5-base',
            't5-base'
        ],
        'code_models': [
            'microsoft/CodeGPT-small-py',
            'Salesforce/codegen-350M-mono',
            'Salesforce/codet5-small'
        ],
        'chat_models': [
            'microsoft/DialoGPT-small',
            'facebook/blenderbot-400M-distill',
            'microsoft/GODEL-v1_1-base-seq2seq'
        ],
        'summarization': [
            'google/pegasus-xsum',
            'facebook/bart-base',
            'google/flan-t5-base'
        ]
    }

if __name__ == "__main__":
    # Example usage
    test_models = [
        'google/flan-t5-small',
        'gpt2', 
        'distilgpt2',
        'microsoft/DialoGPT-small',
        'facebook/bart-base'
    ]
    
    results = check_model_compatibility(test_models)
    
    print("\n🎯 RECOMMENDATIONS:")
    recommendations = get_recommended_models()
    for category, models in recommendations.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for model in models[:3]:  # Show top 3
            print(f"   • {model}")