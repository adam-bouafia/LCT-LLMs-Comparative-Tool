"""
Enhanced Dataset Configuration

This module provides configuration and mapping between algorithms and research-recommended datasets.
It serves as the central configuration point for the enhanced reference data loader.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration for a research-recommended dataset."""
    name: str
    huggingface_id: str
    split: str
    subset: str = None
    trust_remote_code: bool = False
    description: str = ""
    samples_count: int = 0
    research_source: str = ""
    implementation_priority: str = "Medium"
    evaluation_metric: str = ""
    
    def get_load_kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments for loading this dataset."""
        kwargs = {
            'path': self.huggingface_id,
            'split': self.split,
            'trust_remote_code': self.trust_remote_code
        }
        if self.subset:
            kwargs['name'] = self.subset
        return kwargs

class EnhancedDatasetRegistry:
    """Registry of all enhanced datasets with research backing."""
    
    def __init__(self):
        self._datasets = self._initialize_datasets()
        self._algorithm_mappings = self._initialize_algorithm_mappings()
    
    def _initialize_datasets(self) -> Dict[str, DatasetConfig]:
        """Initialize all research-recommended datasets."""
        return {
            'humaneval': DatasetConfig(
                name='HumanEval',
                huggingface_id='openai/openai_humaneval',
                split='test',
                trust_remote_code=True,
                description='Hand-written programming problems for functional correctness',
                samples_count=164,
                research_source='OpenAI (2021) - Evaluating Large Language Models Trained on Code',
                implementation_priority='Critical',
                evaluation_metric='pass@k (functional correctness)'
            ),
            'gsm8k': DatasetConfig(
                name='GSM8K',
                huggingface_id='openai/gsm8k',
                split='test',
                subset='main',
                description='Grade-school math word problems',
                samples_count=1319,
                research_source='Cobbe et al. (2021) - Training Verifiers to Solve Math Word Problems',
                implementation_priority='Critical',
                evaluation_metric='exact_match (answer accuracy)'
            ),
            'hellaswag': DatasetConfig(
                name='HellaSwag',
                huggingface_id='Rowan/hellaswag',
                split='validation',
                description='Commonsense reasoning about physical world',
                samples_count=10042,
                research_source='Zellers et al. (2019) - HellaSwag: Can a Machine Really Finish Your Sentence?',
                implementation_priority='High',
                evaluation_metric='accuracy (multiple choice)'
            ),
            'safetybench': DatasetConfig(
                name='SafetyBench',
                huggingface_id='thu-coai/SafetyBench',
                split='test',
                trust_remote_code=True,
                description='Comprehensive safety alignment evaluation',
                samples_count=11435,
                research_source='Zhang et al. (2023) - SafetyBench: Evaluating the Safety of Large Language Models',
                implementation_priority='Critical',
                evaluation_metric='safety_score (multi-dimensional)'
            ),
            'truthfulqa': DatasetConfig(
                name='TruthfulQA',
                huggingface_id='EleutherAI/truthful_qa_mc',
                split='validation',
                subset='multiple_choice',
                description='Questions designed to test truthfulness',
                samples_count=817,
                research_source='Lin et al. (2021) - TruthfulQA: Measuring How Models Mimic Human Falsehoods',
                implementation_priority='High',
                evaluation_metric='truthful_accuracy'
            ),
            'mt_bench': DatasetConfig(
                name='MT-Bench',
                huggingface_id='lmsys/mt_bench_human_judgments',
                split='train',
                description='Multi-turn dialogue human judgment dataset',
                samples_count=3355,
                research_source='Zheng et al. (2023) - Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
                implementation_priority='Medium',
                evaluation_metric='human_preference_correlation'
            ),
            'alpaca_eval': DatasetConfig(
                name='AlpacaEval',
                huggingface_id='tatsu-lab/alpaca_eval',
                split='eval',
                subset='alpaca_eval_gpt4_baseline',
                description='Instruction-following evaluation with GPT-4 as judge',
                samples_count=805,
                research_source='Li et al. (2023) - AlpacaEval: An Automatic Evaluator of Instruction-following Models',
                implementation_priority='High',
                evaluation_metric='win_rate (vs baseline)'
            )
        }
    
    def _initialize_algorithm_mappings(self) -> Dict[str, List[str]]:
        """Map algorithms to their recommended datasets."""
        return {
            # Code generation and programming
            'code_generation': ['humaneval'],
            'code_completion': ['humaneval'],
            'programming_assistance': ['humaneval'],
            
            # Mathematical and logical reasoning
            'mathematical_reasoning': ['gsm8k'],
            'arithmetic_reasoning': ['gsm8k'], 
            'word_problems': ['gsm8k'],
            
            # Commonsense and world knowledge
            'commonsense_reasoning': ['hellaswag'],
            'physical_reasoning': ['hellaswag'],
            'situation_understanding': ['hellaswag'],
            
            # Safety and alignment
            'safety_alignment': ['safetybench'],
            'bias_detection': ['safetybench'],
            'harmful_content_filtering': ['safetybench'],
            
            # Truthfulness and factuality
            'truthfulness': ['truthfulqa'],
            'factual_accuracy': ['truthfulqa'],
            'hallucination_detection': ['truthfulqa'],
            
            # LLM-as-judge applications
            'llm_as_judge': ['mt_bench'],
            'dialogue_evaluation': ['mt_bench'],
            'conversation_quality': ['mt_bench'],
            
            # Pairwise comparison and ranking
            'pairwise_comparison': ['alpaca_eval'],
            'instruction_following': ['alpaca_eval'],
            'model_ranking': ['alpaca_eval'],
            
            # Generic evaluation (fallback to multiple datasets)
            'general_evaluation': ['humaneval', 'gsm8k', 'hellaswag', 'truthfulqa'],
            'comprehensive_benchmark': ['humaneval', 'gsm8k', 'hellaswag', 'safetybench', 'truthfulqa']
        }
    
    def get_dataset(self, dataset_key: str) -> DatasetConfig:
        """Get dataset configuration by key."""
        return self._datasets.get(dataset_key)
    
    def get_datasets_for_algorithm(self, algorithm_name: str) -> List[DatasetConfig]:
        """Get recommended datasets for an algorithm."""
        dataset_keys = self._algorithm_mappings.get(algorithm_name, [])
        return [self._datasets[key] for key in dataset_keys if key in self._datasets]
    
    def get_all_datasets(self) -> Dict[str, DatasetConfig]:
        """Get all available datasets."""
        return self._datasets.copy()
    
    def get_algorithms_for_dataset(self, dataset_key: str) -> List[str]:
        """Get all algorithms that use a specific dataset."""
        algorithms = []
        for algorithm, datasets in self._algorithm_mappings.items():
            if dataset_key in datasets:
                algorithms.append(algorithm)
        return algorithms
    
    def get_implementation_phases(self) -> Dict[str, List[str]]:
        """Get datasets organized by implementation priority."""
        phases = {
            'Critical': [],
            'High': [],
            'Medium': []
        }
        
        for key, dataset in self._datasets.items():
            priority = dataset.implementation_priority
            if priority in phases:
                phases[priority].append(key)
        
        return phases
    
    def validate_algorithm_mapping(self, algorithm_name: str) -> Tuple[bool, List[str]]:
        """Validate if an algorithm has appropriate dataset mappings."""
        if algorithm_name not in self._algorithm_mappings:
            return False, [f"No dataset mapping found for algorithm: {algorithm_name}"]
        
        issues = []
        dataset_keys = self._algorithm_mappings[algorithm_name]
        
        for key in dataset_keys:
            if key not in self._datasets:
                issues.append(f"Dataset '{key}' referenced but not configured")
        
        return len(issues) == 0, issues
    
    def get_migration_plan(self) -> Dict[str, Any]:
        """Get a complete migration plan from current to enhanced datasets."""
        plan = {
            'overview': {
                'total_datasets': len(self._datasets),
                'total_algorithms': len(self._algorithm_mappings),
                'implementation_phases': len(self.get_implementation_phases())
            },
            'phases': {},
            'dataset_details': {},
            'algorithm_coverage': {}
        }
        
        # Phase breakdown
        phases = self.get_implementation_phases()
        for phase, dataset_keys in phases.items():
            plan['phases'][phase] = {
                'count': len(dataset_keys),
                'datasets': [self._datasets[key].name for key in dataset_keys],
                'total_samples': sum(self._datasets[key].samples_count for key in dataset_keys)
            }
        
        # Dataset details
        for key, dataset in self._datasets.items():
            plan['dataset_details'][key] = {
                'name': dataset.name,
                'source': dataset.research_source,
                'samples': dataset.samples_count,
                'priority': dataset.implementation_priority,
                'algorithms': self.get_algorithms_for_dataset(key)
            }
        
        # Algorithm coverage
        for algorithm, dataset_keys in self._algorithm_mappings.items():
            plan['algorithm_coverage'][algorithm] = {
                'dataset_count': len(dataset_keys),
                'datasets': [self._datasets[key].name for key in dataset_keys],
                'total_samples': sum(self._datasets[key].samples_count for key in dataset_keys)
            }
        
        return plan

# Global registry instance
ENHANCED_REGISTRY = EnhancedDatasetRegistry()

# Convenience functions
def get_dataset_config(dataset_key: str) -> DatasetConfig:
    """Get dataset configuration by key."""
    return ENHANCED_REGISTRY.get_dataset(dataset_key)

def get_algorithm_datasets(algorithm_name: str) -> List[DatasetConfig]:
    """Get recommended datasets for an algorithm."""
    return ENHANCED_REGISTRY.get_datasets_for_algorithm(algorithm_name)

def get_migration_plan() -> Dict[str, Any]:
    """Get the complete migration plan."""
    return ENHANCED_REGISTRY.get_migration_plan()

def validate_algorithm(algorithm_name: str) -> Tuple[bool, List[str]]:
    """Validate algorithm dataset mapping."""
    return ENHANCED_REGISTRY.validate_algorithm_mapping(algorithm_name)