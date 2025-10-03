"""
Reference Data Loader for Research Datasets.
Provides unified access to research evaluation datasets.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ReferenceDataLoader:
    """
    Unified data loader for research evaluation datasets.
    Handles loading and caching of evaluation datasets.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the reference data loader."""
        if cache_dir is None:
            # Use project root/data as cache directory
            project_root = Path(__file__).parent.parent.parent.parent.parent
            cache_dir = project_root / "data"
        
        self.cache_dir = Path(cache_dir)
        self.datasets_dir = self.cache_dir / "research_datasets"
        self.models_dir = self.cache_dir / "models" 
        self.cache_subdir = self.cache_dir / "cache"
        
        # Create directories if they don't exist
        for dir_path in [self.datasets_dir, self.models_dir, self.cache_subdir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache environment variables to use our unified structure
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(self.datasets_dir)
        os.environ['HF_HUB_CACHE'] = str(self.models_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.models_dir)

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ReferenceDataLoader:
    """Enhanced loader with research-backed datasets for each algorithm."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.datasets = {}
        
        # Research-based dataset mappings for algorithm enhancement
        self.algorithm_datasets = {
            # Code Generation - HumanEval (Industry Standard)
            'code_generation': {
                'primary': 'openai/openai_humaneval',
                'alternative': 'google/mbpp',
                'description': 'Python programming problems with test cases'
            },
            
            # Mathematical Reasoning - GSM8K (Most Widely Adopted)
            'mathematical_reasoning': {
                'primary': 'openai/gsm8k',
                'alternative': 'competition_math',
                'description': 'Grade school math word problems requiring multi-step reasoning'
            },
            
            # Commonsense Reasoning - HellaSwag (95.6% Human Accuracy Benchmark)
            'commonsense_reasoning': {
                'primary': 'Rowan/hellaswag',
                'alternative': 'tau/commonsense_qa',
                'description': 'Physical situation understanding scenarios'
            },
            
            # Safety Alignment - SafetyBench (Comprehensive Safety Assessment)
            'safety_alignment': {
                'primary': 'thu-coai/SafetyBench',
                'alternative': 'PKU-Alignment/BeaverTails',
                'description': '11,435 questions across 7 safety categories'
            },
            
            # Truthfulness - TruthfulQA (Standard for Hallucination Detection)
            'truthfulness': {
                'primary': 'EleutherAI/truthful_qa_mc',
                'alternative': 'HiTZ/truthfulqa-multi-MT',
                'description': '817 questions testing resistance to false beliefs'
            },
            
            # LLM-as-Judge - MT-Bench (Expert-Level Human Preferences)
            'llm_as_judge': {
                'primary': 'lmsys/mt_bench_human_judgments',
                'alternative': 'openai/summarize_from_feedback',
                'description': '3.3K expert pairwise human preferences'
            },
            
            # Pairwise Comparison - AlpacaEval (0.98 Human Correlation)
            'pairwise_comparison': {
                'primary': 'tatsu-lab/alpaca_eval',
                'alternative': 'lmsys/chatbot_arena_conversations',
                'description': '805 instructions for pairwise model comparison'
            }
        }
    
    def load_enhanced_dataset(self, algorithm_name: str, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Load the research-recommended dataset for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm needing data
            max_samples: Maximum number of samples to load
            
        Returns:
            List of data points with prompt, reference, and metadata
        """
        if algorithm_name not in self.algorithm_datasets:
            logger.warning(f"No enhanced dataset available for {algorithm_name}")
            return []
        
        dataset_info = self.algorithm_datasets[algorithm_name]
        
        # Try primary dataset first
        data = self._load_specific_dataset(
            dataset_info['primary'], 
            algorithm_name, 
            max_samples
        )
        
        # Fallback to alternative if primary fails
        if not data and 'alternative' in dataset_info:
            logger.info(f"Primary dataset failed, trying alternative for {algorithm_name}")
            data = self._load_specific_dataset(
                dataset_info['alternative'], 
                algorithm_name, 
                max_samples
            )
        
        return data
    
    def _load_specific_dataset(self, dataset_name: str, algorithm_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load a specific dataset with algorithm-appropriate processing."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            if algorithm_name == 'code_generation':
                return self._load_humaneval(dataset_name, max_samples)
            elif algorithm_name == 'mathematical_reasoning':
                return self._load_gsm8k(dataset_name, max_samples)
            elif algorithm_name == 'commonsense_reasoning':
                return self._load_hellaswag(dataset_name, max_samples)
            elif algorithm_name == 'safety_alignment':
                return self._load_safety_bench(dataset_name, max_samples)
            elif algorithm_name == 'truthfulness':
                return self._load_truthfulqa(dataset_name, max_samples)
            elif algorithm_name == 'llm_as_judge':
                return self._load_mt_bench(dataset_name, max_samples)
            elif algorithm_name == 'pairwise_comparison':
                return self._load_alpaca_eval(dataset_name, max_samples)
            else:
                logger.warning(f"No loader implemented for {algorithm_name}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load {dataset_name} for {algorithm_name}: {e}")
            return []
    
    def _load_humaneval(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load HumanEval dataset for code generation evaluation."""
        dataset = load_dataset(dataset_name, split="test", cache_dir=self.cache_dir)
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # HumanEval format: function signature + docstring + test cases
            prompt = f"Complete this Python function:\n\n{item['prompt']}"
            
            # Reference includes the canonical solution
            reference = item['canonical_solution']
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'humaneval',
                'metadata': {
                    'task_id': item['task_id'],
                    'entry_point': item['entry_point'],
                    'test_cases': item['test']
                }
            })
        
        logger.info(f"Loaded {len(data_points)} HumanEval samples")
        return data_points
    
    def _load_gsm8k(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load GSM8K dataset for mathematical reasoning evaluation."""
        dataset = load_dataset(dataset_name, split="test", cache_dir=self.cache_dir)
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = f"Solve this math word problem step by step:\n\n{item['question']}"
            reference = item['answer']
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'gsm8k',
                'metadata': {
                    'requires_multi_step': True,
                    'grade_level': 'elementary'
                }
            })
        
        logger.info(f"Loaded {len(data_points)} GSM8K samples")
        return data_points
    
    def _load_hellaswag(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load HellaSwag dataset for commonsense reasoning evaluation."""
        dataset = load_dataset(dataset_name, split="validation", cache_dir=self.cache_dir)
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # HellaSwag format: context + 4 choices, pick most plausible
            context = item['ctx']
            choices = item['endings']
            correct_idx = int(item['label'])
            correct_ending = choices[correct_idx]
            
            prompt = f"Complete this scenario with the most plausible continuation:\n\n{context}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
            reference = f"The correct answer is {chr(65 + correct_idx)}) {correct_ending}"
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'hellaswag',
                'metadata': {
                    'context': context,
                    'choices': choices,
                    'correct_choice': correct_idx,
                    'activity_label': item.get('activity_label', 'unknown')
                }
            })
        
        logger.info(f"Loaded {len(data_points)} HellaSwag samples")
        return data_points
    
    def _load_safety_bench(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load SafetyBench dataset for safety alignment evaluation."""
        try:
            dataset = load_dataset(dataset_name, split="test", cache_dir=self.cache_dir)
        except:
            # Fallback if main split doesn't work
            dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            dataset = dataset['test'] if 'test' in dataset else list(dataset.values())[0]
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = f"Safety evaluation question:\n\n{item['question']}"
            
            # SafetyBench typically has multiple choice format
            if 'choices' in item:
                choices_text = "\n".join([f"{chr(65+j)}) {choice}" for j, choice in enumerate(item['choices'])])
                prompt += f"\n\nChoices:\n{choices_text}"
                
            reference = f"Correct answer: {item.get('answer', 'Unknown')}"
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'safety_bench',
                'metadata': {
                    'category': item.get('category', 'unknown'),
                    'language': item.get('language', 'en')
                }
            })
        
        logger.info(f"Loaded {len(data_points)} SafetyBench samples")
        return data_points
    
    def _load_truthfulqa(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load TruthfulQA dataset for truthfulness evaluation."""
        dataset = load_dataset(dataset_name, split="validation", cache_dir=self.cache_dir)
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = f"Answer this question truthfully:\n\n{item['question']}"
            
            # TruthfulQA includes best answer and common misconceptions
            reference = f"Best answer: {item.get('best_answer', 'Unknown')}"
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'truthfulqa',
                'metadata': {
                    'category': item.get('category', 'unknown'),
                    'incorrect_answers': item.get('incorrect_answers', []),
                    'correct_answers': item.get('correct_answers', [])
                }
            })
        
        logger.info(f"Loaded {len(data_points)} TruthfulQA samples")
        return data_points
    
    def _load_mt_bench(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load MT-Bench dataset for LLM-as-judge evaluation."""
        try:
            dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            # MT-Bench might have different split structure
            if isinstance(dataset, dict):
                dataset = list(dataset.values())[0]
        except Exception as e:
            logger.error(f"Failed to load MT-Bench: {e}")
            return []
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = f"Multi-turn conversation evaluation:\n\n{item.get('conversation', item.get('question', 'Unknown'))}"
            reference = f"Human judgment: {item.get('judgment', item.get('score', 'Unknown'))}"
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'mt_bench',
                'metadata': {
                    'turn': item.get('turn', 1),
                    'category': item.get('category', 'unknown')
                }
            })
        
        logger.info(f"Loaded {len(data_points)} MT-Bench samples")
        return data_points
    
    def _load_alpaca_eval(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load AlpacaEval dataset for pairwise comparison evaluation."""
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
        
        # AlpacaEval might have multiple splits
        if isinstance(dataset, dict):
            dataset = dataset.get('eval', dataset.get('test', list(dataset.values())[0]))
        
        data_points = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            instruction = item.get('instruction', item.get('prompt', 'Unknown'))
            prompt = f"Instruction for comparison:\n\n{instruction}"
            
            # AlpacaEval includes reference outputs for comparison
            reference = item.get('output', item.get('response', 'Unknown'))
            
            data_points.append({
                'prompt': prompt,
                'reference': reference,
                'source': 'alpaca_eval',
                'metadata': {
                    'generator': item.get('generator', 'unknown'),
                    'instruction_id': item.get('instruction_id', i)
                }
            })
        
        logger.info(f"Loaded {len(data_points)} AlpacaEval samples")
        return data_points
    
    def load_humaneval_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load HumanEval dataset for code generation evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            dataset = load_dataset('openai/openai_humaneval', split='test', cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'prompt': item.get('prompt', ''),
                    'entry_point': item.get('entry_point', ''),
                    'canonical_solution': item.get('canonical_solution', ''),
                    'test': item.get('test', ''),
                    'task_id': item.get('task_id', f'task_{i}')
                })
            
            logger.info(f"Loaded {len(data_points)} HumanEval samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load HumanEval data: {e}")
            return []
    
    def load_gsm8k_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load GSM8K dataset for mathematical reasoning evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            dataset = load_dataset('openai/gsm8k', 'main', split='test', cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'problem_id': f'gsm8k_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} GSM8K samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K data: {e}")
            return []
    
    def load_safetybench_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load SafetyBench dataset for safety alignment evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            # SafetyBench has multiple language splits
            dataset = load_dataset('thu-coai/SafetyBench', 'test', split='en', cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'question': item.get('question', item),
                    'category': item.get('category', 'general'),
                    'safety_id': f'safety_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} SafetyBench samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load SafetyBench data: {e}")
            return []
    
    def load_hellaswag_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load HellaSwag dataset for commonsense reasoning evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            dataset = load_dataset('Rowan/hellaswag', split='validation', cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'ctx': item.get('ctx', ''),
                    'endings': item.get('endings', []),
                    'label': item.get('label', 0),
                    'activity_label': item.get('activity_label', ''),
                    'ctx_a': item.get('ctx_a', ''),
                    'ctx_b': item.get('ctx_b', ''),
                    'problem_id': f'hellaswag_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} HellaSwag samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load HellaSwag data: {e}")
            return []
    
    def get_implementation_priority(self) -> Dict[str, List[str]]:
        """Return implementation priority based on research findings."""
        return {
            'phase_1_high_impact': [
                'code_generation',      # HumanEval - functional correctness
                'mathematical_reasoning', # GSM8K - standard math benchmark
                'safety_alignment'      # SafetyBench - comprehensive safety
            ],
            'phase_2_enhanced_eval': [
                'commonsense_reasoning', # HellaSwag - physical world understanding
                'truthfulness',         # TruthfulQA - fact-checking capabilities
                'pairwise_comparison'   # AlpacaEval - human preference correlation
            ],
            'phase_3_advanced': [
                'llm_as_judge'         # MT-Bench - multi-turn dialogue evaluation
            ]
        }
    
    def load_mtbench_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load MT-Bench dataset for LLM-as-judge evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            # Try to load MT-Bench human judgments
            dataset = load_dataset('lmsys/mt_bench_human_judgments', cache_dir=self.cache_dir)
            
            # Handle different dataset structures
            if isinstance(dataset, dict):
                dataset = list(dataset.values())[0]
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'question': item.get('question', item.get('prompt', '')),
                    'conversation': item.get('conversation', ''),
                    'judgment': item.get('judgment', item.get('score', '')),
                    'turn': item.get('turn', 1),
                    'category': item.get('category', 'general'),
                    'problem_id': f'mtbench_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} MT-Bench samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load MT-Bench data: {e}")
            return []
    
    def load_truthfulqa_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load TruthfulQA dataset for truthfulness evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            # Load TruthfulQA multiple choice format
            dataset = load_dataset('EleutherAI/truthful_qa_mc', split='validation', cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'question': item.get('question', ''),
                    'best_answer': item.get('best_answer', ''),
                    'correct_answers': item.get('correct_answers', []),
                    'incorrect_answers': item.get('incorrect_answers', []),
                    'category': item.get('category', 'general'),
                    'problem_id': f'truthfulqa_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} TruthfulQA samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA data: {e}")
            return []
    
    def load_alpacaeval_data(self, max_samples: int = 50) -> List[Dict[str, Any]]:
        """Load AlpacaEval dataset for pairwise comparison evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
        
        try:
            # Try to load AlpacaEval dataset
            dataset = load_dataset('tatsu-lab/alpaca_eval', cache_dir=self.cache_dir)
            
            # Handle different dataset structures
            if isinstance(dataset, dict):
                dataset = dataset.get('eval', dataset.get('test', list(dataset.values())[0]))
            
            data_points = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                    
                data_points.append({
                    'instruction': item.get('instruction', item.get('prompt', '')),
                    'output': item.get('output', item.get('response', '')),
                    'generator': item.get('generator', 'unknown'),
                    'instruction_id': item.get('instruction_id', f'alpaca_{i}'),
                    'problem_id': f'alpacaeval_{i}'
                })
            
            logger.info(f"Loaded {len(data_points)} AlpacaEval samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load AlpacaEval data: {e}")
            return []
    
    def validate_all_loaders(self) -> Dict[str, bool]:
        """
        Validate that all expected dataset loader methods are available.
        Returns a dictionary of method names and their availability status.
        """
        expected_loaders = [
            'load_humaneval_data',
            'load_gsm8k_data', 
            'load_hellaswag_data',
            'load_safetybench_data',
            'load_truthfulqa_data',
            'load_mtbench_data',
            'load_alpacaeval_data'
        ]
        
        status = {}
        for loader_name in expected_loaders:
            try:
                method = getattr(self, loader_name)
                # Test if it's callable
                if callable(method):
                    status[loader_name] = True
                    logger.debug(f"✓ {loader_name} is available")
                else:
                    status[loader_name] = False
                    logger.warning(f"✗ {loader_name} exists but is not callable")
            except AttributeError:
                status[loader_name] = False
                logger.warning(f"✗ {loader_name} is not available")
        
        return status
    
    def __getattr__(self, name: str):
        """
        Defensive method to handle any missing dataset loader methods gracefully.
        This prevents AttributeError and provides graceful fallback for future datasets.
        """
        if name.startswith('load_') and name.endswith('_data'):
            # Extract dataset name from method name
            dataset_name = name.replace('load_', '').replace('_data', '')
            
            def fallback_loader(max_samples: int = 50) -> List[Dict[str, Any]]:
                logger.warning(f"Dataset loader method '{name}' not implemented yet. "
                             f"Returning empty dataset for '{dataset_name}'.")
                return []
            
            # Cache the fallback method to avoid repeated warnings
            setattr(self, name, fallback_loader)
            return fallback_loader
        
        # For any other missing attributes, raise the normal AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get_dataset_info(self, algorithm_name: str) -> Dict[str, str]:
        """Get information about the dataset for an algorithm."""
        return self.algorithm_datasets.get(algorithm_name, {})