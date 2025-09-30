"""
Reference Data Loader for Comparison Algorithms

Loads high-quality reference datasets for algorithms that require ground truth data.
Based on research showing optimal datasets for each algorithm type.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("WARNING: 'datasets' library not available. Reference data loading disabled.")

logger = logging.getLogger(__name__)

@dataclass
class ReferenceDataPoint:
    """Single reference data point with prompt, reference answer, and metadata."""
    prompt: str
    reference: str
    source: str
    metadata: Dict[str, Any] = None

class ReferenceDataLoader:
    """Loads and manages reference datasets for comparison algorithms."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the reference data loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.datasets = {}
        self._load_status = {}
        
    def load_all_datasets(self, max_samples_per_dataset: int = 100) -> Dict[str, bool]:
        """Load all recommended datasets.
        
        Args:
            max_samples_per_dataset: Maximum samples to load from each dataset
            
        Returns:
            Dictionary mapping dataset names to success status
        """
        dataset_loaders = {
            'sts_benchmark': self._load_sts_benchmark,
            'wmt_human_eval': self._load_wmt_human_eval,
            'cnn_dailymail': self._load_cnn_dailymail,
            'sts_multi': self._load_sts_multi
        }
        
        results = {}
        successful_loads = 0
        total_datasets = len(dataset_loaders)
        
        for name, loader_func in dataset_loaders.items():
            try:
                logger.info(f"Loading {name} dataset...")
                data = loader_func(max_samples_per_dataset)
                if data:
                    self.datasets[name] = data
                    results[name] = True
                    successful_loads += 1
                    logger.info(f"✓ Loaded {len(data)} samples from {name}")
                else:
                    results[name] = False
                    logger.warning(f"✗ Failed to load {name} - no data returned")
            except Exception as e:
                results[name] = False
                logger.warning(f"✗ Failed to load {name}: {e}")
                
        self._load_status = results
        
        # Log summary
        logger.info(f"Reference data loading complete: {successful_loads}/{total_datasets} datasets loaded")
        if successful_loads < total_datasets:
            failed = [name for name, success in results.items() if not success]
            logger.info(f"Failed datasets will use fallbacks: {failed}")
        
        return results
    
    def _load_sts_benchmark(self, max_samples: int) -> List[ReferenceDataPoint]:
        """Load STS Benchmark for semantic similarity evaluation."""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return []
            
        try:
            # Primary: sentence-transformers/stsb (validation split)
            dataset = load_dataset("sentence-transformers/stsb", split="validation", cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                # Create prompt asking to evaluate similarity
                prompt = f"How similar are these two sentences?\nSentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}"
                
                # Reference is the similarity score explanation
                similarity = item['score']  # 0-5 scale
                reference = f"These sentences have a similarity score of {similarity:.1f} out of 5.0"
                if similarity >= 4.0:
                    reference += " They are very similar in meaning."
                elif similarity >= 3.0:
                    reference += " They are moderately similar."
                elif similarity >= 2.0:
                    reference += " They are somewhat related but different."
                else:
                    reference += " They are quite different in meaning."
                
                data_points.append(ReferenceDataPoint(
                    prompt=prompt,
                    reference=reference,
                    source="sts_benchmark",
                    metadata={
                        "sentence1": item['sentence1'],
                        "sentence2": item['sentence2'],
                        "similarity_score": similarity
                    }
                ))
            
            logger.info(f"Loaded {len(data_points)} STS benchmark samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load STS benchmark: {e}")
            return []
    
    def _load_wmt_human_eval(self, max_samples: int) -> List[ReferenceDataPoint]:
        """Load WMT human evaluation data for BLEU/BERTScore evaluation."""
        if not DATASETS_AVAILABLE:
            return []
            
        try:
            # RicardoRei/wmt-da-human-evaluation
            dataset = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train", cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Use source text as prompt for translation
                if 'src' in item and 'mt' in item and 'ref' in item:
                    prompt = f"Translate this text to English: {item['src']}"
                    reference = item['ref']  # Reference translation
                    
                    data_points.append(ReferenceDataPoint(
                        prompt=prompt,
                        reference=reference,
                        source="wmt_human_eval",
                        metadata={
                            "mt_system": item.get('system', 'unknown'),
                            "language_pair": item.get('lp', 'unknown'),
                            "human_score": item.get('score', 0)
                        }
                    ))
            
            logger.info(f"Loaded {len(data_points)} WMT evaluation samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load WMT data: {e}")
            # Fallback: create synthetic translation-like tasks
            return self._create_translation_fallback(max_samples)

    
    def _load_cnn_dailymail(self, max_samples: int) -> List[ReferenceDataPoint]:
        """Load CNN/DailyMail for ROUGE evaluation (summarization)."""
        if not DATASETS_AVAILABLE:
            return []
            
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation", cache_dir=self.cache_dir, trust_remote_code=False)
            
            data_points = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                prompt = f"Summarize this article:\n\n{item['article']}"
                reference = item['highlights']
                
                data_points.append(ReferenceDataPoint(
                    prompt=prompt,
                    reference=reference,
                    source="cnn_dailymail",
                    metadata={
                        "article_length": len(item['article'].split()),
                        "highlights_length": len(item['highlights'].split())
                    }
                ))
            
            logger.info(f"Loaded {len(data_points)} CNN/DailyMail samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load CNN/DailyMail: {e}")
            return []
    
    def _load_sts_multi(self, max_samples: int) -> List[ReferenceDataPoint]:
        """Load multilingual STS as additional semantic similarity data."""
        if not DATASETS_AVAILABLE:
            return []
            
        try:
            dataset = load_dataset("PhilipMay/stsb_multi_mt", name="en", split="test", cache_dir=self.cache_dir)
            
            data_points = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                prompt = f"Rate the semantic similarity between:\n1: {item['sentence1']}\n2: {item['sentence2']}"
                score = item['similarity_score']
                reference = f"Similarity score: {score:.2f} (scale 0-5)"
                
                data_points.append(ReferenceDataPoint(
                    prompt=prompt,
                    reference=reference,
                    source="sts_multi",
                    metadata={"similarity_score": score}
                ))
            
            logger.info(f"Loaded {len(data_points)} multilingual STS samples")
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load multilingual STS: {e}")
            return []
    
    def _create_translation_fallback(self, max_samples: int) -> List[ReferenceDataPoint]:
        """Create synthetic translation-like tasks as fallback."""
        fallback_pairs = [
            ("Hello, how are you?", "Hello, how are you doing?"),
            ("The weather is nice today.", "Today's weather is pleasant."),
            ("I like to read books.", "Reading books is something I enjoy."),
            ("The cat is sleeping.", "A cat is taking a nap."),
            ("Programming is fun.", "Coding can be enjoyable."),
            ("The sun is shining.", "It's sunny outside."),
            ("I need some help.", "Could you assist me?"),
            ("This is interesting.", "That's quite fascinating."),
            ("Thank you very much.", "I really appreciate it."),
            ("See you tomorrow.", "Until tomorrow then.")
        ]
        
        data_points = []
        for i in range(min(max_samples, len(fallback_pairs) * 5)):  # Repeat if needed
            source, target = fallback_pairs[i % len(fallback_pairs)]
            
            prompt = f"Rephrase this sentence: {source}"
            reference = target
            
            data_points.append(ReferenceDataPoint(
                prompt=prompt,
                reference=reference,
                source="synthetic_fallback",
                metadata={"pair_id": i % len(fallback_pairs)}
            ))
        
        logger.info(f"Created {len(data_points)} synthetic reference pairs")
        return data_points
    
    def get_reference_data_for_algorithms(self, 
                                        algorithm_names: List[str], 
                                        num_samples: int = 50) -> Dict[str, List[str]]:
        """Get appropriate reference data for specific algorithms.
        
        Args:
            algorithm_names: List of algorithm names needing reference data
            num_samples: Number of samples to return per algorithm
            
        Returns:
            Dictionary mapping algorithm names to reference text lists
        """
        reference_mapping = {
            'bleu': ['wmt_human_eval', 'cnn_dailymail'],
            'rouge': ['cnn_dailymail'],
            'bert_score': ['sts_benchmark', 'wmt_human_eval'],
            'semantic_similarity': ['sts_benchmark', 'sts_multi'],
            'semantic_textual_similarity': ['sts_benchmark', 'sts_multi']
        }
        
        result = {}
        
        for algo_name in algorithm_names:
            if algo_name not in reference_mapping:
                continue
                
            references = []
            preferred_datasets = reference_mapping[algo_name]
            
            # Try preferred datasets in order
            for dataset_name in preferred_datasets:
                if dataset_name in self.datasets:
                    dataset_points = self.datasets[dataset_name][:num_samples]
                    references.extend([point.reference for point in dataset_points])
                    if len(references) >= num_samples:
                        break
            
            # If we still don't have enough, try any available dataset
            if len(references) < num_samples:
                for dataset_name, dataset_points in self.datasets.items():
                    if dataset_name not in preferred_datasets:
                        needed = num_samples - len(references)
                        references.extend([point.reference for point in dataset_points[:needed]])
                        if len(references) >= num_samples:
                            break
            
            # Trim to exact number needed
            result[algo_name] = references[:num_samples]
            logger.info(f"Prepared {len(result[algo_name])} reference texts for {algo_name}")
        
        return result
    
    def get_prompts_and_references(self, num_samples: int = 50) -> Tuple[List[str], List[str]]:
        """Get matched prompts and reference answers for comprehensive evaluation.
        
        Args:
            num_samples: Number of prompt-reference pairs to return
            
        Returns:
            Tuple of (prompts, references) lists
        """
        all_data_points = []
        for dataset_points in self.datasets.values():
            all_data_points.extend(dataset_points)
        
        if not all_data_points:
            logger.warning("No reference data available")
            return [], []
        
        # Randomly sample to get diverse data
        if len(all_data_points) > num_samples:
            sampled_points = random.sample(all_data_points, num_samples)
        else:
            sampled_points = all_data_points
        
        prompts = [point.prompt for point in sampled_points]
        references = [point.reference for point in sampled_points]
        
        logger.info(f"Prepared {len(prompts)} prompt-reference pairs from {len(self.datasets)} datasets")
        return prompts, references
    
    def get_load_status(self) -> Dict[str, bool]:
        """Get status of dataset loading attempts."""
        return self._load_status.copy()
    
    def is_available(self) -> bool:
        """Check if any reference data is available."""
        return len(self.datasets) > 0


# Global instance for easy access
_global_loader = None

def get_reference_loader(cache_dir: Optional[str] = None) -> ReferenceDataLoader:
    """Get global reference data loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ReferenceDataLoader(cache_dir=cache_dir)
    return _global_loader

def load_reference_data(max_samples_per_dataset: int = 100, 
                       cache_dir: Optional[str] = None) -> ReferenceDataLoader:
    """Convenience function to load all reference datasets.
    
    Args:
        max_samples_per_dataset: Maximum samples to load from each dataset
        cache_dir: Directory to cache datasets
        
    Returns:
        Loaded ReferenceDataLoader instance
    """
    loader = get_reference_loader(cache_dir)
    status = loader.load_all_datasets(max_samples_per_dataset)
    
    loaded_count = sum(status.values())
    total_count = len(status)
    
    logger.info(f"Reference data loading complete: {loaded_count}/{total_count} datasets loaded")
    
    if loaded_count == 0:
        logger.warning("No reference datasets could be loaded - algorithms requiring references will not work")
    
    return loader


if __name__ == "__main__":
    # Test the reference data loader
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Reference Data Loader...")
    loader = load_reference_data(max_samples_per_dataset=10)
    
    status = loader.get_load_status()
    print(f"\nDataset loading status:")
    for dataset, success in status.items():
        print(f"  {'✓' if success else '✗'} {dataset}")
    
    if loader.is_available():
        print(f"\nTotal reference data points loaded: {sum(len(data) for data in loader.datasets.values())}")
        
        # Test getting reference data for algorithms
        ref_data = loader.get_reference_data_for_algorithms(
            ['bleu', 'rouge', 'bert_score', 'semantic_similarity', 'semantic_textual_similarity'], 
            num_samples=5
        )
        
        print(f"\nReference data prepared for algorithms:")
        for algo, refs in ref_data.items():
            print(f"  {algo}: {len(refs)} references")
            if refs:
                print(f"    Example: {refs[0][:100]}...")
    else:
        print("\nNo reference data available - check dataset access")