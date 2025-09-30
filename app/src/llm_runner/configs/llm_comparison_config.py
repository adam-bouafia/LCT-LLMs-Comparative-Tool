"""
LLM Comparison Configuration

This module provides the main configuration class for LLM comparison experiments,
extending the Experiment Runner's RunnerConfig.
"""

import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

# Use direct imports from experiment_runner package
from experiment_runner.EventManager.Models.RunnerEvents import RunnerEvents
from experiment_runner.EventManager.EventSubscriptionController import EventSubscriptionController
from experiment_runner.ConfigValidator.Config.Models.RunTableModel import RunTableModel
from experiment_runner.ConfigValidator.Config.Models.FactorModel import FactorModel
from experiment_runner.ConfigValidator.Config.Models.RunnerContext import RunnerContext
from experiment_runner.ConfigValidator.Config.Models.OperationType import OperationType
from experiment_runner.ExtendedTyping.Typing import SupportsStr
from experiment_runner.ProgressManager.Output.OutputProcedure import OutputProcedure as output

from ..discovery.hf_model_discovery import HuggingFaceModelDiscovery, ModelMetadata, ModelSearchCriteria
from ..loaders.universal_loader import UniversalModelLoader, ModelLoadConfig, LoadedModel
from ..algorithms.comparison_algorithms import ComparisonEngine, ModelResponse

logger = logging.getLogger(__name__)


class LLMComparisonConfig:
    """Configuration class for LLM comparison experiments."""
    
    ROOT_DIR = Path(dirname(realpath(__file__)))
    
    # ================================ USER SPECIFIC CONFIG ================================
    
    """The name of the experiment."""
    name: str = "llm_comparison_experiment"
    
    """The path in which Experiment Runner will create a folder with the name `self.name`."""
    results_output_path: Path = ROOT_DIR / 'experiments'
    
    """Experiment operation type."""
    operation_type: OperationType = OperationType.AUTO
    
    """The time Experiment Runner will wait after a run completes."""
    time_between_runs_in_ms: int = 2000  # Longer delay for LLM experiments
    
    # ================================ LLM SPECIFIC CONFIG ================================
    
    """List of model IDs to compare."""
    model_ids: List[str] = []
    
    """List of prompts to test with."""
    prompts: List[str] = ["Hello, how are you?"]
    
    """Reference texts for quality metrics (optional)."""
    reference_texts: Optional[List[str]] = None
    
    """Comparison algorithms to run."""
    comparison_algorithms: List[str] = ["response_time", "text_length"]
    
    """Model loading configuration."""
    model_load_config: ModelLoadConfig = ModelLoadConfig()
    
    """Maximum memory for loaded models (GB)."""
    max_memory_gb: float = 8.0
    
    """Number of repetitions per prompt-model combination."""
    repetitions: int = 3
    
    """Generation parameters for models."""
    generation_params: Dict[str, Any] = {
        "max_length": 150,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9
    }
    
    def __init__(self):
        """Initialize the LLM comparison configuration."""
        
        # Subscribe to experiment events
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        
        # Initialize components
        self.discovery = HuggingFaceModelDiscovery()
        self.loader = UniversalModelLoader(self.model_load_config, self.max_memory_gb, self.discovery)
        self.comparison_engine = ComparisonEngine()
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.current_prompt = ""
        self.current_model_id = ""
        self.model_responses: List[ModelResponse] = []
        
        self.run_table_model = None
        
        output.console_log("LLM Comparison Config loaded")
    
    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model for the experiment."""
        
        # Validate configuration
        self._validate_config()
        
        # Load model metadata
        self._load_model_metadata()
        
        # Create factors: models and prompts
        model_factor = FactorModel("model", self.model_ids)
        prompt_factor = FactorModel("prompt", [f"prompt_{i}" for i in range(len(self.prompts))])
        
        # Data columns for results
        data_columns = []
        available_algorithms = self.comparison_engine.get_available_algorithms()
        
        for algo_name in self.comparison_algorithms:
            if algo_name in available_algorithms:
                data_columns.extend([
                    f"{algo_name}_score",
                    f"{algo_name}_execution_time_ms"
                ])
            else:
                logger.warning(f"Algorithm {algo_name} not available, skipping")
        
        # Add performance metrics
        data_columns.extend([
            "response_time_ms",
            "tokens_generated", 
            "memory_usage_mb",
            "response_text"
        ])
        
        self.run_table_model = RunTableModel(
            factors=[model_factor, prompt_factor],
            exclude_variations=[],  # No exclusions by default
            repetitions=self.repetitions,
            data_columns=data_columns,
            shuffle=True  # Randomize run order
        )
        
        output.console_log(f"Created run table with {len(self.model_ids)} models and {len(self.prompts)} prompts")
        return self.run_table_model
    
    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment."""
        output.console_log("Starting LLM comparison experiment")
        output.console_log(f"Models to compare: {', '.join(self.model_ids)}")
        output.console_log(f"Algorithms: {', '.join(self.comparison_algorithms)}")
        
        # Pre-load all models to check for issues
        self._preload_models()
    
    def before_run(self) -> None:
        """Perform any activity required before starting a run."""
        pass
    
    def start_run(self, context: RunnerContext) -> None:
        """Start a specific run."""
        # Extract current run parameters
        self.current_model_id = context.run_variation["model"]
        prompt_index = int(context.run_variation["prompt"].replace("prompt_", ""))
        self.current_prompt = self.prompts[prompt_index]
        
        output.console_log(f"Starting run: {self.current_model_id} with prompt {prompt_index}")
    
    def start_measurement(self, context: RunnerContext) -> None:
        """Start measurement phase."""
        output.console_log(f"Starting measurement for {self.current_model_id}")
    
    def interact(self, context: RunnerContext) -> None:
        """Main interaction - generate response from the model."""
        try:
            # Ensure model is available for text generation
            if self.current_model_id not in self.loader.memory_manager.loaded_models:
                # Reload the model if it's not available
                metadata = self.model_metadata[self.current_model_id]
                loaded_model = self.loader.load_model(metadata)
                if not loaded_model:
                    raise RuntimeError(f"Failed to reload model {self.current_model_id}")
            
            # Generate response
            start_time = time.time()
            response_text = self.loader.generate_text(
                self.current_model_id,
                self.current_prompt,
                **self.generation_params
            )
            end_time = time.time()
            
            if response_text is None:
                raise RuntimeError(f"Failed to generate response from {self.current_model_id}")
            
            # Calculate response time
            response_time_ms = (end_time - start_time) * 1000
            
            # Estimate tokens generated (rough approximation)
            tokens_generated = len(response_text.split()) * 1.3  # Approximate tokens
            
            # Get memory usage
            memory_stats = self.loader.get_memory_stats()
            current_memory_mb = memory_stats.get("loaded_models", {}).get(self.current_model_id, {}).get("memory_mb", 0)
            
            # Store response for later analysis
            model_response = ModelResponse(
                model_id=self.current_model_id,
                prompt=self.current_prompt,
                response=response_text,
                response_time_ms=response_time_ms,
                tokens_generated=int(tokens_generated),
                memory_usage_mb=current_memory_mb
            )
            
            # Store in context for populate_run_data
            context.run_data = {
                "model_response": model_response,
                "response_time_ms": response_time_ms,
                "tokens_generated": int(tokens_generated),
                "memory_usage_mb": current_memory_mb,
                "response_text": response_text[:500] + "..." if len(response_text) > 500 else response_text  # Truncate for storage
            }
            
            output.console_log(f"Generated response in {response_time_ms:.1f}ms: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")
            
        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            context.run_data = {
                "model_response": None,
                "error": str(e),
                "response_time_ms": 0,
                "tokens_generated": 0,
                "memory_usage_mb": 0,
                "response_text": ""
            }
    
    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop measurement phase."""
        output.console_log(f"Stopping measurement for {self.current_model_id}")
    
    def stop_run(self, context: RunnerContext) -> None:
        """Stop the current run."""
        pass
    
    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        """Process measurement data and return results."""
        if not hasattr(context, 'run_data') or context.run_data is None:
            return None
        
        run_data = context.run_data
        result_data = {}
        
        # Add basic performance metrics
        result_data["response_time_ms"] = run_data.get("response_time_ms", 0)
        result_data["tokens_generated"] = run_data.get("tokens_generated", 0)
        result_data["memory_usage_mb"] = run_data.get("memory_usage_mb", 0)
        result_data["response_text"] = run_data.get("response_text", "")
        
        # Run comparison algorithms if we have a valid response
        model_response = run_data.get("model_response")
        if model_response:
            try:
                # Run algorithms that don't need reference data
                standalone_algorithms = ["response_time", "text_length", "token_throughput"]
                algorithms_to_run = [alg for alg in self.comparison_algorithms if alg in standalone_algorithms]
                
                if algorithms_to_run:
                    comparison_results = self.comparison_engine.run_comparison(
                        algorithms_to_run,
                        [model_response],
                        None
                    )
                    
                    # Add algorithm results to run data
                    for algo_name, results in comparison_results.items():
                        if results:
                            result = results[0]  # Only one model response
                            result_data[f"{algo_name}_score"] = result.score
                            result_data[f"{algo_name}_execution_time_ms"] = result.execution_time_ms
                
                # For algorithms that need reference data, run them separately if available
                if self.reference_texts:
                    reference_algorithms = ["bleu", "rouge", "semantic_similarity", "bert_score"]
                    ref_algorithms_to_run = [alg for alg in self.comparison_algorithms if alg in reference_algorithms]
                    
                    if ref_algorithms_to_run:
                        # Get reference text for current prompt
                        prompt_index = int(context.run_variation["prompt"].replace("prompt_", ""))
                        reference_text = self.reference_texts[prompt_index] if prompt_index < len(self.reference_texts) else self.reference_texts[0]
                        
                        ref_comparison_results = self.comparison_engine.run_comparison(
                            ref_algorithms_to_run,
                            [model_response],
                            [reference_text]
                        )
                        
                        # Add reference-based algorithm results
                        for algo_name, results in ref_comparison_results.items():
                            if results:
                                result = results[0]
                                result_data[f"{algo_name}_score"] = result.score
                                result_data[f"{algo_name}_execution_time_ms"] = result.execution_time_ms
                
            except Exception as e:
                logger.error(f"Error running comparison algorithms: {e}")
                # Add error indicators
                for algo_name in self.comparison_algorithms:
                    result_data[f"{algo_name}_score"] = -1
                    result_data[f"{algo_name}_execution_time_ms"] = -1
        
        return result_data
    
    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment."""
        output.console_log("LLM comparison experiment completed")
        
        # Cleanup loaded models
        self.loader.unload_all_models()
        
        # Generate summary report
        try:
            self._generate_summary_report()
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if not self.model_ids:
            raise ValueError("No model IDs specified")
        
        if not self.prompts:
            raise ValueError("No prompts specified")
        
        if not self.comparison_algorithms:
            raise ValueError("No comparison algorithms specified")
        
        # Check if algorithms are available
        available_algorithms = self.comparison_engine.get_available_algorithms()
        for algo in self.comparison_algorithms:
            if algo not in available_algorithms:
                logger.warning(f"Algorithm {algo} not available: missing dependencies")
    
    def _load_model_metadata(self):
        """Load metadata for all specified models."""
        output.console_log("Loading model metadata...")
        
        for model_id in self.model_ids:
            try:
                metadata = self.discovery.get_model_info(model_id)
                if metadata:
                    self.model_metadata[model_id] = metadata
                    if not metadata.compatible:
                        logger.warning(f"Model {model_id} may not be compatible: {metadata.compatibility_notes}")
                else:
                    logger.error(f"Could not load metadata for model {model_id}")
                    raise ValueError(f"Invalid model ID: {model_id}")
            except Exception as e:
                logger.error(f"Error loading metadata for {model_id}: {e}")
                raise
    
    def _preload_models(self):
        """Pre-load models to check for issues and cache them."""
        output.console_log("Pre-loading models...")
        
        for model_id in self.model_ids:
            try:
                metadata = self.model_metadata[model_id]
                loaded_model = self.loader.load_model(metadata)
                if loaded_model:
                    self.loaded_models[model_id] = loaded_model
                    output.console_log(f"Successfully loaded {model_id}")
                else:
                    raise RuntimeError(f"Failed to load {model_id}")
                    
            except Exception as e:
                logger.error(f"Error pre-loading {model_id}: {e}")
                # Remove from experiment if it can't be loaded
                self.model_ids.remove(model_id)
                if model_id in self.model_metadata:
                    del self.model_metadata[model_id]
        
        if not self.model_ids:
            raise RuntimeError("No models could be loaded successfully")
    
    def _generate_summary_report(self):
        """Generate a summary report of the experiment."""
        summary_path = self.results_output_path / self.name / "experiment_summary.json"
        
        summary = {
            "experiment_name": self.name,
            "models_compared": self.model_ids,
            "algorithms_used": self.comparison_algorithms,
            "prompts_count": len(self.prompts),
            "repetitions": self.repetitions,
            "total_runs": len(self.model_ids) * len(self.prompts) * self.repetitions,
            "memory_stats": self.loader.get_memory_stats() if hasattr(self, 'loader') else {},
            "model_metadata": {
                model_id: {
                    "author": metadata.author,
                    "downloads": metadata.downloads,
                    "size_gb": metadata.size_gb,
                    "license": metadata.license,
                    "compatible": metadata.compatible
                }
                for model_id, metadata in self.model_metadata.items()
            }
        }
        
        # Ensure directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        output.console_log(f"Summary report saved to {summary_path}")
    
    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None


# For backward compatibility and easier importing
RunnerConfig = LLMComparisonConfig