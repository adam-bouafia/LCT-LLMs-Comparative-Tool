"""
Energy-Profiled LLM Comparison Configuration

Production-ready configuration for energy-efficient LLM evaluation with granular
profiling of dataset loading, model initialization, inference, and metric calculation.
Uses CodeCarbon for comprehensive energy analysis (RAPL CPU + GPU + CO2 tracking).
"""

import sys
import json
import logging
import time
import subprocess
import shlex
import signal
import os
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path
from os.path import dirname, realpath

from experiment_runner.EventManager.Models.RunnerEvents import RunnerEvents

try:
    from experiment_runner.EventManager.EventSubscriptionController import (
        EventSubscriptionController,
    )
    from experiment_runner.ConfigValidator.Config.Models.RunTableModel import (
        RunTableModel,
    )
    from experiment_runner.ConfigValidator.Config.Models.RunnerContext import (
        RunnerContext as ExperimentRunnerContext,
    )
    from experiment_runner.ConfigValidator.Config.Models.OperationType import (
        OperationType as ExperimentOperationType,
    )
    from experiment_runner.ProgressManager.Output.OutputProcedure import (
        OutputProcedure as output,  # type: ignore[assignment]
    )
    
    # Type aliases for local use
    RunnerContext = ExperimentRunnerContext  # type: ignore[assignment,misc]
    OperationType = ExperimentOperationType  # type: ignore[assignment,misc]
except ImportError:

    class RunnerContext:  # type: ignore[no-redef]
        """Fallback context for standalone use."""

        def __init__(self):
            self.run_dir = Path(".")
            self.run_nr = 0
            self.run_variation: Dict[str, Any] = {}

    class OperationType:  # type: ignore[no-redef]
        """Fallback operation type."""

        AUTO = "AUTO"

    def output():
        """Fallback output function."""
        pass


from llm_runner.configs.llm_comparison_config import LLMComparisonConfig
from llm_runner.loaders.universal_loader import ModelLoadConfig
from llm_runner.data.reference_data_loader import (
    ReferenceDataLoader,
    load_reference_data,
)
from llm_runner.algorithms.comparison_algorithms import ComparisonEngine, ModelResponse
from llm_runner.profiling.energy_tracker import EnergyTracker, EnergyStage

import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)


try:
    from experiment_runner.ConfigValidator.Config.Models.FactorModel import FactorModel as ExperimentFactorModel
    from experiment_runner.ExtendedTyping.Typing import SupportsStr as ExperimentSupportsStr
    
    # Type aliases
    FactorModel = ExperimentFactorModel  # type: ignore[misc]
    SupportsStr = ExperimentSupportsStr  # type: ignore[misc]
except ImportError:

    class SupportsStr:  # type: ignore[no-redef]
        """Fallback string support type."""

        pass

    class FactorModel:  # type: ignore[no-redef]
        """Fallback factor model for standalone use."""

        def __init__(self, name, values):
            self.name = name
            self.values = values


from llm_runner.discovery.hf_model_discovery import (
    HuggingFaceModelDiscovery,
    ModelMetadata,
    ModelSearchCriteria,
)
from llm_runner.loaders.universal_loader import UniversalModelLoader, LoadedModel


class EnergyProfiledLLMConfig:
    """Energy-profiled LLM comparison configuration with power measurement capabilities."""

    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================

    """The name of the experiment."""
    name: str = "energy_profiled_llm_comparison"

    """The path in which Experiment Runner will create a folder with the name `self.name`."""
    results_output_path: Path = ROOT_DIR / "experiments"

    """Experiment operation type."""
    operation_type: Any = OperationType.AUTO  # type: ignore[assignment]

    """Time between runs (allow for cooling down)."""
    time_between_runs_in_ms: int = 5000  # 5 seconds for energy measurement cooldown

    """Whether to preload all models before experiment starts (slower startup, faster runs)."""
    preload_models: bool = False

    # ================================ LLM COMPARISON CONFIG ================================

    """List of model IDs to compare."""
    model_ids: List[str] = [
        "microsoft/DialoGPT-small",  # Lightweight for testing
        "distilgpt2",
        "gpt2",
    ]

    """Test prompts for comparison."""
    prompts: List[str] = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
    ]

    """Comparison algorithms to use."""
    comparison_algorithms: List[str] = [
        # Basic performance metrics (always work)
        "response_time",
        "text_length",
        "token_throughput",
        # Reference-requiring algorithms (work with reference data)
        "bleu",
        "rouge",
        "bert_score",
        "semantic_similarity",
        "semantic_textual_similarity",
        # Self-evaluating algorithms (work without reference data)
        "pairwise_comparison",
        "llm_as_judge",
        "g_eval",
        "rlhf_preference",
        "code_generation",
        "commonsense_reasoning",
        "mathematical_reasoning",
        "safety_alignment",
        "truthfulness",
    ]

    """Reference texts for reference-based algorithms (optional)."""
    reference_texts: Optional[List[str]] = None

    """Number of repetitions per variation."""
    repetitions: int = 3

    """Maximum memory per model in GB."""
    max_memory_gb: float = 8.0

    """Model loading configuration."""
    model_load_config: Optional[ModelLoadConfig] = None  # Initialized in __init__

    """Text generation parameters."""
    """Text generation parameters for LLM inference."""
    generation_params: Dict[str, Any] = {
        "max_length": 100,
        "temperature": 0.8,
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": 50256,
        # NOTE: max_new_tokens removed to avoid conflicts with max_length
        # Use max_length for consistent behavior
    }

    # ================================ ENERGY PROFILING CONFIG ================================

    """Energy profiler to use: 'codecarbon', 'none'"""
    energy_profiler: str = "codecarbon"  # Default to CodeCarbon (RAPL + GPU + CO2)

    """Energibridge specific settings."""
    energibridge_config: Dict[str, Any] = {
        "device_address": "/dev/ttyUSB0",
        "sample_rate": 1000,  # samples per second
    }

    """RAPL specific settings."""
    rapl_config: Dict[str, Any] = {
        "domains": ["package", "dram", "gpu"],  # RAPL domains to monitor
        "sample_interval": 1000,  # milliseconds
    }

    # ================================ REFERENCE DATA CONFIG ================================

    """Enable reference data for algorithms that require ground truth (BLEU, ROUGE, etc.)"""
    enable_reference_data: bool = True

    """Number of reference samples to load per dataset"""
    max_reference_samples: int = 100

    """Cache directory for downloaded reference datasets"""
    reference_cache_dir: Optional[str] = None

    def __init__(self):
        """
        Initialize the energy-profiled LLM comparison configuration.

        Sets up model loading, discovery, comparison engine, reference data,
        and granular energy tracking capabilities.
        """
        self.model_load_config = ModelLoadConfig()
        self.model_load_config.device = "auto"
        self.model_load_config.torch_dtype = "auto"
        self.model_load_config.trust_remote_code = False
        self.model_load_config.use_cache = True
        self.model_load_config.low_cpu_mem_usage = True

        EventSubscriptionController.subscribe_to_multiple_events(
            [
                (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
                (RunnerEvents.BEFORE_RUN, self.before_run),
                (RunnerEvents.START_RUN, self.start_run),
                (RunnerEvents.START_MEASUREMENT, self.start_measurement),
                (RunnerEvents.INTERACT, self.interact),
                (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
                (RunnerEvents.STOP_RUN, self.stop_run),
                (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
                (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
            ]
        )

        self.discovery = HuggingFaceModelDiscovery()
        self.loader = UniversalModelLoader(
            self.model_load_config, self.max_memory_gb, self.discovery
        )
        self.comparison_engine = ComparisonEngine()
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}

        self.energy_tracker = EnergyTracker(profiler_type=self.energy_profiler)

        self.reference_loader: Optional[ReferenceDataLoader] = None
        if self.enable_reference_data:
            try:
                self.reference_loader = load_reference_data(
                    max_samples_per_dataset=self.max_reference_samples,
                    cache_dir=self.reference_cache_dir,
                )
                logger.info(
                    f"Reference data loaded: {len(self.reference_loader.datasets)} datasets available"
                )
            except Exception as e:
                logger.warning(f"Failed to load reference data: {e}")
                logger.warning(
                    "Reference-requiring algorithms will not work without reference data"
                )
                self.reference_loader = None
        else:
            logger.info(
                "Reference data disabled - only standalone algorithms will work"
            )

        self.current_prompt = ""
        self.current_model_id = ""
        self.current_model_response: Optional[ModelResponse] = None

        # Energy profiling state
        self.profiler_process = None
        self.target_process = None

        self.run_table_model = None

        output.console_log("Energy-Profiled LLM Comparison Config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model for the experiment."""

        # Validate configuration
        self._validate_config()

        # Load model metadata
        self._load_model_metadata()

        # Create factors: models and prompts
        model_factor = FactorModel("model", self.model_ids)
        prompt_factor = FactorModel(
            "prompt", [f"prompt_{i}" for i in range(len(self.prompts))]
        )

        # Data columns for results
        data_columns = []

        # Add LLM comparison algorithm columns
        available_algorithms = self.comparison_engine.get_available_algorithms()
        for algo_name in self.comparison_algorithms:
            if algo_name in available_algorithms:
                data_columns.extend(
                    [f"{algo_name}_score", f"{algo_name}_execution_time_ms"]
                )
            else:
                logger.warning(f"Algorithm {algo_name} not available, skipping")

        # Add LLM performance metrics
        data_columns.extend(
            ["response_time_ms", "tokens_generated", "memory_usage_mb", "response_text"]
        )

        # Add energy profiling columns
        if self.energy_profiler == "codecarbon":
            data_columns.extend(
                [
                    "emissions_kg_co2",
                    "energy_consumed_kwh",
                    "cpu_energy_kwh",
                    "gpu_energy_kwh",
                    "ram_energy_kwh",
                    "power_efficiency_tokens_per_kwh",
                ]
            )

        self.run_table_model = RunTableModel(
            factors=[model_factor, prompt_factor],  # type: ignore[list-item]
            exclude_variations=[],
            repetitions=self.repetitions,
            data_columns=data_columns,
            shuffle=True,
        )

        output.console_log(
            f"Created run table with {len(self.model_ids)} models, {len(self.prompts)} prompts, and {self.energy_profiler} energy profiling"
        )
        return self.run_table_model

    def _validate_config(self):
        """Validate the configuration."""
        if not self.model_ids:
            raise ValueError("At least one model ID must be specified")
        if not self.prompts:
            raise ValueError("At least one prompt must be specified")
        if self.energy_profiler not in [
            "codecarbon",
            "none",
        ]:
            raise ValueError(f"Unsupported energy profiler: {self.energy_profiler}")

    def _load_model_metadata(self):
        """Load metadata for all models."""
        output.console_log("Loading model metadata...")
        print(f"🔍 LOADING METADATA for {len(self.model_ids)} models: {self.model_ids}")

        for model_id in self.model_ids:
            try:
                print(f"   📋 Fetching metadata for {model_id}...")

                # Try direct model lookup first (better for exact model IDs)
                metadata = self.discovery.get_model_info(model_id)

                if metadata:
                    self.model_metadata[model_id] = metadata
                    print(f"   ✅ Metadata loaded via direct lookup for {model_id}")
                    print(f"      • Downloads: {metadata.downloads:,}")
                    print(
                        f"      • Size: {metadata.size_gb:.1f}GB"
                        if metadata.size_gb
                        else "      • Size: Unknown"
                    )
                    print(f"      • Task: {metadata.task}")
                else:
                    print(f"   🔍 Direct lookup failed, trying search method...")
                    # Fallback to search method
                    criteria = ModelSearchCriteria(query=model_id, limit=1)
                    models = self.discovery.search_models(criteria)
                    if models:
                        self.model_metadata[model_id] = models[0]
                        print(f"   ✅ Metadata loaded via search for {model_id}")
                    else:
                        print(f"   ⚠️  No metadata found for {model_id}, using defaults")
                        logger.warning(f"Could not find metadata for {model_id}")
                        # Create minimal metadata
                        self.model_metadata[model_id] = ModelMetadata(
                            id=model_id,
                            author="unknown",
                            task="text-generation",
                            library="transformers",
                            language=None,
                            license=None,
                            downloads=0,
                            likes=0,
                            tags=[],
                            size_gb=0.5,  # Default small size estimate
                            created_at=None,
                            last_modified=None,
                            description=f"Minimal metadata for {model_id}",
                            pipeline_tag="text-generation",
                            model_card_available=False,
                        )
            except Exception as e:
                print(f"   ❌ Error loading metadata for {model_id}: {e}")
                logger.error(f"Error loading metadata for {model_id}: {e}")
                # Create minimal metadata for the model
                self.model_metadata[model_id] = ModelMetadata(
                    id=model_id,
                    author="unknown",
                    task="text-generation",
                    library="transformers",
                    language=None,
                    license=None,
                    downloads=0,
                    likes=0,
                    tags=[],
                    size_gb=0.5,  # Default small size estimate
                    created_at=None,
                    last_modified=None,
                    description=f"Fallback metadata for {model_id}",
                    pipeline_tag="text-generation",
                    model_card_available=False,
                )

        print(f"✅ METADATA LOADING COMPLETE for {len(self.model_metadata)} models")

    # ================================ EXPERIMENT RUNNER EVENT HANDLERS ================================

    def before_experiment(self) -> None:
        """
        Initialize experiment and track dataset loading energy.

        Pre-loads models if enabled and validates energy profiler availability.
        """
        output.console_log("Starting Energy-Profiled LLM Comparison Experiment")
        output.console_log(f"Models to compare: {', '.join(self.model_ids)}")
        output.console_log(f"Energy profiler: {self.energy_profiler}")
        output.console_log(f"Algorithms: {', '.join(self.comparison_algorithms)}")

        if self.energy_profiler == "codecarbon":
            output.console_log(
                "✓ CodeCarbon will track CPU (RAPL), GPU (nvidia-smi), and CO2"
            )

        try:
            self.energy_tracker.start_stage(EnergyStage.DATASET_LOADING)
        except Exception as e:
            logger.warning(f"Failed to start dataset loading energy tracking: {e}")

        preload_setting = getattr(self, "preload_models", True)

        if preload_setting:
            output.console_log("Model preloading enabled - loading all models")
            self._preload_models()
        else:
            output.console_log(
                "Model preloading disabled - on-demand loading during runs"
            )

        try:
            self.energy_tracker.stop_stage()
        except Exception as e:
            logger.warning(f"Failed to stop dataset loading energy tracking: {e}")

    def _preload_models(self):
        """
        Pre-load all models with granular energy tracking per model.

        Tracks model initialization energy for each model individually
        and removes models that fail to load.
        """
        output.console_log(f"Pre-loading {len(self.model_ids)} models...")

        for i, model_id in enumerate(self.model_ids, 1):
            try:
                output.console_log(
                    f"Loading model {i}/{len(self.model_ids)}: {model_id}"
                )
                metadata = self.model_metadata[model_id]

                try:
                    self.energy_tracker.start_stage(EnergyStage.MODEL_INITIALIZATION)
                except Exception as e:
                    logger.warning(f"Failed to start model init energy tracking: {e}")

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Model loading timed out after 300 seconds")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)

                try:
                    loaded_model = self.loader.load_model(metadata)
                    signal.alarm(0)

                    if loaded_model:
                        self.loaded_models[model_id] = loaded_model
                        output.console_log(
                            f"✓ Loaded {model_id} ({loaded_model.memory_usage_mb:.1f}MB)"
                        )
                    else:
                        logger.error(f"Failed to load {model_id}")
                        self.model_ids.remove(model_id)
                except TimeoutError as e:
                    signal.alarm(0)
                    logger.error(f"Timeout loading {model_id}: {e}")
                    self.model_ids.remove(model_id)

                try:
                    self.energy_tracker.stop_stage()
                except Exception as e:
                    logger.warning(f"Failed to stop model init energy tracking: {e}")

            except Exception as e:
                logger.error(f"Error loading {model_id}: {e}")
                self.model_ids.remove(model_id)

        output.console_log(f"Successfully loaded {len(self.loaded_models)} models")

    def before_run(self) -> None:
        """Perform any activity required before starting a run."""
        pass

    def start_run(self, context: RunnerContext) -> None:
        """
        Start a specific run with per-prompt energy tracking.

        Initializes prompt-level energy monitoring and loads model if needed.
        """
        print("\n" + "=" * 80)
        print(f"🚀 STARTING RUN #{context.run_nr + 1}")
        print("=" * 80)

        self.current_model_id = context.run_variation["model"]
        prompt_index = int(context.run_variation["prompt"].replace("prompt_", ""))
        self.current_prompt = self.prompts[prompt_index]

        print(f"📊 Run Configuration:")
        print(f"   • Model ID: {self.current_model_id}")
        print(f"   • Prompt #{prompt_index + 1}: '{self.current_prompt}'")
        print(f"   • Algorithms: {', '.join(self.comparison_algorithms)}")

        output.console_log(
            f"Starting run: {self.current_model_id} with prompt {prompt_index}"
        )

        try:
            self.energy_tracker.start_prompt(prompt_index, self.current_prompt)
            logger.debug(f"Started energy tracking for prompt {prompt_index}")
        except Exception as e:
            logger.warning(f"Failed to start prompt energy tracking: {e}")

        # Ensure model is loaded
        if self.current_model_id not in self.loaded_models:
            print(f"\n🔄 Loading model: {self.current_model_id}")
            print("   This may take a few minutes for first-time downloads...")

            metadata = self.model_metadata[self.current_model_id]
            print(f"   • Model size: ~{metadata.size_gb:.1f}GB")
            print(f"   • Task: {metadata.task or 'text-generation'}")
            if metadata.library:
                print(f"   • Library: {metadata.library}")

            loaded_model = self.loader.load_model(metadata)
            if loaded_model:
                self.loaded_models[self.current_model_id] = loaded_model
                print(f"   ✅ Model loaded successfully!")
                print(f"   • Memory usage: {loaded_model.memory_usage_mb:.1f}MB")
                print(f"   • Device: {loaded_model.device}")
            else:
                print(f"   ❌ Failed to load model!")
                raise RuntimeError(f"Failed to load model {self.current_model_id}")
        else:
            print(f"\n✅ Model already loaded: {self.current_model_id}")
            loaded_model = self.loaded_models[self.current_model_id]
            print(f"   • Memory usage: {loaded_model.memory_usage_mb:.1f}MB")
            print(f"   • Device: {loaded_model.device}")

    def start_measurement(self, context: RunnerContext) -> None:
        """Start energy measurement."""
        print(f"\n🔋 Starting energy measurement with {self.energy_profiler}")
        output.console_log(f"Starting energy measurement with {self.energy_profiler}")

        if self.energy_profiler == "codecarbon":
            print("   • Using CodeCarbon for RAPL CPU + GPU + CO2 tracking")
            self._start_codecarbon_measurement(context)

        print("   ✅ Energy monitoring started")

    def _start_codecarbon_measurement(self, context: RunnerContext):
        """Start CodeCarbon energy measurement."""
        try:
            from codecarbon import EmissionsTracker

            # Ensure output directory exists (CodeCarbon requires it)
            context.run_dir.mkdir(parents=True, exist_ok=True)

            self.emissions_tracker = EmissionsTracker(
                output_dir=str(context.run_dir),
                project_name=f"llm_comparison_{self.current_model_id}",
                experiment_id=str(context.run_nr),
            )
            self.emissions_tracker.start()
            output.console_log("CodeCarbon tracking started")
        except ImportError:
            logger.error(
                "CodeCarbon not installed. Install with: pip install codecarbon"
            )
            self.emissions_tracker = None
        except Exception as e:
            logger.error(f"Error starting CodeCarbon: {e}")
            self.emissions_tracker = None

    def interact(self, context: RunnerContext) -> None:
        """
        Generate response from model with granular energy tracking.

        Tracks inference energy consumption for the current prompt.
        """
        try:
            try:
                self.energy_tracker.start_stage(EnergyStage.INFERENCE)
            except Exception as e:
                logger.warning(f"Failed to start inference energy tracking: {e}")

            start_time = time.time()
            response_text = self.loader.generate_text(
                self.current_model_id, self.current_prompt, **self.generation_params
            )
            end_time = time.time()

            if response_text is None:
                raise RuntimeError(
                    f"Failed to generate response from {self.current_model_id}"
                )

            response_time_ms = (end_time - start_time) * 1000
            tokens_generated = len(response_text.split()) * 1.3

            memory_stats = self.loader.get_memory_stats()
            current_memory_mb = (
                memory_stats["loaded_models"]
                .get(self.current_model_id, {})
                .get("memory_mb", 0)
            )

            self.current_model_response = ModelResponse(
                model_id=self.current_model_id,
                prompt=self.current_prompt,
                response=response_text,
                response_time_ms=response_time_ms,
                tokens_generated=int(tokens_generated),
                memory_usage_mb=current_memory_mb,
            )

            try:
                self.energy_tracker.stop_stage()
            except Exception as e:
                logger.warning(f"Failed to stop inference energy tracking: {e}")

            output.console_log(
                f"Generated response in {response_time_ms:.1f}ms ({int(tokens_generated)} tokens)"
            )

        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            self.current_model_response = None

    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop energy measurement."""
        output.console_log(f"Stopping {self.energy_profiler} measurement")

        if (
            self.energy_profiler == "codecarbon"
            and hasattr(self, "emissions_tracker")
            and self.emissions_tracker
        ):
            try:
                self.emissions_tracker.stop()
                output.console_log("CodeCarbon tracking stopped")
            except Exception as e:
                logger.error(f"Error stopping CodeCarbon: {e}")

    def stop_run(self, context: RunnerContext) -> None:
        """
        Stop the current run and finalize per-prompt energy tracking.

        Captures final energy metrics for the current prompt.
        """
        try:
            prompt_index = int(context.run_variation["prompt"].replace("prompt_", ""))
            self.energy_tracker.stop_prompt(prompt_index)
            logger.debug(f"Stopped energy tracking for prompt {prompt_index}")
        except Exception as e:
            logger.warning(f"Failed to stop prompt energy tracking: {e}")

    def populate_run_data(
        self, context: RunnerContext
    ) -> Optional[Dict[str, SupportsStr]]:
        """Process measurement data and return results."""
        print(f"\n📊 Processing run results...")
        result_data = {}

        # 🎯 ADD COMPREHENSIVE CONTEXT INFORMATION TO CSV
        result_data.update(
            {
                "model": getattr(context, "current_model_id", None)
                or self.current_model_id,
                "prompt": getattr(context, "current_prompt", None)
                or self.current_prompt,
                "run_id": getattr(
                    context, "run_id", f"run_{getattr(context, 'run_nr', 0)}"
                ),
                "repetition": getattr(context, "repetition", 0),
                "experiment_name": self.name,
            }
        )

        # Add LLM performance metrics
        if self.current_model_response:
            response = self.current_model_response
            print(f"\n📝 Model Response Details:")
            print(f"   • Response time: {response.response_time_ms:.1f}ms")
            print(f"   • Tokens generated: {response.tokens_generated}")
            print(f"   • Memory usage: {response.memory_usage_mb:.1f}MB")
            print(
                f"   • Response preview: '{response.response[:100]}{'...' if len(response.response) > 100 else ''}'"
            )

            result_data.update(
                {
                    "response_time_ms": response.response_time_ms,
                    "tokens_generated": response.tokens_generated,
                    "memory_usage_mb": response.memory_usage_mb,
                    "response_text": (
                        response.response[:500] + "..."
                        if len(response.response) > 500
                        else response.response
                    ),
                }
            )

            try:
                self.energy_tracker.start_stage(EnergyStage.METRIC_CALCULATION)

                basic_algorithms = ["response_time", "text_length", "token_throughput"]
                reference_requiring_algorithms = [
                    "bleu",
                    "rouge",
                    "bert_score",
                    "semantic_similarity",
                    "semantic_textual_similarity",
                ]
                self_evaluating_algorithms = [
                    "pairwise_comparison",
                    "llm_as_judge",
                    "g_eval",
                    "rlhf_preference",
                    "code_generation",
                    "commonsense_reasoning",
                    "mathematical_reasoning",
                    "safety_alignment",
                    "truthfulness",
                ]

                basic_to_run = [
                    alg for alg in self.comparison_algorithms if alg in basic_algorithms
                ]
                if basic_to_run:
                    logger.info(f"Running basic algorithms: {basic_to_run}")

                    for algo_name in basic_to_run:
                        self.energy_tracker.start_algorithm(algo_name)

                    comparison_results = self.comparison_engine.run_comparison(
                        basic_to_run, [response], None
                    )

                    for algo_name, results in comparison_results.items():
                        if results:
                            result = results[0]
                            result_data[f"{algo_name}_score"] = result.score
                            result_data[f"{algo_name}_execution_time_ms"] = (
                                result.execution_time_ms
                            )
                            logger.info(f"✓ {algo_name}: {result.score:.2f}")
                            self.energy_tracker.stop_algorithm(algo_name)
                        else:
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                            logger.warning(f"✗ {algo_name}: no results")
                            self.energy_tracker.stop_algorithm(algo_name)

                self_eval_to_run = [
                    alg
                    for alg in self.comparison_algorithms
                    if alg in self_evaluating_algorithms
                ]
                if self_eval_to_run:
                    logger.info(
                        f"Running self-evaluating algorithms: {self_eval_to_run}"
                    )
                    for algo_name in self_eval_to_run:
                        try:
                            self.energy_tracker.start_algorithm(algo_name)
                            algorithm_results = self.comparison_engine.run_comparison(
                                [algo_name],
                                [response],
                                None,
                            )

                            if (
                                algo_name in algorithm_results
                                and algorithm_results[algo_name]
                            ):
                                result = algorithm_results[algo_name][0]
                                result_data[f"{algo_name}_score"] = result.score
                                result_data[f"{algo_name}_execution_time_ms"] = (
                                    result.execution_time_ms
                                )
                                logger.info(f"✓ {algo_name}: {result.score:.2f}")
                            else:
                                result_data[f"{algo_name}_score"] = None
                                result_data[f"{algo_name}_execution_time_ms"] = None
                                logger.warning(f"✗ {algo_name}: no results")

                            self.energy_tracker.stop_algorithm(algo_name)

                        except Exception as algo_error:
                            logger.warning(f"✗ {algo_name} failed: {algo_error}")
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                            self.energy_tracker.stop_algorithm(algo_name)

                # Run reference-requiring algorithms (if reference data available)
                ref_requiring_to_run = [
                    alg
                    for alg in self.comparison_algorithms
                    if alg in reference_requiring_algorithms
                ]
                if ref_requiring_to_run:
                    if self.reference_loader and self.reference_loader.is_available():
                        logger.info(
                            f"Running reference-requiring algorithms: {ref_requiring_to_run}"
                        )

                        # Get appropriate reference data for these algorithms
                        reference_data = self.reference_loader.get_reference_data_for_algorithms(
                            ref_requiring_to_run,
                            num_samples=10,  # Use fewer samples per response for performance
                        )

                        for algo_name in ref_requiring_to_run:
                            try:
                                # Get reference texts for this algorithm
                                refs = reference_data.get(algo_name, [])
                                if not refs:
                                    logger.warning(
                                        f"✗ {algo_name}: no reference data available"
                                    )
                                    result_data[f"{algo_name}_score"] = None
                                    result_data[f"{algo_name}_execution_time_ms"] = None
                                    continue

                                # Use first reference text for single-response evaluation
                                single_ref = [refs[0]] if refs else None

                                algorithm_results = (
                                    self.comparison_engine.run_comparison(
                                        [algo_name], [response], single_ref
                                    )
                                )

                                if (
                                    algo_name in algorithm_results
                                    and algorithm_results[algo_name]
                                ):
                                    result = algorithm_results[algo_name][0]
                                    result_data[f"{algo_name}_score"] = result.score
                                    result_data[f"{algo_name}_execution_time_ms"] = (
                                        result.execution_time_ms
                                    )
                                    logger.info(
                                        f"✓ {algo_name}: {result.score:.2f} (with reference data)"
                                    )
                                else:
                                    result_data[f"{algo_name}_score"] = None
                                    result_data[f"{algo_name}_execution_time_ms"] = None
                                    logger.warning(
                                        f"✗ {algo_name}: no results with reference data"
                                    )

                            except Exception as algo_error:
                                logger.warning(
                                    f"✗ {algo_name} failed with reference data: {algo_error}"
                                )
                                result_data[f"{algo_name}_score"] = None
                                result_data[f"{algo_name}_execution_time_ms"] = None
                    else:
                        logger.warning(
                            f"Reference data not available - skipping algorithms: {ref_requiring_to_run}"
                        )
                        for algo_name in ref_requiring_to_run:
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                            logger.info(
                                f"⚠ {algo_name}: requires reference data (not loaded)"
                            )

                self.energy_tracker.stop_stage()
            except Exception as e:
                logger.error(f"Error running comparison algorithms: {e}")
                for algo_name in self.comparison_algorithms:
                    if f"{algo_name}_score" not in result_data:
                        result_data[f"{algo_name}_score"] = None
                        result_data[f"{algo_name}_execution_time_ms"] = None
                self.energy_tracker.stop_stage()

        if self.energy_profiler == "codecarbon":
            energy_data = self._parse_codecarbon_results(context)
            result_data.update(energy_data)

        energy_summary = self.energy_tracker.get_full_report()
        result_data.update(
            {
                "energy_dataset_loading_joules": energy_summary["stages"]
                .get(EnergyStage.DATASET_LOADING.value, {})
                .get("total_energy_joules", 0),
                "energy_model_init_joules": energy_summary["stages"]
                .get(EnergyStage.MODEL_INITIALIZATION.value, {})
                .get("total_energy_joules", 0),
                "energy_inference_joules": energy_summary["stages"]
                .get(EnergyStage.INFERENCE.value, {})
                .get("total_energy_joules", 0),
                "energy_metrics_joules": energy_summary["stages"]
                .get(EnergyStage.METRIC_CALCULATION.value, {})
                .get("total_energy_joules", 0),
                "energy_total_joules": energy_summary["total_energy_joules"],
            }
        )

        self._save_complete_run_data(context, result_data)

        return result_data

    def _parse_codecarbon_results(self, context: RunnerContext) -> Dict[str, float]:
        """Parse CodeCarbon results - COMPREHENSIVE energy data capture."""
        energy_data = {
            "emissions_kg_co2": 0.0,
            "energy_consumed_kwh": 0.0,
            "cpu_energy_kwh": 0.0,
            "gpu_energy_kwh": 0.0,
            "ram_energy_kwh": 0.0,
            # Additional CodeCarbon fields
            "duration_seconds": 0.0,
            "emissions_rate_kg_co2_per_s": 0.0,
            "cpu_power_w": 0.0,
            "gpu_power_w": 0.0,
            "ram_power_w": 0.0,
            "cpu_count": 0,
            "gpu_count": 0,
            "tracking_mode": "unknown",
            "country_name": "unknown",
            "country_iso_code": "unknown",
            "pue": 1.0,
        }

        try:
            # CodeCarbon saves emissions.csv
            csv_path = context.run_dir / "emissions.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)

                if not df.empty:
                    # Get the last (most recent) record
                    last_record = df.iloc[-1]

                    # Core energy metrics - handle NaN values safely
                    energy_data["emissions_kg_co2"] = self._safe_float(
                        last_record.get("emissions", 0)
                    )
                    energy_data["energy_consumed_kwh"] = self._safe_float(
                        last_record.get("energy_consumed", 0)
                    )
                    energy_data["cpu_energy_kwh"] = self._safe_float(
                        last_record.get("cpu_energy", 0)
                    )
                    energy_data["gpu_energy_kwh"] = self._safe_float(
                        last_record.get("gpu_energy", 0)
                    )
                    energy_data["ram_energy_kwh"] = self._safe_float(
                        last_record.get("ram_energy", 0)
                    )

                    # Additional useful fields from CodeCarbon - handle NaN values safely
                    energy_data["duration_seconds"] = self._safe_float(
                        last_record.get("duration", 0)
                    )
                    energy_data["emissions_rate_kg_co2_per_s"] = self._safe_float(
                        last_record.get("emissions_rate", 0)
                    )
                    energy_data["cpu_power_w"] = self._safe_float(
                        last_record.get("cpu_power", 0)
                    )
                    energy_data["gpu_power_w"] = self._safe_float(
                        last_record.get("gpu_power", 0)
                    )
                    energy_data["ram_power_w"] = self._safe_float(
                        last_record.get("ram_power", 0)
                    )
                    energy_data["cpu_count"] = self._safe_int(
                        last_record.get("cpu_count", 0)
                    )
                    energy_data["gpu_count"] = self._safe_int(
                        last_record.get("gpu_count", 0)
                    )
                    energy_data["tracking_mode"] = str(
                        last_record.get("tracking_mode", "unknown")
                    )
                    energy_data["country_name"] = str(
                        last_record.get("country_name", "unknown")
                    )
                    energy_data["country_iso_code"] = str(
                        last_record.get("country_iso_code", "unknown")
                    )
                    energy_data["pue"] = self._safe_float(last_record.get("pue", 1.0))

                    if energy_data["duration_seconds"] > 0:
                        if (
                            self.current_model_response
                            and self.current_model_response.tokens_generated is not None
                            and energy_data["cpu_power_w"] > 0
                        ):
                            tokens_per_second = (
                                self.current_model_response.tokens_generated
                                / (energy_data["duration_seconds"] or 1)
                            )
                            total_power_w = (
                                energy_data["cpu_power_w"]
                                + energy_data["gpu_power_w"]
                                + energy_data["ram_power_w"]
                            )
                            if total_power_w > 0:
                                energy_data["tokens_per_second_per_watt"] = (
                                    tokens_per_second / total_power_w
                                )
                            else:
                                energy_data["tokens_per_second_per_watt"] = 0.0
                        else:
                            energy_data["tokens_per_second_per_watt"] = 0.0

        except Exception as e:
            logger.error(f"Error parsing CodeCarbon results: {e}")

        return energy_data

    def after_experiment(self) -> None:
        """
        Finalize experiment and export comprehensive energy reports.

        Generates detailed energy analysis reports and cleans up resources.
        """
        output.console_log("Energy-Profiled LLM Comparison Experiment completed")

        try:
            self._export_energy_reports()
        except Exception as e:
            logger.error(f"Failed to export energy reports: {e}")

        for model_id in self.loaded_models:
            try:
                self.loader.unload_model(model_id)
            except Exception as e:
                logger.error(f"Error unloading {model_id}: {e}")

        output.console_log("All models unloaded. Experiment complete.")

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float, handling NaN and None cases."""
        try:
            if value is None:
                return default
            converted = float(value)
            # Check for NaN
            if converted != converted:  # NaN != NaN is True
                return default
            return converted
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert value to int, handling NaN and None cases."""
        try:
            if value is None:
                return default
            # First convert to float to handle NaN
            converted_float = float(value)
            if converted_float != converted_float:  # NaN check
                return default
            return int(converted_float)
        except (ValueError, TypeError):
            return default

    def _extract_repetition_from_context(self, context) -> int:
        """Extract repetition number from context, using run_dir path if available."""
        try:
            # Try to get repetition from context attributes first
            if hasattr(context, "repetition"):
                return getattr(context, "repetition")

            # Try to extract from run_dir path (like "run_0_repetition_0")
            if hasattr(context, "run_dir"):
                run_dir_name = str(context.run_dir).split("/")[
                    -1
                ]  # Get just the folder name
                if "repetition_" in run_dir_name:
                    repetition_part = run_dir_name.split("repetition_")[-1]
                    return int(
                        repetition_part.split("_")[0]
                    )  # Handle any trailing parts

            # Fallback to 0 if no repetition info found
            return 0

        except (ValueError, IndexError, AttributeError):
            return 0

    def _save_complete_run_data(
        self, context: RunnerContext, result_data: Dict[str, Any]
    ) -> None:
        """Save comprehensive run data to individual run folder as requested by user.

        This implements the user's request: 'i want the csv to collect all the infos,
        even the emissions, not run x repetition x folder have only emissions,
        it is not smart, i want to keep this folder thing yet it contain the
        info of that repetition not just emission'
        """
        try:
            import json
            from pathlib import Path

            # Get the run directory path
            run_dir = Path(context.run_dir)
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=True)

            # Save comprehensive run data as JSON
            run_data_file = run_dir / "run_data.json"

            # Create comprehensive data structure
            comprehensive_data = {
                "experiment_info": {
                    "name": self.name,
                    "model_id": self.current_model_id,
                    "prompt": self.current_prompt,
                    "repetition": self._extract_repetition_from_context(context),
                    "run_id": getattr(
                        context, "run_id", f"run_{getattr(context, 'run_nr', 0)}"
                    ),
                    "timestamp": getattr(
                        context, "run_id", f"run_{getattr(context, 'run_nr', 0)}"
                    ),
                },
                "model_metadata": self.model_metadata.get(self.current_model_id, {}),
                "generation_params": self.generation_params.copy(),
                "algorithm_results": result_data.copy(),
                "model_response": {
                    "text": (
                        getattr(self.current_model_response, "text", "")
                        if self.current_model_response
                        else ""
                    ),
                    "response_time": result_data.get("response_time", 0),
                    "token_count": result_data.get("token_throughput", 0),
                },
            }

            # Write comprehensive data to JSON file
            with open(run_data_file, "w", encoding="utf-8") as f:
                json.dump(comprehensive_data, f, indent=2, default=str)

            # Also save a simple CSV for easy analysis
            run_csv_file = run_dir / "run_results.csv"

            # Convert result_data to CSV format
            import pandas as pd

            df = pd.DataFrame([result_data])
            df.to_csv(run_csv_file, index=False)

            logger.info(f"✅ Complete run data saved to {run_dir}")

        except Exception as e:
            logger.error(f"❌ Error saving complete run data: {e}")
            pass

    def _export_energy_reports(self) -> None:
        """
        Export comprehensive energy consumption reports.

        Generates detailed JSON and summary CSV reports with granular energy breakdowns
        by stage, algorithm, and prompt.
        """
        try:
            report_dir = self.results_output_path / self.name / "energy_reports"
            report_dir.mkdir(parents=True, exist_ok=True)

            full_report = self.energy_tracker.get_full_report()

            report_file = report_dir / "energy_analysis_full.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(full_report, f, indent=2, default=str)

            logger.info(f"✅ Full energy report saved to {report_file}")

            summary_data = []

            for stage_name, stage_data in full_report["stages"].items():
                summary_data.append(
                    {
                        "category": "stage",
                        "name": stage_name,
                        "total_energy_joules": stage_data.get("total_energy_joules", 0),
                        "avg_power_watts": stage_data.get("avg_power_watts", 0),
                        "duration_seconds": stage_data.get("total_duration_seconds", 0),
                        "measurement_count": stage_data.get("measurement_count", 0),
                    }
                )

            for algo_name, algo_data in full_report.get("algorithms", {}).items():
                summary_data.append(
                    {
                        "category": "algorithm",
                        "name": algo_name,
                        "total_energy_joules": algo_data.get("energy_joules", 0),
                        "avg_power_watts": algo_data.get("power_watts", 0),
                        "duration_seconds": algo_data.get("execution_time_ms", 0)
                        / 1000,
                        "measurement_count": 1,
                    }
                )

            for prompt_id, prompt_data in full_report.get("prompts", {}).items():
                summary_data.append(
                    {
                        "category": "prompt",
                        "name": f"prompt_{prompt_id}",
                        "total_energy_joules": prompt_data.get(
                            "total_energy_joules", 0
                        ),
                        "avg_power_watts": prompt_data.get("avg_power_watts", 0),
                        "duration_seconds": prompt_data.get("duration_seconds", 0),
                        "measurement_count": 1,
                    }
                )

            if summary_data:
                summary_file = report_dir / "energy_summary.csv"
                df = pd.DataFrame(summary_data)
                df.to_csv(summary_file, index=False)
                logger.info(f"✅ Energy summary saved to {summary_file}")

            output.console_log(f"📊 Energy reports exported to {report_dir}")
            print(f"\n📊 Energy Analysis Summary:")
            print(
                f"   • Total energy consumed: {full_report['total_energy_joules']:.2f} J"
            )
            print(f"   • Average power: {full_report['avg_power_watts']:.2f} W")
            print(f"   • Total duration: {full_report['total_duration_seconds']:.2f} s")
            print(f"   • Stages tracked: {len(full_report['stages'])}")
            print(f"   • Algorithms tracked: {len(full_report.get('algorithms', {}))}")
            print(f"   • Prompts tracked: {len(full_report.get('prompts', {}))}")

        except Exception as e:
            logger.error(f"Failed to export energy reports: {e}")
            raise


# Export the RunnerConfig class for Experiment Runner
RunnerConfig = EnergyProfiledLLMConfig
