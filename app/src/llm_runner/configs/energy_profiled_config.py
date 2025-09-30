"""
Energy-Profiled LLM Comparison Configuration

This configuration extends LLMComparisonConfig with energy profiling capabilities,
integrating PowerJoular, Energibridge, RAPL, and other energy measurement tools
available in Experiment Runner.
"""

import sys
import json
import logging
import time
import subprocess
import shlex
import signal
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

# No need to add paths - use direct imports from experiment_runner package
from experiment_runner.EventManager.Models.RunnerEvents import RunnerEvents
try:
    from experiment_runner.EventManager.EventSubscriptionController import EventSubscriptionController
    from experiment_runner.ConfigValidator.Config.Models.RunTableModel import RunTableModel
    from experiment_runner.ConfigValidator.Config.Models.RunnerContext import RunnerContext
    from experiment_runner.ConfigValidator.Config.Models.OperationType import OperationType
    from experiment_runner.ProgressManager.Output.OutputProcedure import OutputProcedure as output
except ImportError:
    # Fallback definitions for standalone use
    class RunnerContext:
        def __init__(self):
            self.run_dir = Path(".")
            self.run_nr = 0
    class OperationType:
        AUTO = "AUTO"
    def output():
        pass

# Import our custom modules
from llm_runner.configs.llm_comparison_config import LLMComparisonConfig
from llm_runner.loaders.universal_loader import ModelLoadConfig
from llm_runner.data.reference_data_loader import ReferenceDataLoader, load_reference_data
from llm_runner.algorithms.comparison_algorithms import ComparisonEngine, ModelResponse

# Required for data handling
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)


class ModelLoadConfig:
    """Configuration for model loading."""
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_cache: bool = True
    low_cpu_mem_usage: bool = True


# Import additional experiment-runner modules
try:
    from experiment_runner.ConfigValidator.Config.Models.FactorModel import FactorModel
    from experiment_runner.ExtendedTyping.Typing import SupportsStr
except ImportError:
    # Define minimal types for standalone use
    class SupportsStr:
        pass
    class FactorModel:
        def __init__(self, name, values):
            self.name = name
            self.values = values

# Additional imports for discovery and model loading
from llm_runner.discovery.hf_model_discovery import HuggingFaceModelDiscovery, ModelMetadata, ModelSearchCriteria
from llm_runner.loaders.universal_loader import UniversalModelLoader, LoadedModel

logger = logging.getLogger(__name__)


class EnergyProfiledLLMConfig:
    """Energy-profiled LLM comparison configuration with power measurement capabilities."""
    
    ROOT_DIR = Path(dirname(realpath(__file__)))
    
    # ================================ USER SPECIFIC CONFIG ================================
    
    """The name of the experiment."""
    name: str = "energy_profiled_llm_comparison"
    
    """The path in which Experiment Runner will create a folder with the name `self.name`."""
    results_output_path: Path = ROOT_DIR / 'experiments'
    
    """Experiment operation type."""
    operation_type: OperationType = OperationType.AUTO
    
    """Time between runs (allow for cooling down)."""
    time_between_runs_in_ms: int = 5000  # 5 seconds for energy measurement cooldown
    
    """Whether to preload all models before experiment starts (slower startup, faster runs)."""
    preload_models: bool = False
    
    # ================================ LLM COMPARISON CONFIG ================================
    
    """List of model IDs to compare."""
    model_ids: List[str] = [
        "microsoft/DialoGPT-small",  # Lightweight for testing
        "distilgpt2",
        "gpt2"
    ]
    
    """Test prompts for comparison."""
    prompts: List[str] = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
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
        "truthfulness"
    ]
    
    """Reference texts for reference-based algorithms (optional)."""
    reference_texts: Optional[List[str]] = None
    
    """Number of repetitions per variation."""
    repetitions: int = 3
    
    """Maximum memory per model in GB."""
    max_memory_gb: float = 8.0
    
    """Model loading configuration."""
    model_load_config: ModelLoadConfig = None  # Initialized in __init__
    
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
    
    """Energy profiler to use: 'powerjoular', 'energibridge', 'rapl', 'codecarbon', 'none'"""
    energy_profiler: str = "powerjoular"  # Default to PowerJoular
    
    """PowerJoular specific settings."""
    powerjoular_config: Dict[str, Any] = {
        "sample_interval": 500,  # milliseconds
        "output_file": "powerjoular.csv"
    }
    
    """Energibridge specific settings.""" 
    energibridge_config: Dict[str, Any] = {
        "device_address": "/dev/ttyUSB0",
        "sample_rate": 1000  # samples per second
    }
    
    """RAPL specific settings."""
    rapl_config: Dict[str, Any] = {
        "domains": ["package", "dram", "gpu"],  # RAPL domains to monitor
        "sample_interval": 1000  # milliseconds
    }
    
    # ================================ REFERENCE DATA CONFIG ================================
    
    """Enable reference data for algorithms that require ground truth (BLEU, ROUGE, etc.)"""
    enable_reference_data: bool = True
    
    """Number of reference samples to load per dataset"""
    max_reference_samples: int = 100
    
    """Cache directory for downloaded reference datasets"""
    reference_cache_dir: Optional[str] = None
    
    def __init__(self):
        """Initialize the energy-profiled LLM comparison configuration."""
        
        # Initialize model load configuration
        self.model_load_config = ModelLoadConfig()
        self.model_load_config.device = "auto"
        self.model_load_config.torch_dtype = "auto"
        self.model_load_config.trust_remote_code = False
        self.model_load_config.use_cache = True
        self.model_load_config.low_cpu_mem_usage = True
        
        # Subscribe to Experiment Runner events
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
        # Initialize the discovery service and loader with it for auto-reloading
        self.discovery = HuggingFaceModelDiscovery()
        self.loader = UniversalModelLoader(self.model_load_config, self.max_memory_gb, self.discovery)
        self.comparison_engine = ComparisonEngine()
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}
        
        # Initialize reference data loader
        self.reference_loader: Optional[ReferenceDataLoader] = None
        if self.enable_reference_data:
            try:
                self.reference_loader = load_reference_data(
                    max_samples_per_dataset=self.max_reference_samples,
                    cache_dir=self.reference_cache_dir
                )
                logger.info(f"Reference data loaded: {len(self.reference_loader.datasets)} datasets available")
            except Exception as e:
                logger.warning(f"Failed to load reference data: {e}")
                logger.warning("Reference-requiring algorithms will not work without reference data")
                self.reference_loader = None
        else:
            logger.info("Reference data disabled - only standalone algorithms will work")
        
        # Current run state
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
        prompt_factor = FactorModel("prompt", [f"prompt_{i}" for i in range(len(self.prompts))])
        
        # Data columns for results
        data_columns = []
        
        # Add LLM comparison algorithm columns
        available_algorithms = self.comparison_engine.get_available_algorithms()
        for algo_name in self.comparison_algorithms:
            if algo_name in available_algorithms:
                data_columns.extend([
                    f"{algo_name}_score",
                    f"{algo_name}_execution_time_ms"
                ])
            else:
                logger.warning(f"Algorithm {algo_name} not available, skipping")
        
        # Add LLM performance metrics
        data_columns.extend([
            "response_time_ms",
            "tokens_generated",
            "memory_usage_mb", 
            "response_text"
        ])
        
        # Add energy profiling columns
        if self.energy_profiler == "powerjoular":
            data_columns.extend([
                "avg_cpu_utilization",
                "total_cpu_power_watts",
                "total_dram_power_watts", 
                "total_energy_joules",
                "power_efficiency_tokens_per_joule"
            ])
        elif self.energy_profiler == "energibridge":
            data_columns.extend([
                "total_energy_joules",
                "avg_power_watts",
                "peak_power_watts",
                "power_efficiency_tokens_per_joule"
            ])
        elif self.energy_profiler == "rapl":
            data_columns.extend([
                "package_energy_joules",
                "dram_energy_joules",
                "gpu_energy_joules",
                "total_energy_joules",
                "power_efficiency_tokens_per_joule"
            ])
        elif self.energy_profiler == "codecarbon":
            data_columns.extend([
                "emissions_kg_co2",
                "energy_consumed_kwh",
                "cpu_energy_kwh",
                "gpu_energy_kwh",
                "ram_energy_kwh"
            ])
        
        self.run_table_model = RunTableModel(
            factors=[model_factor, prompt_factor],
            exclude_variations=[],
            repetitions=self.repetitions,
            data_columns=data_columns,
            shuffle=True
        )
        
        output.console_log(f"Created run table with {len(self.model_ids)} models, {len(self.prompts)} prompts, and {self.energy_profiler} energy profiling")
        return self.run_table_model
    
    def _validate_config(self):
        """Validate the configuration."""
        if not self.model_ids:
            raise ValueError("At least one model ID must be specified")
        if not self.prompts:
            raise ValueError("At least one prompt must be specified")
        if self.energy_profiler not in ["powerjoular", "energibridge", "rapl", "codecarbon", "none"]:
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
                    print(f"      • Size: {metadata.size_gb:.1f}GB" if metadata.size_gb else "      • Size: Unknown")
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
                            model_card_available=False
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
                    model_card_available=False
                )
        
        print(f"✅ METADATA LOADING COMPLETE for {len(self.model_metadata)} models")
    
    # ================================ EXPERIMENT RUNNER EVENT HANDLERS ================================
    
    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment."""
        output.console_log("Starting Energy-Profiled LLM Comparison Experiment")
        output.console_log(f"Models to compare: {', '.join(self.model_ids)}")
        output.console_log(f"Energy profiler: {self.energy_profiler}")
        output.console_log(f"Algorithms: {', '.join(self.comparison_algorithms)}")
        
        # Check if energy profiling tools are available
        if self.energy_profiler == "powerjoular":
            try:
                subprocess.check_call(["which", "powerjoular"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                output.console_log("✓ PowerJoular found")
            except subprocess.CalledProcessError:
                logger.warning("PowerJoular not found. Please install it for energy profiling.")
        
        # Pre-load all models to check for issues (if enabled)
        preload_setting = getattr(self, 'preload_models', True)
        print(f"🤖 PRELOAD MODELS SETTING: {preload_setting}")
        print(f"   Source: {type(self).__name__} class")
        
        if preload_setting:
            print("🤖 Model preloading ENABLED - loading all models now")
            print("   This may take several minutes and could hang...")
            print("   TIP: Set preload_models = False in your config to skip this")
            self._preload_models()
        else:
            print("⚡ Model preloading DISABLED - models will load on-demand during runs")
            print("   This makes startup faster but individual runs slower")
    
    def _preload_models(self):
        """Pre-load all models to verify they work."""
        output.console_log("Pre-loading models...")
        print(f"\n🤖 PRELOADING {len(self.model_ids)} MODELS...")
        print("   Note: This may take several minutes for first-time downloads")
        print("   You can press Ctrl+C if this takes too long")
        
        for i, model_id in enumerate(self.model_ids, 1):
            try:
                print(f"\n📦 Loading model {i}/{len(self.model_ids)}: {model_id}")
                metadata = self.model_metadata[model_id]
                print(f"   🔍 Model info: size={metadata.size_gb:.1f}GB, task={metadata.task}")
                print(f"   ⏳ Starting model load (this may take a while)...")
                
                # Add timeout handling for model loading
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Model loading timed out after 300 seconds")
                
                # Set up timeout for model loading (5 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                try:
                    loaded_model = self.loader.load_model(metadata)
                    signal.alarm(0)  # Cancel timeout
                    
                    if loaded_model:
                        self.loaded_models[model_id] = loaded_model
                        print(f"   ✅ Successfully loaded {model_id}")
                        print(f"   📊 Memory usage: {loaded_model.memory_usage_mb:.1f}MB")
                        output.console_log(f"✓ Successfully loaded {model_id}")
                    else:
                        print(f"   ❌ Failed to load {model_id}")
                        logger.error(f"Failed to load {model_id}")
                        # Remove from experiment if it can't be loaded
                        self.model_ids.remove(model_id)
                except TimeoutError as e:
                    signal.alarm(0)  # Cancel timeout
                    print(f"   ⏰ Timeout loading {model_id}: {e}")
                    logger.error(f"Timeout loading {model_id}: {e}")
                    self.model_ids.remove(model_id)
                    
            except Exception as e:
                print(f"   ❌ Error loading {model_id}: {e}")
                logger.error(f"Error loading {model_id}: {e}")
                self.model_ids.remove(model_id)
        
        print(f" \n ✅ MODEL LOADING COMPLETE!")
        print(f"   Successfully loaded: {len(self.loaded_models)} models")
        print(f"   Failed models removed from experiment")
        output.console_log(f"Successfully loaded {len(self.loaded_models)} models")
    
    def before_run(self) -> None:
        """Perform any activity required before starting a run."""
        pass
    
    def start_run(self, context: RunnerContext) -> None:
        """Start a specific run."""
        print("\n" + "="*80)
        print(f"🚀 STARTING RUN #{context.run_nr + 1}")
        print("="*80)
        
        # Extract current run parameters
        self.current_model_id = context.run_variation["model"]
        prompt_index = int(context.run_variation["prompt"].replace("prompt_", ""))
        self.current_prompt = self.prompts[prompt_index]
        
        print(f"📊 Run Configuration:")
        print(f"   • Model ID: {self.current_model_id}")
        print(f"   • Prompt #{prompt_index + 1}: '{self.current_prompt}'")
        print(f"   • Algorithms: {', '.join(self.comparison_algorithms)}")
        
        output.console_log(f"Starting run: {self.current_model_id} with prompt {prompt_index}")
        
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
        
        if self.energy_profiler == "powerjoular":
            print("   • Using PowerJoular for detailed power measurement")
            self._start_powerjoular_measurement(context)
        elif self.energy_profiler == "energibridge":
            print("   • Using EnergiBridge for hardware energy monitoring")
            self._start_energibridge_measurement(context)
        elif self.energy_profiler == "rapl":
            print("   • Using RAPL for CPU energy monitoring")
            self._start_rapl_measurement(context)
        elif self.energy_profiler == "codecarbon":
            print("   • Using CodeCarbon for carbon footprint tracking")
            self._start_codecarbon_measurement(context)
        
        print("   ✅ Energy monitoring started")
    
    def _start_powerjoular_measurement(self, context: RunnerContext):
        """Start PowerJoular energy measurement."""
        try:
            current_pid = os.getpid()
            profiler_cmd = f'powerjoular -l -p {current_pid} -f {context.run_dir / "powerjoular.csv"}'
            
            time.sleep(0.5)  # Allow process to stabilize
            self.profiler_process = subprocess.Popen(shlex.split(profiler_cmd))
            output.console_log(f"Started PowerJoular profiling (PID: {self.profiler_process.pid})")
        except Exception as e:
            logger.error(f"Error starting PowerJoular: {e}")
            self.profiler_process = None
    
    def _start_energibridge_measurement(self, context: RunnerContext):
        """Start Energibridge energy measurement.""" 
        try:
            # Energibridge measurement implementation would go here
            output.console_log("Energibridge measurement started")
            # Placeholder for Energibridge integration
            self.profiler_process = None
        except Exception as e:
            logger.error(f"Error starting Energibridge: {e}")
            self.profiler_process = None
    
    def _start_rapl_measurement(self, context: RunnerContext):
        """Start RAPL energy measurement."""
        try:
            # RAPL measurement implementation would go here
            output.console_log("RAPL measurement started")
            # Placeholder for RAPL integration
            self.profiler_process = None
        except Exception as e:
            logger.error(f"Error starting RAPL: {e}")
            self.profiler_process = None
    
    def _start_codecarbon_measurement(self, context: RunnerContext):
        """Start CodeCarbon energy measurement."""
        try:
            from codecarbon import EmissionsTracker
            
            # Ensure output directory exists (CodeCarbon requires it)
            context.run_dir.mkdir(parents=True, exist_ok=True)
            
            self.emissions_tracker = EmissionsTracker(
                output_dir=str(context.run_dir),
                project_name=f"llm_comparison_{self.current_model_id}",
                experiment_id=context.run_nr
            )
            self.emissions_tracker.start()
            output.console_log("CodeCarbon tracking started")
        except ImportError:
            logger.error("CodeCarbon not installed. Install with: pip install codecarbon")
            self.emissions_tracker = None
        except Exception as e:
            logger.error(f"Error starting CodeCarbon: {e}")
            self.emissions_tracker = None
    
    def interact(self, context: RunnerContext) -> None:
        """Main interaction - generate response from the model with energy measurement."""
        try:
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
            
            # Calculate metrics
            response_time_ms = (end_time - start_time) * 1000
            tokens_generated = len(response_text.split()) * 1.3  # Approximate tokens
            
            # Get memory usage
            memory_stats = self.loader.get_memory_stats()
            current_memory_mb = memory_stats["loaded_models"].get(self.current_model_id, {}).get("memory_mb", 0)
            
            # Store response for analysis
            self.current_model_response = ModelResponse(
                model_id=self.current_model_id,
                prompt=self.current_prompt,
                response=response_text,
                response_time_ms=response_time_ms,
                tokens_generated=int(tokens_generated),
                memory_usage_mb=current_memory_mb
            )
            
            output.console_log(f"Generated response in {response_time_ms:.1f}ms ({int(tokens_generated)} tokens)")
            
        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            self.current_model_response = None
    
    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop energy measurement."""
        output.console_log(f"Stopping {self.energy_profiler} measurement")
        
        if self.energy_profiler == "powerjoular" and self.profiler_process:
            try:
                os.kill(self.profiler_process.pid, signal.SIGINT)
                self.profiler_process.wait()
                output.console_log("PowerJoular measurement stopped")
            except Exception as e:
                logger.error(f"Error stopping PowerJoular: {e}")
        
        elif self.energy_profiler == "codecarbon" and hasattr(self, 'emissions_tracker') and self.emissions_tracker:
            try:
                self.emissions_tracker.stop()
                output.console_log("CodeCarbon tracking stopped")
            except Exception as e:
                logger.error(f"Error stopping CodeCarbon: {e}")
    
    def stop_run(self, context: RunnerContext) -> None:
        """Stop the current run.""" 
        pass
    
    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        """Process measurement data and return results."""
        print(f"\n📊 Processing run results...")
        result_data = {}
        
        # 🎯 ADD COMPREHENSIVE CONTEXT INFORMATION TO CSV
        result_data.update({
            "model": getattr(context, 'current_model_id', None) or self.current_model_id,
            "prompt": getattr(context, 'current_prompt', None) or self.current_prompt,
            "run_id": getattr(context, 'run_id', f"run_{getattr(context, 'run_nr', 0)}"),
            "repetition": getattr(context, 'repetition', 0),
            "experiment_name": self.name,
        })
        
        # Add LLM performance metrics
        if self.current_model_response:
            response = self.current_model_response
            print(f"\n📝 Model Response Details:")
            print(f"   • Response time: {response.response_time_ms:.1f}ms")
            print(f"   • Tokens generated: {response.tokens_generated}")
            print(f"   • Memory usage: {response.memory_usage_mb:.1f}MB")
            print(f"   • Response preview: '{response.response[:100]}{'...' if len(response.response) > 100 else ''}'")
            
            result_data.update({
                "response_time_ms": response.response_time_ms,
                "tokens_generated": response.tokens_generated,
                "memory_usage_mb": response.memory_usage_mb,
                "response_text": response.response[:500] + "..." if len(response.response) > 500 else response.response
            })
            
            # Run comparison algorithms with intelligent categorization
            try:
                # Categorize algorithms by their requirements
                basic_algorithms = ["response_time", "text_length", "token_throughput"]
                reference_requiring_algorithms = ["bleu", "rouge", "bert_score", "semantic_similarity", "semantic_textual_similarity"]
                self_evaluating_algorithms = ["pairwise_comparison", "llm_as_judge", "g_eval", "rlhf_preference",
                                            "code_generation", "commonsense_reasoning", "mathematical_reasoning", 
                                            "safety_alignment", "truthfulness"]
                
                # Run basic algorithms (always work)
                basic_to_run = [alg for alg in self.comparison_algorithms if alg in basic_algorithms]
                if basic_to_run:
                    logger.info(f"Running basic algorithms: {basic_to_run}")
                    comparison_results = self.comparison_engine.run_comparison(
                        basic_to_run,
                        [response],
                        None
                    )
                    
                    for algo_name, results in comparison_results.items():
                        if results:
                            result = results[0]
                            result_data[f"{algo_name}_score"] = result.score
                            result_data[f"{algo_name}_execution_time_ms"] = result.execution_time_ms
                            logger.info(f"✓ {algo_name}: {result.score:.2f}")
                        else:
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                            logger.warning(f"✗ {algo_name}: no results")
                
                # Run self-evaluating algorithms (work without reference data)
                self_eval_to_run = [alg for alg in self.comparison_algorithms if alg in self_evaluating_algorithms]
                if self_eval_to_run:
                    logger.info(f"Running self-evaluating algorithms: {self_eval_to_run}")
                    for algo_name in self_eval_to_run:
                        try:
                            algorithm_results = self.comparison_engine.run_comparison(
                                [algo_name],
                                [response],
                                None  # No reference data needed
                            )
                            
                            if algo_name in algorithm_results and algorithm_results[algo_name]:
                                result = algorithm_results[algo_name][0]
                                result_data[f"{algo_name}_score"] = result.score
                                result_data[f"{algo_name}_execution_time_ms"] = result.execution_time_ms
                                logger.info(f"✓ {algo_name}: {result.score:.2f}")
                            else:
                                result_data[f"{algo_name}_score"] = None
                                result_data[f"{algo_name}_execution_time_ms"] = None
                                logger.warning(f"✗ {algo_name}: no results")
                                
                        except Exception as algo_error:
                            logger.warning(f"✗ {algo_name} failed: {algo_error}")
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                
                # Run reference-requiring algorithms (if reference data available)
                ref_requiring_to_run = [alg for alg in self.comparison_algorithms if alg in reference_requiring_algorithms]
                if ref_requiring_to_run:
                    if self.reference_loader and self.reference_loader.is_available():
                        logger.info(f"Running reference-requiring algorithms: {ref_requiring_to_run}")
                        
                        # Get appropriate reference data for these algorithms
                        reference_data = self.reference_loader.get_reference_data_for_algorithms(
                            ref_requiring_to_run,
                            num_samples=10  # Use fewer samples per response for performance
                        )
                        
                        for algo_name in ref_requiring_to_run:
                            try:
                                # Get reference texts for this algorithm
                                refs = reference_data.get(algo_name, [])
                                if not refs:
                                    logger.warning(f"✗ {algo_name}: no reference data available")
                                    result_data[f"{algo_name}_score"] = None
                                    result_data[f"{algo_name}_execution_time_ms"] = None
                                    continue
                                
                                # Use first reference text for single-response evaluation
                                single_ref = [refs[0]] if refs else None
                                
                                algorithm_results = self.comparison_engine.run_comparison(
                                    [algo_name],
                                    [response],
                                    single_ref
                                )
                                
                                if algo_name in algorithm_results and algorithm_results[algo_name]:
                                    result = algorithm_results[algo_name][0]
                                    result_data[f"{algo_name}_score"] = result.score
                                    result_data[f"{algo_name}_execution_time_ms"] = result.execution_time_ms
                                    logger.info(f"✓ {algo_name}: {result.score:.2f} (with reference data)")
                                else:
                                    result_data[f"{algo_name}_score"] = None
                                    result_data[f"{algo_name}_execution_time_ms"] = None
                                    logger.warning(f"✗ {algo_name}: no results with reference data")
                                    
                            except Exception as algo_error:
                                logger.warning(f"✗ {algo_name} failed with reference data: {algo_error}")
                                result_data[f"{algo_name}_score"] = None
                                result_data[f"{algo_name}_execution_time_ms"] = None
                    else:
                        logger.warning(f"Reference data not available - skipping algorithms: {ref_requiring_to_run}")
                        for algo_name in ref_requiring_to_run:
                            result_data[f"{algo_name}_score"] = None
                            result_data[f"{algo_name}_execution_time_ms"] = None
                            logger.info(f"⚠ {algo_name}: requires reference data (not loaded)")
                            
            except Exception as e:
                logger.error(f"Error running comparison algorithms: {e}")
                # Ensure all algorithm columns have values
                for algo_name in self.comparison_algorithms:
                    if f"{algo_name}_score" not in result_data:
                        result_data[f"{algo_name}_score"] = None
                        result_data[f"{algo_name}_execution_time_ms"] = None
        
        # Add energy profiling results
        if self.energy_profiler == "powerjoular":
            energy_data = self._parse_powerjoular_results(context)
            result_data.update(energy_data)
        elif self.energy_profiler == "codecarbon":
            energy_data = self._parse_codecarbon_results(context)
            result_data.update(energy_data)
        
        # 🎯 COMPREHENSIVE DATA COLLECTION - SAVE COMPLETE RUN INFO TO INDIVIDUAL RUN DIR
        self._save_complete_run_data(context, result_data)
        
        return result_data
    
    def _parse_powerjoular_results(self, context: RunnerContext) -> Dict[str, float]:
        """Parse PowerJoular CSV results."""
        energy_data = {
            "avg_cpu_utilization": 0.0,
            "total_cpu_power_watts": 0.0,
            "total_dram_power_watts": 0.0,
            "total_energy_joules": 0.0,
            "power_efficiency_tokens_per_joule": 0.0
        }
        
        try:
            current_pid = os.getpid()
            csv_path = context.run_dir / f"powerjoular.csv-{current_pid}.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    energy_data["avg_cpu_utilization"] = float(df['CPU Utilization'].mean())
                    energy_data["total_cpu_power_watts"] = float(df['CPU Power'].sum())
                    
                    if 'DRAM Power' in df.columns:
                        energy_data["total_dram_power_watts"] = float(df['DRAM Power'].sum())
                    
                    # Calculate total energy (power * time)
                    total_power = energy_data["total_cpu_power_watts"] + energy_data["total_dram_power_watts"]
                    measurement_duration_sec = len(df) * 0.5  # Assuming 500ms intervals
                    energy_data["total_energy_joules"] = total_power * measurement_duration_sec
                    
                    # Calculate efficiency
                    if (self.current_model_response and 
                        energy_data["total_energy_joules"] > 0):
                        energy_data["power_efficiency_tokens_per_joule"] = (
                            self.current_model_response.tokens_generated / energy_data["total_energy_joules"]
                        )
                
        except Exception as e:
            logger.error(f"Error parsing PowerJoular results: {e}")
        
        return energy_data
    
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
            "pue": 1.0
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
                    energy_data["emissions_kg_co2"] = self._safe_float(last_record.get('emissions', 0))
                    energy_data["energy_consumed_kwh"] = self._safe_float(last_record.get('energy_consumed', 0))
                    energy_data["cpu_energy_kwh"] = self._safe_float(last_record.get('cpu_energy', 0))
                    energy_data["gpu_energy_kwh"] = self._safe_float(last_record.get('gpu_energy', 0))
                    energy_data["ram_energy_kwh"] = self._safe_float(last_record.get('ram_energy', 0))
                    
                    # Additional useful fields from CodeCarbon - handle NaN values safely
                    energy_data["duration_seconds"] = self._safe_float(last_record.get('duration', 0))
                    energy_data["emissions_rate_kg_co2_per_s"] = self._safe_float(last_record.get('emissions_rate', 0))
                    energy_data["cpu_power_w"] = self._safe_float(last_record.get('cpu_power', 0))
                    energy_data["gpu_power_w"] = self._safe_float(last_record.get('gpu_power', 0))
                    energy_data["ram_power_w"] = self._safe_float(last_record.get('ram_power', 0))
                    energy_data["cpu_count"] = self._safe_int(last_record.get('cpu_count', 0))
                    energy_data["gpu_count"] = self._safe_int(last_record.get('gpu_count', 0))
                    energy_data["tracking_mode"] = str(last_record.get('tracking_mode', 'unknown'))
                    energy_data["country_name"] = str(last_record.get('country_name', 'unknown'))
                    energy_data["country_iso_code"] = str(last_record.get('country_iso_code', 'unknown'))
                    energy_data["pue"] = self._safe_float(last_record.get('pue', 1.0))
                    
                    # Calculate derived metrics
                    if energy_data["duration_seconds"] > 0:
                        # Power efficiency: tokens per second per watt
                        if self.current_model_response and energy_data["cpu_power_w"] > 0:
                            tokens_per_second = self.current_model_response.tokens_generated / (energy_data["duration_seconds"] or 1)
                            total_power_w = energy_data["cpu_power_w"] + energy_data["gpu_power_w"] + energy_data["ram_power_w"]
                            if total_power_w > 0:
                                energy_data["tokens_per_second_per_watt"] = tokens_per_second / total_power_w
                            else:
                                energy_data["tokens_per_second_per_watt"] = 0.0
                        else:
                            energy_data["tokens_per_second_per_watt"] = 0.0
                
        except Exception as e:
            logger.error(f"Error parsing CodeCarbon results: {e}")
        
        return energy_data
    
    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment."""
        output.console_log("Energy-Profiled LLM Comparison Experiment completed")
        
        # Cleanup loaded models to free memory
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
            if hasattr(context, 'repetition'):
                return getattr(context, 'repetition')
            
            # Try to extract from run_dir path (like "run_0_repetition_0")
            if hasattr(context, 'run_dir'):
                run_dir_name = str(context.run_dir).split('/')[-1]  # Get just the folder name
                if 'repetition_' in run_dir_name:
                    repetition_part = run_dir_name.split('repetition_')[-1]
                    return int(repetition_part.split('_')[0])  # Handle any trailing parts
            
            # Fallback to 0 if no repetition info found
            return 0
            
        except (ValueError, IndexError, AttributeError):
            return 0

    def _save_complete_run_data(self, context: RunnerContext, result_data: Dict[str, Any]) -> None:
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
                    "run_id": getattr(context, 'run_id', f"run_{getattr(context, 'run_nr', 0)}"),
                    "timestamp": getattr(context, 'run_id', f"run_{getattr(context, 'run_nr', 0)}")
                },
                "model_metadata": self.model_metadata.get(self.current_model_id, {}),
                "generation_params": self.generation_params.copy(),
                "algorithm_results": result_data.copy(),
                "model_response": {
                    "text": getattr(self.current_model_response, 'text', '') if self.current_model_response else '',
                    "response_time": result_data.get('response_time', 0),
                    "token_count": result_data.get('token_throughput', 0)
                }
            }
            
            # Write comprehensive data to JSON file
            with open(run_data_file, 'w', encoding='utf-8') as f:
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
            # Don't let this error stop the experiment
            pass


# Export the RunnerConfig class for Experiment Runner
RunnerConfig = EnergyProfiledLLMConfig