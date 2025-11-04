"""Universal Model Loader Module

This module provides functionality to load, manage, and interact with
various Hugging Face models in a memory-efficient way.
"""

import importlib.util
import logging
import gc
import os
import math
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import psutil
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoConfig, pipeline, Pipeline, LlamaTokenizer
)
from transformers.utils.import_utils import is_torch_available  # type: ignore[attr-defined]

try:
    from transformers.models.mamba import MambaForCausalLM  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    MambaForCausalLM = None  # type: ignore[assignment]

try:
    from transformers.models.rwkv import RWKVForCausalLM  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    RWKVForCausalLM = None  # type: ignore[assignment]


try:
    from ...core.data_manager import get_data_manager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelLoadConfig:
    device: str = "auto"
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False
    use_cache: bool = True
    max_memory_per_model_gb: Optional[float] = None
    offload_folder: Optional[str] = None
    low_cpu_mem_usage: bool = True
    device_map: Optional[Union[str, Dict[str, str]]] = None
    enable_cpu_offload: bool = False
    max_cpu_memory_gb: Optional[float] = None
    kv_cache_offload: bool = False
    quantization_mode: Optional[str] = None  # e.g. "4bit" or "8bit"
    bnb_compute_dtype: Optional[str] = None
    bnb_double_quant: bool = True
    bnb_quant_type: str = "nf4"
    enable_flash_attention: bool = False
    compile_model: bool = False

@dataclass 
class LoadedModel:
    """Container for a loaded model with metadata."""
    id: str
    model: Any
    tokenizer: Any
    pipeline_obj: Optional[Pipeline] = None
    metadata: Optional[Any] = None
    device: str = "cpu"
    memory_usage_mb: float = 0.0
    load_time_seconds: float = 0.0
    last_used: float = 0.0
    lock: Optional[threading.Lock] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()
        if self.last_used == 0.0:
            self.last_used = time.time()


class ModelMemoryManager:
    """Manages model memory usage and automatic unloading."""
    
    def __init__(self, max_total_memory_gb: float = 8.0):
        self.max_total_memory_gb = max_total_memory_gb
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.memory_lock = threading.Lock()
    
    def get_current_memory_usage_gb(self) -> float:
        """Get current total memory usage in GB."""
        total_mb = sum(model.memory_usage_mb for model in self.loaded_models.values())
        return total_mb / 1024
    
    def can_load_model(self, estimated_size_gb: float) -> bool:
        """Check if we can load a model of given size."""
        current_usage = self.get_current_memory_usage_gb()
        
        # Check system memory
        memory_info = psutil.virtual_memory()
        available_system_memory = memory_info.available / (1024**3)  # GB
        
        return (
            current_usage + estimated_size_gb <= self.max_total_memory_gb and
            estimated_size_gb <= available_system_memory * 0.8  # Leave 20% buffer
        )
    
    def free_memory_for_model(self, required_gb: float) -> bool:
        """Free memory by unloading least recently used models."""
        with self.memory_lock:
            current_usage = self.get_current_memory_usage_gb()
            
            if current_usage + required_gb <= self.max_total_memory_gb:
                return True
            
            # Sort by last used time (oldest first)
            models_by_usage = sorted(
                self.loaded_models.items(),
                key=lambda x: x[1].last_used
            )
            
            freed_memory = 0.0
            for model_id, loaded_model in models_by_usage:
                if current_usage - freed_memory + required_gb <= self.max_total_memory_gb:
                    break
                
                logger.info(f"Unloading model {model_id} to free memory")
                freed_memory += loaded_model.memory_usage_mb / 1024
                self._unload_model(model_id)
            
            return self.get_current_memory_usage_gb() + required_gb <= self.max_total_memory_gb
    
    def _unload_model(self, model_id: str):
        """Unload a specific model from memory."""
        if model_id in self.loaded_models:
            loaded_model = self.loaded_models[model_id]
            if loaded_model.lock:
                with loaded_model.lock:
                    # Clean up the model
                    if hasattr(loaded_model.model, 'to'):
                        loaded_model.model.to('cpu')
                    del loaded_model.model
                    del loaded_model.tokenizer
                    if loaded_model.pipeline_obj:
                        del loaded_model.pipeline_obj
                
                # Remove from tracking
                del self.loaded_models[model_id]
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

class UniversalModelLoader:
    """Universal loader for Hugging Face models with memory management."""
    
    def __init__(self, config: ModelLoadConfig = ModelLoadConfig(), max_memory_gb: float = 8.0, discovery=None):
        """Initialize the model loader.
        
        Args:
            config: Model loading configuration
            max_memory_gb: Maximum total memory for loaded models
            discovery: Optional discovery service for automatic model reloading
        """
        self.config = config
        self.memory_manager = ModelMemoryManager(max_memory_gb)
        self.device = self._determine_device()
        self.discovery = discovery
        
        # Initialize data manager for organized downloads
        if DATA_MANAGER_AVAILABLE:
            self.data_manager = get_data_manager()
        else:
            self.data_manager = None
            
        logger.info(f"Initialized UniversalModelLoader with device: {self.device}")
        if self.data_manager:
            logger.info(f"Using organized data cache: {self.data_manager.data_root}")
    
    def load_model_by_id(self, model_id: str, force_reload: bool = False) -> Optional[LoadedModel]:
        """Load a model by ID string (convenience method).
        
        Args:
            model_id: HuggingFace model ID
            force_reload: Force reload even if already loaded
            
        Returns:
            LoadedModel object or None if loading failed
        """
        # Create a simple metadata object
        class SimpleMetadata:
            def __init__(self, model_id):
                self.id = model_id
                self.model_id = model_id
                self.size_gb = 2.0  # Default estimate
                self.pipeline_tag = None
                self.task = None
        
        metadata = SimpleMetadata(model_id)
        return self.load_model(metadata, force_reload)
    
    def _determine_device(self) -> str:
        """Determine the best device for model loading."""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
        
    def load_model(self, metadata, force_reload: bool = False) -> Optional[LoadedModel]:
        """Load a model with automatic memory management.
        
        Args:
            metadata: Model metadata with ID and other info
            force_reload: Force reload even if already loaded
            
        Returns:
            LoadedModel object or None if loading failed
        """
        # Resolve model aliases before any operations
        original_model_id = getattr(metadata, 'id', 'unknown')
        resolved_model_id = self._resolve_model_alias(original_model_id)
        
        # Update metadata with resolved ID if different
        if resolved_model_id != original_model_id:
            logger.info(f"Model alias resolved: {original_model_id} → {resolved_model_id}")
            # Update metadata object
            if hasattr(metadata, 'id'):
                metadata.id = resolved_model_id
            if hasattr(metadata, 'model_id'):
                metadata.model_id = resolved_model_id
        
        model_id = resolved_model_id
        
        # Check for both original and resolved model IDs in cache
        cache_keys_to_check = [model_id]
        if resolved_model_id != original_model_id:
            cache_keys_to_check.append(original_model_id)
        
        # Return existing model if already loaded and not forcing reload
        if not force_reload:
            for cache_key in cache_keys_to_check:
                if cache_key in self.memory_manager.loaded_models:
                    loaded_model = self.memory_manager.loaded_models[cache_key]
                    loaded_model.last_used = time.time()
                    logger.debug(f"Using cached model: {cache_key} (requested: {original_model_id})")
                    
                    # Cache under both keys for future lookups
                    if resolved_model_id != original_model_id:
                        self.memory_manager.loaded_models[original_model_id] = loaded_model
                        self.memory_manager.loaded_models[resolved_model_id] = loaded_model
                    
                    return loaded_model
        
        # Check if we have enough memory
        estimated_size = getattr(metadata, 'size_gb', 2.0) or 2.0
        if not self.memory_manager.can_load_model(estimated_size):
            if not self.memory_manager.free_memory_for_model(estimated_size):
                logger.error(f"Cannot load model {model_id}: insufficient memory")
                return None
        
        logger.info(f"Loading model: {model_id}")
        print(f"\n 📥 DOWNLOADING/LOADING MODEL: {model_id}")
        print(f"   🔍 Checking cache and downloading if needed...")
        start_time = time.time()
        
        try:
            # Determine cache directory
            cache_dir = None
            if self.data_manager:
                cache_dir = str(self.data_manager.get_model_cache_dir(model_id))
                print(f"   � Using organized cache: {cache_dir}")
            
            # Load tokenizer with fallback for SentencePiece/Tiktoken issues
            print(f"   📝 Loading tokenizer...")
            tokenizer_kwargs: Dict[str, Any] = {
                'trust_remote_code': self.config.trust_remote_code,
                'use_fast': True if is_torch_available() else False
            }
            if cache_dir:
                tokenizer_kwargs['cache_dir'] = cache_dir  # type: ignore[assignment]
            
            # Add HuggingFace token for gated/private models
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                tokenizer_kwargs['token'] = hf_token  # type: ignore[assignment]
                print(f"   🔐 Using HuggingFace authentication for gated models")
            
            # Try loading tokenizer with fast tokenizer first, fallback to slow if it fails
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
                print(f"   ✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
            except Exception as tokenizer_error:
                error_msg = str(tokenizer_error)
                # Check if this is a conversion error (SentencePiece/Tiktoken)
                is_conversion_error = (
                    "SentencePiece" in error_msg or 
                    "Tiktoken" in error_msg or
                    "Converting" in error_msg
                )
                
                if is_conversion_error:
                    print(f"   ⚠️  Fast tokenizer conversion failed, using legacy slow tokenizer...")
                    # Try with legacy=True to bypass fast tokenizer conversion issues
                    try:
                        # First, try LlamaTokenizer directly for LLaMA-based models
                        if 'llama' in model_id.lower() or 'eagle' in model_id.lower():
                            print(f"   🔧 Trying LlamaTokenizer directly...")
                            tokenizer_kwargs_legacy = {k: v for k, v in tokenizer_kwargs.items() if k != 'use_fast'}
                            tokenizer = LlamaTokenizer.from_pretrained(model_id, **tokenizer_kwargs_legacy)
                            print(f"   ✅ LlamaTokenizer loaded (vocab size: {tokenizer.vocab_size})")
                        else:
                            # For other models, try with legacy parameter
                            tokenizer_kwargs_legacy = tokenizer_kwargs.copy()
                            tokenizer_kwargs_legacy['use_fast'] = False
                            tokenizer_kwargs_legacy['legacy'] = True  # Use legacy loading
                            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs_legacy)
                            print(f"   ✅ Legacy tokenizer loaded (vocab size: {tokenizer.vocab_size})")
                    except Exception as legacy_error:
                        print(f"   ❌ Legacy tokenizer also failed!")
                        print(f"   🔍 Error: {legacy_error}")
                        raise RuntimeError(f"Failed to load tokenizer for {model_id}: {legacy_error}")
                else:
                    # Different error, re-raise it
                    print(f"   ❌ Tokenizer loading failed with non-conversion error")
                    raise tokenizer_error
            
            # Determine model class with fallback
            model_class = self._get_model_class(metadata)
            print(f"   🏗️  Trying model class: {model_class.__name__}")
            
            # Set up loading parameters with advanced configuration
            model_kwargs: Dict[str, Any] = {
                'trust_remote_code': self.config.trust_remote_code,
                'low_cpu_mem_usage': self.config.low_cpu_mem_usage,
                'use_cache': self.config.use_cache
            }
            
            # Add cache directory if available
            if cache_dir:
                model_kwargs['cache_dir'] = cache_dir  # type: ignore[assignment]
            
            # Add HuggingFace token for gated/private models
            if hf_token:
                model_kwargs['token'] = hf_token  # type: ignore[assignment]
            
            # Handle torch_dtype
            if self.config.torch_dtype and self.config.torch_dtype != "auto":
                if hasattr(torch, self.config.torch_dtype):
                    torch_dtype = getattr(torch, self.config.torch_dtype)
                    model_kwargs['torch_dtype'] = torch_dtype
                    print(f"   🎯 Torch dtype: {self.config.torch_dtype}")
                else:
                    logger.warning(f"Unknown torch dtype '{self.config.torch_dtype}', defaulting to auto")
                    print(f"   🎯 Torch dtype: auto (invalid override ignored)")
            else:
                print(f"   🎯 Torch dtype: auto")

            # Device mapping and offloading configuration
            if self.config.device_map:
                model_kwargs['device_map'] = self.config.device_map  # type: ignore[assignment]
                print(f"   �️  Using custom device map")
            elif self.config.enable_cpu_offload:
                model_kwargs['device_map'] = 'auto'
                print(f"   🗺️  Enabling automatic device map for CPU offload")

            if self.config.enable_cpu_offload:
                max_memory: Dict[Any, str] = {}
                if torch.cuda.is_available():
                    for idx in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(idx)
                        total_gb = int(math.floor(props.total_memory / (1024 ** 3)))
                        max_memory[f"cuda:{idx}"] = f"{total_gb}GiB"
                cpu_cap_gb = self.config.max_cpu_memory_gb or max(4, int(psutil.virtual_memory().available / (1024 ** 3) * 0.9))
                max_memory['cpu'] = f"{int(cpu_cap_gb)}GiB"
                model_kwargs['max_memory'] = max_memory  # type: ignore[assignment]
                if self.config.offload_folder:
                    model_kwargs['offload_folder'] = self.config.offload_folder
                    print(f"   🗄️  Offloading weights to: {self.config.offload_folder}")
                if self.config.kv_cache_offload and self.config.offload_folder:
                    model_kwargs['offload_state_dict'] = True
                    print(f"   🧠 KV-cache offload enabled")

            # Quantization support (bitsandbytes)
            if self.config.quantization_mode:
                quant_mode = self.config.quantization_mode.lower()
                try:
                    if importlib.util.find_spec("bitsandbytes") is None:
                        raise ImportError("bitsandbytes not installed")

                    if quant_mode == '8bit':
                        model_kwargs['load_in_8bit'] = True
                        print(f"   🪄 Loading model in 8-bit precision")
                    elif quant_mode == '4bit':
                        model_kwargs['load_in_4bit'] = True
                        compute_dtype_str = self.config.bnb_compute_dtype or 'float16'
                        if hasattr(torch, compute_dtype_str):
                            model_kwargs['bnb_4bit_compute_dtype'] = getattr(torch, compute_dtype_str)
                        else:
                            logger.warning(f"Unknown compute dtype '{compute_dtype_str}', defaulting to float16")
                            model_kwargs['bnb_4bit_compute_dtype'] = torch.float16
                        model_kwargs['bnb_4bit_use_double_quant'] = self.config.bnb_double_quant
                        model_kwargs['bnb_4bit_quant_type'] = self.config.bnb_quant_type
                        print(f"   🪄 Loading model in 4-bit precision (quant type: {self.config.bnb_quant_type})")
                    else:
                        logger.warning(f"Unsupported quantization mode '{self.config.quantization_mode}'")
                except ImportError:
                    logger.warning("bitsandbytes not installed; skipping quantization mode")
                    print(f"   ⚠️  bitsandbytes not installed, skipping quantization mode")

            if self.config.enable_flash_attention:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                print(f"   ⚡ Flash Attention enabled")

            # Load model with extended fallback strategy
            model = None
            print(f"   � Loading model weights (this may take a while for first download)...")
            load_errors: List[Tuple[str, Exception]] = []

            for candidate_class in self._iterate_candidate_model_classes(model_class):
                class_name = candidate_class.__name__
                try:
                    model = candidate_class.from_pretrained(model_id, **model_kwargs)
                    print(f"   ✅ Model weights loaded successfully with {class_name}!")
                    model_class = candidate_class
                    break
                except Exception as candidate_error:
                    load_errors.append((class_name, candidate_error))
                    print(f"   ⚠️  Failed with {class_name}: {candidate_error}")
                    continue

            if model is None:
                print(f"   ❌ All model class attempts failed")
                for class_name, err in load_errors:
                    print(f"      - {class_name}: {err}")
                raise RuntimeError(f"Failed to load {model_id} with available model classes")
            
            # Move to device
            print(f"   🚚 Moving model to device: {self.device}")
            model = model.to(self.device)  # type: ignore[arg-type]

            if self.config.compile_model and hasattr(torch, "compile"):
                try:
                    print(f"   🧩 Compiling model with torch.compile for faster inference")
                    model = torch.compile(model)  # type: ignore[attr-defined]
                except Exception as compile_error:
                    logger.warning(f"torch.compile failed, continuing with eager mode: {compile_error}")
                    print(f"   ⚠️  torch.compile failed, continuing without compilation")
            
            # Set up pipeline if possible
            pipeline_obj = None
            try:
                print(f"   🔗 Setting up inference pipeline...")
                pipeline_task = self._get_pipeline_task(metadata)
                if pipeline_task:
                    pipeline_obj = pipeline(  # type: ignore[call-overload]
                        pipeline_task,  # type: ignore[arg-type]
                        model=model,  # type: ignore[arg-type]
                        tokenizer=tokenizer,
                        device=0 if self.device == "cuda" else -1,
                        trust_remote_code=self.config.trust_remote_code
                    )
                    print(f"   ✅ Pipeline created for task: {pipeline_task}")
            except Exception as e:
                logger.warning(f"Could not create pipeline for {model_id}: {e}")
                print(f"   ⚠️  Pipeline creation failed, will use model directly")
            
            # Calculate memory usage
            print(f"   📊 Calculating memory usage...")
            memory_usage_mb = self._calculate_memory_usage(model)
            load_time = time.time() - start_time
            print(f"   ✅ Model loaded in {load_time:.2f}s, using {memory_usage_mb:.1f}MB")
            
            # Create loaded model object
            loaded_model = LoadedModel(
                id=model_id,
                model=model,
                tokenizer=tokenizer,
                pipeline_obj=pipeline_obj,
                metadata=metadata,
                device=self.device,
                memory_usage_mb=memory_usage_mb,
                load_time_seconds=load_time,
                last_used=time.time(),
                lock=threading.Lock()
            )
            
            # Register with memory manager
            self.memory_manager.loaded_models[model_id] = loaded_model
            
            # Also cache under original alias if different (for future lookups)
            if resolved_model_id != original_model_id:
                self.memory_manager.loaded_models[original_model_id] = loaded_model
                logger.debug(f"Model cached under both IDs: {original_model_id} and {model_id}")
            
            logger.info(f"Successfully loaded model {model_id}")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            print(f"   ❌ Failed to load model: {e}")
            return None
    
    def _resolve_model_alias(self, model_id: str) -> str:
        """Resolve model ID aliases to actual model IDs.
        
        Args:
            model_id: Original model ID that might be an alias
            
        Returns:
            Actual model ID that should be used for loading
        """
        # Common model aliases mapping
        model_aliases = {
            # GPT-2 family
            'gpt2': 'openai-community/gpt2',
            'gpt2-medium': 'openai-community/gpt2-medium',
            'gpt2-large': 'openai-community/gpt2-large',
            'gpt2-xl': 'openai-community/gpt2-xl',
            
            # DistilGPT2
            'distilgpt2': 'distilbert/distilgpt2',
            
            # Conversational models
            'blenderbot': 'facebook/blenderbot-400M-distill',
            'blenderbot-small': 'facebook/blenderbot-90M',
            'dialogpt': 'microsoft/DialoGPT-medium',
            'dialogpt-small': 'microsoft/DialoGPT-small',
            'dialogpt-large': 'microsoft/DialoGPT-large',
            
            # BERT family
            'bert-base': 'bert-base-uncased',
            'bert-large': 'bert-large-uncased',
            'distilbert': 'distilbert-base-uncased',
            
            # T5 family
            't5-small': 'google-t5/t5-small',
            't5-base': 'google-t5/t5-base',
            't5-large': 'google-t5/t5-large',
        }
        
        # Check if it's a known alias
        resolved_id = model_aliases.get(model_id, model_id)
        
        if resolved_id != model_id:
            logger.debug(f"Resolved model alias: {model_id} → {resolved_id}")
        
        return resolved_id
        return model_id
    
    def generate_text(self, model_id: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using a loaded model with auto-reload capability.
        
        Args:
            model_id: ID of the model to use (e.g., 'gpt2', 'distilbert/distilgpt2')
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or None if failed
        """
        # Check if model is loaded, if not try to reload it automatically
        # Also check for common aliases (gpt2 -> openai-community/gpt2)
        actual_model_id = self._resolve_model_alias(model_id)
        
        # Try to find the model using either the original ID or the resolved alias
        loaded_model = None
        if model_id in self.memory_manager.loaded_models:
            loaded_model = self.memory_manager.loaded_models[model_id]
        elif actual_model_id in self.memory_manager.loaded_models:
            loaded_model = self.memory_manager.loaded_models[actual_model_id]
            # Cache the alias for future use
            self.memory_manager.loaded_models[model_id] = loaded_model
        
        if loaded_model is None:
            logger.debug(f"Model {model_id} not loaded, attempting to reload...")
            
            # Try to reload the model using discovery service
            if self.discovery:
                try:
                    model_metadata = self.discovery.get_model_info(actual_model_id) or self.discovery.get_model_info(model_id)
                    if model_metadata:
                        logger.info(f"Reloading model {model_id} automatically")
                        # Load the model
                        loaded_model = self.load_model(model_metadata)
                        if loaded_model:
                            # Store the loaded model under both the original request ID and the actual model ID
                            self.memory_manager.loaded_models[model_id] = loaded_model
                            if actual_model_id != model_id:
                                self.memory_manager.loaded_models[actual_model_id] = loaded_model
                                logger.debug(f"Model aliasing: {model_id} -> {actual_model_id}")
                        else:
                            logger.error(f"Failed to reload model {model_id}")
                            return None
                    else:
                        logger.error(f"Could not find metadata for model {model_id}")
                        return None
                except Exception as e:
                    logger.error(f"Error reloading model {model_id}: {e}")
                    return None
            else:
                logger.error(f"Model {model_id} not loaded and no discovery service available for reloading")
                return None

        print(f"\n🤖 GENERATING RESPONSE...")
        print(f"   📝 Prompt: '{prompt}'")
        print(f"   ⚙️  Generation parameters: {kwargs}")
        
        # Use the loaded_model we found or loaded above
        if loaded_model is None:
            logger.error(f"Model {model_id} is not available")
            return None
            
        loaded_model.last_used = time.time()
        
        generation_start = time.time()
        
        if not loaded_model.lock:
            logger.error(f"Model {model_id} has no lock")
            return None
        
        with loaded_model.lock:
            try:
                if loaded_model.pipeline_obj:
                    # Use pipeline if available
                    print(f"   🔧 Using inference pipeline")
                    pipeline_kwargs = kwargs.copy()
                    
                    # Handle parameter conflicts
                    if 'max_length' in pipeline_kwargs:
                        pipeline_kwargs.pop('max_new_tokens', None)
                    
                    # Remove non-generation parameters that shouldn't be passed to the model
                    non_generation_params = ['cache_dir', 'device', 'trust_remote_code', 'torch_dtype', 'low_cpu_mem_usage']
                    for param in non_generation_params:
                        pipeline_kwargs.pop(param, None)
                    
                    # Fix model-specific compatibility issues
                    print(f"   🔧 Checking model compatibility for: {loaded_model.id}")
                    if 'blenderbot' in loaded_model.id.lower() or 'facebook/blenderbot' in loaded_model.id.lower():
                        print(f"   🤖 Applying BlenderBot-specific fixes")
                        pipeline_kwargs['use_cache'] = False  # Disable cache to avoid index errors
                        pipeline_kwargs['min_length'] = 1    # Set reasonable min_length
                    
                    # Handle max_length vs max_new_tokens parameter conflicts
                    if 'max_length' in pipeline_kwargs and 'max_new_tokens' in pipeline_kwargs:
                        # Prefer max_new_tokens for better control
                        pipeline_kwargs.pop('max_length', None)
                    
                    # Fix model-specific parameters
                    if 'pad_token_id' in pipeline_kwargs:
                        # Don't use hardcoded pad_token_id for seq2seq models
                        if any(seq2seq_name in loaded_model.id.lower() for seq2seq_name in ['blenderbot', 't5', 'bart', 'pegasus']):
                            print(f"   🔧 Removing hardcoded pad_token_id for seq2seq model")
                            pipeline_kwargs.pop('pad_token_id', None)
                        # Use model's tokenizer's pad_token_id instead of hardcoded values
                        elif hasattr(loaded_model, 'tokenizer') and loaded_model.tokenizer:
                            if hasattr(loaded_model.tokenizer, 'pad_token_id') and loaded_model.tokenizer.pad_token_id is not None:
                                pipeline_kwargs['pad_token_id'] = loaded_model.tokenizer.pad_token_id
                            elif hasattr(loaded_model.tokenizer, 'eos_token_id') and loaded_model.tokenizer.eos_token_id is not None:
                                pipeline_kwargs['pad_token_id'] = loaded_model.tokenizer.eos_token_id
                            else:
                                # Remove pad_token_id if tokenizer doesn't have one
                                pipeline_kwargs.pop('pad_token_id', None)
                        else:
                            # Remove pad_token_id if no tokenizer available
                            pipeline_kwargs.pop('pad_token_id', None)
                    
                    print(f"   ⏳ Running inference...")
                    try:
                        result = loaded_model.pipeline_obj(prompt, **pipeline_kwargs)
                        print(f"   🔍 Pipeline result type: {type(result)}")
                        
                        # Handle different pipeline return formats
                        if isinstance(result, list):
                            if len(result) > 0:
                                print(f"   🔍 List result with {len(result)} items, first item type: {type(result[0])}")
                                # For causal LM (text-generation): [{'generated_text': 'full text including prompt'}]
                                # For seq2seq (text2text-generation): [{'generated_text': 'generated text only'}]
                                first_result = result[0]
                                if isinstance(first_result, dict):
                                    print(f"   🔍 Dict keys: {list(first_result.keys())}")
                                    response = first_result.get('generated_text', '')
                                else:
                                    response = str(first_result)
                            else:
                                print(f"   ⚠️ Empty result list from pipeline")
                                response = ""
                        elif isinstance(result, str):
                            response = result
                        elif hasattr(result, 'generated_text'):
                            response = result.generated_text  # type: ignore[union-attr]
                        else:
                            print(f"   ⚠️ Unexpected result format: {type(result)}, content: {result}")
                            response = str(result)
                    except Exception as e:
                        print(f"   ❌ Pipeline execution failed: {e}")
                        print(f"   🔍 Pipeline kwargs: {pipeline_kwargs}")
                        raise
                        
                else:
                    # Use model and tokenizer directly
                    print(f"   🔧 Using model and tokenizer directly")
                    tokenizer = loaded_model.tokenizer
                    model = loaded_model.model
                    
                    # Tokenize input
                    inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                    
                    # Generate
                    generation_kwargs = kwargs.copy()
                    generation_kwargs.pop('max_new_tokens', None)
                    
                    print(f"   ⏳ Running inference...")
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            pad_token_id=tokenizer.eos_token_id,
                            **generation_kwargs
                        )
                    
                    # Decode output
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                generation_time = time.time() - generation_start
                print(f"   ✅ Response generated in {generation_time:.2f}s")
                print(f"   📄 Response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
                
                return response
                
            except Exception as e:
                logger.error(f"Generation failed for model {model_id}: {e}")
                print(f"   ❌ Generation failed: {e}")
                return None
    
    def unload_model(self, model_id: str) -> bool:
        """Manually unload a specific model."""
        try:
            self.memory_manager._unload_model(model_id)
            logger.info(f"Unloaded model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def get_loaded_models(self) -> Dict[str, LoadedModel]:
        """Get dictionary of currently loaded models."""
        return self.memory_manager.loaded_models.copy()
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics for all loaded models.
        
        Returns:
            Dict with format:
            {
                "loaded_models": {
                    "model_id": {
                        "memory_mb": float,
                        "memory_gb": float,
                        "last_used": str
                    }
                },
                "total_memory_mb": float,
                "total_memory_gb": float,
                "available_memory_gb": float
            }
        """
        loaded_models_stats = {}
        total_memory_mb = 0.0
        
        for model_id, loaded_model in self.memory_manager.loaded_models.items():
            try:
                # Calculate memory usage for this model
                if hasattr(loaded_model, 'model') and loaded_model.model:
                    memory_mb = self._calculate_memory_usage(loaded_model.model)
                elif hasattr(loaded_model, 'pipeline_obj') and loaded_model.pipeline_obj:  # type: ignore[attr-defined]
                    memory_mb = self._calculate_memory_usage(loaded_model.pipeline_obj.model)  # type: ignore[attr-defined]
                else:
                    memory_mb = 0.0
                
                loaded_models_stats[model_id] = {
                    "memory_mb": memory_mb,
                    "memory_gb": memory_mb / 1024,
                    "last_used": str(loaded_model.last_used) if hasattr(loaded_model, 'last_used') else "unknown"
                }
                total_memory_mb += memory_mb
                
            except Exception as e:
                logger.warning(f"Failed to calculate memory for model {model_id}: {e}")
                loaded_models_stats[model_id] = {
                    "memory_mb": 0.0,
                    "memory_gb": 0.0,
                    "last_used": "unknown"
                }
        
        return {
            "loaded_models": loaded_models_stats,
            "total_memory_mb": total_memory_mb,
            "total_memory_gb": total_memory_mb / 1024,
            "available_memory_gb": max(0, self.memory_manager.max_total_memory_gb - (total_memory_mb / 1024))
        }
    
    def _get_model_class(self, metadata):
        """Determine the appropriate model class based on metadata with comprehensive detection."""
        task = getattr(metadata, 'pipeline_tag', None) or getattr(metadata, 'task', None)
        model_id = getattr(metadata, 'model_id', None) or getattr(metadata, 'id', None)
        
        # Comprehensive model ID patterns for seq2seq models
        seq2seq_patterns = [
            # T5 family
            't5', 'flan-t5', 'ul2', 'mt5', 'byt5', 'codet5',
            # BART family  
            'bart', 'mbart', 'plbart',
            # Pegasus family
            'pegasus', 'bigbird_pegasus',
            # Marian translation models
            'marian', 'opus-mt',
            # Other encoder-decoder models
            'led', 'longformer-encoder-decoder', 'blenderbot',
            'prophetnet', 'fsmt', 'encoder-decoder', 'godel', 'seq2seq'
        ]
        
        # Comprehensive model ID patterns for causal LM models
        causal_patterns = [
            # GPT family
            'gpt', 'gpt2', 'gpt-2', 'distilgpt', 'dialogpt', 'gpt-neo', 'gpt-j', 'gpt4all',
            # LLaMA family (Meta AI)
            'llama', 'llama-2', 'llama2', 'code-llama', 'codellama', 'alpaca', 'vicuna', 'wizard', 'orca',
            # Mistral AI models
            'mistral', 'mixtral',
            # Google models
            'gemma', 'gemma-7b', 'gemma-2b',
            # Alibaba Qwen models  
            'qwen', 'qwen2', 'qwen-7b', 'qwen2-7b',
            # Microsoft models
            'phi', 'phi-2', 'phi-3', 'phi3',
            # Other popular models
            'bloom', 'opt', 'codegen', 'starcoder', 'santacoder',
            'falcon', 'mpt', 'dolly', 'stablelm', 'redpajama',
            # 01-ai models
            'yi', 'yi-6b', 'yi-34b',
            # Code-specific models
            'incoder', 'codet5p',
            # Chat/instruct models
            'chat', 'instruct', 'tulu', 'oasst'
        ]

        state_space_patterns = [
            'mamba', 'state-spaces', 'rwkv'
        ]
        
        # Check model ID patterns first (most reliable)
        if model_id:
            model_lower = model_id.lower()
            
            # Check for seq2seq patterns
            for pattern in seq2seq_patterns:
                if pattern in model_lower:
                    print(f"   🔍 Detected seq2seq model (pattern: {pattern})")
                    return AutoModelForSeq2SeqLM
            
            # Check for state-space models (Mamba/RWKV)
            for pattern in state_space_patterns:
                if pattern in model_lower:
                    if 'mamba' in pattern and MambaForCausalLM is not None:
                        print(f"   🔍 Detected Mamba model (pattern: {pattern})")
                        return MambaForCausalLM  # type: ignore[return-value]
                    if 'rwkv' in pattern and RWKVForCausalLM is not None:
                        print(f"   🔍 Detected RWKV model (pattern: {pattern})")
                        return RWKVForCausalLM  # type: ignore[return-value]
                    print(f"   🔍 Detected state-space model, defaulting to causal LM loader")
                    return AutoModelForCausalLM

            # Check for causal LM patterns  
            for pattern in causal_patterns:
                if pattern in model_lower:
                    print(f"   🔍 Detected causal LM model (pattern: {pattern})")
                    return AutoModelForCausalLM
        
        # Check pipeline task
        if task == "text2text-generation":
            print("   🔍 Detected seq2seq model (task: text2text-generation)")
            return AutoModelForSeq2SeqLM
        elif task in ["text-generation", "conversational"]:
            print("   🔍 Detected causal LM model (task: text-generation)")
            return AutoModelForCausalLM
        
        # Try to detect from model configuration
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id or "", trust_remote_code=False)
            
            # Check architecture names
            architectures = getattr(config, 'architectures', [])
            if architectures:
                arch_name = architectures[0] if isinstance(architectures, list) else str(architectures)
                arch_lower = arch_name.lower()
                
                # Seq2seq architectures
                seq2seq_archs = [
                    't5', 'bart', 'pegasus', 'mbart', 'plbart', 'bigbird_pegasus',
                    'led', 'marian', 'blenderbot', 'prophetnet', 'fsmt'
                ]
                
                # Causal LM architectures  
                causal_archs = [
                    'llama', 'mistral', 'gemma', 'qwen', 'gpt', 'gptneo', 'gptj', 
                    'opt', 'bloom', 'falcon', 'phi', 'yi'
                ]

                state_space_archs = ['mamba', 'rwkv']
                
                for arch in seq2seq_archs:
                    if arch in arch_lower:
                        print(f"   🔍 Detected seq2seq model (architecture: {arch_name})")
                        return AutoModelForSeq2SeqLM
                
                # Check causal LM architectures
                for arch in causal_archs:
                    if arch in arch_lower:
                        print(f"   🔍 Detected causal LM model (architecture: {arch_name})")
                        return AutoModelForCausalLM

                for arch in state_space_archs:
                    if arch in arch_lower:
                        if arch == 'mamba' and MambaForCausalLM is not None:
                            print(f"   🔍 Detected Mamba architecture: {arch_name}")
                            return MambaForCausalLM  # type: ignore[return-value]
                        if arch == 'rwkv' and RWKVForCausalLM is not None:
                            print(f"   🔍 Detected RWKV architecture: {arch_name}")
                            return RWKVForCausalLM  # type: ignore[return-value]
                        print(f"   🔍 Detected state-space architecture, defaulting to causal LM")
                        return AutoModelForCausalLM
            
            # Check model type
            model_type = getattr(config, 'model_type', '')
            if model_type:
                seq2seq_types = [
                    't5', 'mt5', 'ul2', 'bart', 'mbart', 'pegasus', 'bigbird_pegasus',
                    'led', 'marian', 'blenderbot', 'prophetnet'
                ]
                
                causal_types = [
                    'llama', 'mistral', 'gemma', 'qwen', 'qwen2', 'gpt2', 'gpt_neo', 'gptj',
                    'opt', 'bloom', 'falcon', 'RefinedWebModel', 'phi', 'phi3', 'yi'  
                ]

                state_space_types = ['mamba', 'rwkv']
                
                if model_type in seq2seq_types:
                    print(f"   🔍 Detected seq2seq model (model_type: {model_type})")
                    return AutoModelForSeq2SeqLM
                elif model_type in causal_types:
                    print(f"   🔍 Detected causal LM model (model_type: {model_type})")
                    return AutoModelForCausalLM
                elif model_type in state_space_types:
                    if model_type == 'mamba' and MambaForCausalLM is not None:
                        print(f"   🔍 Detected Mamba model (model_type: {model_type})")
                        return MambaForCausalLM  # type: ignore[return-value]
                    if model_type == 'rwkv' and RWKVForCausalLM is not None:
                        print(f"   🔍 Detected RWKV model (model_type: {model_type})")
                        return RWKVForCausalLM  # type: ignore[return-value]
                    print(f"   🔍 Detected state-space model (model_type: {model_type}), defaulting to causal LM")
                    return AutoModelForCausalLM
                else:
                    # Unknown model type - try seq2seq first as it's more specific
                    print(f"   🔍 Unknown model type '{model_type}' - trying seq2seq first")
                    return AutoModelForSeq2SeqLM
                    
        except Exception as e:
            print(f"   ⚠️  Could not load config for detection: {e}")
        
        # Smart default: Try seq2seq first for unknown models
        # (Most failures happen when trying causal LM on seq2seq models)
        print("   🔍 Unknown model type - will try both classes with fallback")
        return AutoModelForSeq2SeqLM

    def _iterate_candidate_model_classes(self, primary_class):
        """Yield primary and fallback model classes without duplicates."""
        seen = set()
        for candidate in [primary_class, *self._get_fallback_model_classes(primary_class)]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate

    def _get_fallback_model_classes(self, primary_class):
        """Return an ordered list of fallback model classes based on the primary selection."""
        fallbacks: List[Any] = []

        # Prefer matching architecture-specific classes first
        if primary_class not in {AutoModelForCausalLM, AutoModelForSeq2SeqLM}:
            fallbacks.extend([AutoModelForCausalLM, AutoModelForSeq2SeqLM])
        elif primary_class == AutoModelForSeq2SeqLM:
            fallbacks.append(AutoModelForCausalLM)
        else:
            fallbacks.append(AutoModelForSeq2SeqLM)

        # Always provide a generic auto loader as a last resort
        if primary_class is not AutoModelForCausalLM:
            fallbacks.append(AutoModelForCausalLM)
        if primary_class is not AutoModelForSeq2SeqLM:
            fallbacks.append(AutoModelForSeq2SeqLM)

        # Remove duplicates while preserving order
        ordered_unique: List[Any] = []
        seen = set()
        for fb in fallbacks:
            if fb not in seen:
                ordered_unique.append(fb)
                seen.add(fb)
        return ordered_unique
    
    def _get_pipeline_task(self, metadata) -> Optional[str]:
        """Get the pipeline task for the model with comprehensive detection."""
        task = getattr(metadata, 'pipeline_tag', None) or getattr(metadata, 'task', None)
        model_id = getattr(metadata, 'model_id', None) or getattr(metadata, 'id', None)
        
        # Comprehensive model patterns for task detection
        seq2seq_model_patterns = [
            't5', 'flan-t5', 'ul2', 'mt5', 'byt5', 'codet5',
            'bart', 'mbart', 'plbart', 'pegasus', 'bigbird_pegasus',
            'led', 'marian', 'opus-mt', 'blenderbot', 'prophetnet', 'fsmt',
            'godel', 'seq2seq'  # Added GODEL and explicit seq2seq pattern
        ]
        
        # Check model ID patterns
        if model_id:
            model_lower = model_id.lower()
            for pattern in seq2seq_model_patterns:
                if pattern in model_lower:
                    return "text2text-generation"
        
        # Task mapping
        task_mapping = {
            "text-generation": "text-generation",
            "text2text-generation": "text2text-generation", 
            "conversational": "conversational",
            "translation": "text2text-generation",
            "summarization": "text2text-generation"
        }
        
        mapped_task = task_mapping.get(task) if task else None  # type: ignore[arg-type]
        if mapped_task:
            return mapped_task
        
        # Default based on model patterns
        if model_id:
            model_lower = model_id.lower()
            for pattern in seq2seq_model_patterns:
                if pattern in model_lower:
                    return "text2text-generation"
        
        # Conservative default
        return "text-generation"
    
    def _calculate_memory_usage(self, model) -> float:
        """Calculate model memory usage in MB."""
        try:
            # Try to get actual GPU memory if using CUDA
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                if allocated > 0:
                    return allocated
            
            # Estimate based on parameter and buffer sizes
            total_bytes = 0
            for param in model.parameters():
                total_bytes += param.numel() * param.element_size()
            for buffer in model.buffers():
                total_bytes += buffer.numel() * buffer.element_size()

            memory_mb = total_bytes / (1024 ** 2)
            if memory_mb == 0:
                # Fallback estimation
                return 500.0  # 500MB default
            return memory_mb
            
        except Exception:
            # Fallback estimation
            return 500.0  # 500MB default
