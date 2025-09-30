"""
Hugging Face Model Discovery Module

This module provides functionality to search, filter, and discover models
from the Hugging Face Hub for LLM comparisons.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import os
import requests
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import HfHubHTTPError

# Try to import newer API features, fallback if not available
try:
    from huggingface_hub import ModelFilter, ModelSearchArguments
    HAS_ADVANCED_SEARCH = True
except ImportError:
    HAS_ADVANCED_SEARCH = False
    ModelFilter = None
    ModelSearchArguments = None
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


@dataclass
class ModelSearchCriteria:
    """Criteria for searching models on Hugging Face Hub."""
    task: Optional[str] = None
    language: Optional[str] = None
    library: Optional[str] = None
    min_downloads: Optional[int] = None
    max_size_gb: Optional[float] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    query: Optional[str] = None
    license: Optional[str] = None
    sort_by: str = "downloads"  # downloads, created_at, modified_at
    limit: int = 50


@dataclass
class ModelMetadata:
    """Structured metadata for a discovered model."""
    id: str
    author: str
    task: Optional[str]
    library: Optional[str]
    language: Optional[str]
    license: Optional[str]
    downloads: int
    likes: int
    tags: List[str]
    size_gb: Optional[float]
    created_at: Optional[str]
    last_modified: Optional[str]
    description: Optional[str]
    pipeline_tag: Optional[str]
    model_card_available: bool
    compatible: bool = True
    compatibility_notes: Optional[str] = None


class HuggingFaceModelDiscovery:
    """Service for discovering and filtering models from Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the discovery service.
        
        Args:
            token: HuggingFace API token. If None, will check HF_TOKEN env var.
                  Falls back to public access for some operations.
        """
        # Get token from parameter, environment, or None for public access
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        
        try:
            self.api = HfApi(token=self.token)
            # Test API access
            self.api.whoami() if self.token else None
            logger.info(f"HuggingFace API initialized {'with authentication' if self.token else 'with public access'}")
        except Exception as e:
            logger.warning(f"HuggingFace API initialization warning: {e}")
            logger.info("Continuing with limited access - some features may be unavailable")
            self.api = HfApi()  # Fallback to public access
            
        self._model_cache = {}
        
        # Define compatible tasks and libraries
        self.text_generation_tasks = {
            'text-generation', 'conversational', 'text2text-generation',
            'causal-lm', 'seq2seq-lm'
        }
        
        self.compatible_libraries = {
            'transformers', 'pytorch', 'torch', 'tensorflow', 'tf', 'flax', 'jax'
        }
        
        # Compatible libraries for text generation
        self.compatible_libraries = {
            "transformers", "pytorch", "tensorflow", "jax", "onnx", 
            "safetensors", "diffusers", "sentence-transformers"
        }
        
        # Common text generation tasks
        self.text_generation_tasks = {
            "text-generation", "text2text-generation", "conversational",
            "question-answering", "summarization", "translation"
        }
    
    def search_models(self, criteria: ModelSearchCriteria) -> List[ModelMetadata]:
        """Search for models based on criteria."""
        try:
            # Use simplified search approach compatible with all API versions
            kwargs = {
                'sort': criteria.sort_by,
                'direction': -1,  # Descending
                'limit': criteria.limit,
                'full': True
            }
            
            # Add task filter if supported and specified
            if criteria.task and HAS_ADVANCED_SEARCH:
                try:
                    kwargs['filter'] = ModelFilter(task=criteria.task)
                except:
                    # Fallback to basic search
                    pass
                    
            # Perform the search
            models = list(self.api.list_models(**kwargs))
            
            # Convert to our metadata format and apply additional filters
            results = []
            for model in models:
                try:
                    metadata = self._model_to_metadata(model)
                    if self._meets_criteria(metadata, criteria):
                        results.append(metadata)
                except Exception as e:
                    logger.warning(f"Error processing model {getattr(model, 'modelId', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Found {len(results)} models matching criteria")
            return results
            
        except HfHubHTTPError as e:
            if "401" in str(e):
                logger.error("Authentication error: Invalid or missing HuggingFace token")
                logger.info("Set HF_TOKEN environment variable or provide token to enable full access")
            else:
                logger.error(f"HuggingFace API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            logger.info("Consider checking your internet connection or HuggingFace token")
            return []
    
    def _model_to_metadata(self, model_info: ModelInfo) -> ModelMetadata:
        """Convert HuggingFace ModelInfo to our ModelMetadata format."""
        
        # Extract size information if available
        size_gb = None
        
        # Try to get size from SafeTensors parameter information first
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            try:
                total_params = getattr(model_info.safetensors, 'total', 0)
                if total_params > 0:
                    # Estimate size: roughly 4 bytes per parameter for float32
                    size_bytes = total_params * 4
                    size_gb = size_bytes / (1024**3)
            except Exception:
                pass
        
        # Fallback: try siblings size information
        if size_gb is None and hasattr(model_info, 'siblings') and model_info.siblings:
            total_size = sum(getattr(sibling, 'size', 0) or 0 for sibling in model_info.siblings)
            size_gb = total_size / (1024**3) if total_size > 0 else None
        
        # If still no size info available, estimate based on model name patterns with better estimates
        if size_gb is None:
            model_name = getattr(model_info, 'modelId', getattr(model_info, 'id', '')).lower()
            if 'gpt2' in model_name and 'medium' not in model_name and 'large' not in model_name and 'xl' not in model_name:
                size_gb = 0.5  # Standard GPT2 is ~500MB (117M params)
            elif 'gpt2' in model_name and 'medium' in model_name:
                size_gb = 1.4  # GPT2-medium is ~1.4GB (345M params)
            elif 'gpt2' in model_name and 'large' in model_name:
                size_gb = 3.2  # GPT2-large is ~3.2GB (774M params)
            elif 'gpt2' in model_name and 'xl' in model_name:
                size_gb = 6.4  # GPT2-xl is ~6.4GB (1558M params)
            elif 'distilgpt2' in model_name:
                size_gb = 0.3  # DistilGPT2 is ~300MB (82M params)
            elif 'distil' in model_name:
                size_gb = 0.3  # Other distil models are typically small
            elif 'dialog' in model_name and 'medium' in model_name:
                size_gb = 1.4  # DialoGPT-medium similar to GPT2-medium
            elif 'dialog' in model_name and 'large' in model_name:
                size_gb = 3.2  # DialoGPT-large similar to GPT2-large
            elif 'dialog' in model_name:
                size_gb = 0.5  # DialoGPT-small
            elif 'small' in model_name:
                size_gb = 0.3
            elif 'base' in model_name:
                size_gb = 0.5
            elif 'large' in model_name:
                size_gb = 1.5
            else:
                size_gb = 0.5  # Default estimate
        
        # Extract task information
        task = getattr(model_info, 'pipeline_tag', None)
        
        # Extract language from tags
        language = None
        if hasattr(model_info, 'tags') and model_info.tags:
            for tag in model_info.tags:
                if tag.startswith('lang:') or tag in ['en', 'multilingual', 'english']:
                    language = tag.replace('lang:', '')
                    break
        
        # Extract library information
        library = None
        if hasattr(model_info, 'tags') and model_info.tags:
            for tag in model_info.tags:
                if tag in ['pytorch', 'tensorflow', 'transformers', 'flax']:
                    library = tag
                    break
        
        # Check compatibility
        compatible, compatibility_notes = self._check_compatibility(model_info)
        
        return ModelMetadata(
            id=getattr(model_info, 'modelId', getattr(model_info, 'id', 'unknown')),
            author=getattr(model_info, 'author', 'unknown'),
            task=task,
            library=library,
            language=language,
            license=getattr(model_info, 'license', None),
            downloads=getattr(model_info, 'downloads', 0) or 0,
            likes=getattr(model_info, 'likes', 0) or 0,
            tags=list(getattr(model_info, 'tags', [])) if hasattr(model_info, 'tags') else [],
            size_gb=size_gb,
            created_at=str(getattr(model_info, 'created_at', '')),
            last_modified=str(getattr(model_info, 'last_modified', '')),
            description=getattr(model_info, 'description', ''),
            pipeline_tag=task,  # Use task as pipeline_tag
            model_card_available=hasattr(model_info, 'card_data') and model_info.card_data is not None,
            compatible=compatible,
            compatibility_notes=compatibility_notes
        )
    
    def get_model_info(self, model_id: str, use_cache: bool = True) -> Optional[ModelMetadata]:
        """Get detailed information about a specific model.
        
        Args:
            model_id: The model ID (e.g., 'microsoft/DialoGPT-medium')
            
        Returns:
            ModelMetadata object or None if not found
        """
        try:
            model_info = self.api.model_info(model_id)
            metadata = self._convert_to_metadata(model_info)
            
            # Check compatibility more thoroughly for individual models
            metadata.compatible, metadata.compatibility_notes = self._check_compatibility(model_info)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def search_by_query(self, query: str, limit: int = 20) -> List[ModelMetadata]:
        """Simple text search across models.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching models
        """
        criteria = ModelSearchCriteria(query=query, limit=limit)
        return self.search_models(criteria)
    
    def get_popular_models(self, task: str = "text-generation", limit: int = 10) -> List[ModelMetadata]:
        """Get most popular models for a given task.
        
        Args:
            task: Task type (e.g., 'text-generation')
            limit: Maximum number of results
            
        Returns:
            List of popular models
        """
        criteria = ModelSearchCriteria(
            task=task,
            sort_by="downloads",
            limit=limit,
            min_downloads=1000  # Only include models with substantial downloads
        )
        return self.search_models(criteria)
    
    def _convert_to_metadata(self, model_info: ModelInfo) -> ModelMetadata:
        """Convert HF ModelInfo to our ModelMetadata format."""
        
        # Try to get size from SafeTensors metadata first (most accurate)
        size_gb = self._get_safetensors_size(model_info.modelId)
        
        # Fallback to siblings file size if SafeTensors not available
        if size_gb is None and hasattr(model_info, 'siblings') and model_info.siblings:
            total_size = sum(getattr(sibling, 'size', 0) or 0 for sibling in model_info.siblings)
            size_gb = total_size / (1024**3) if total_size > 0 else None
        
        # If still no size info, use enhanced model-specific estimates
        if size_gb is None:
            model_name = getattr(model_info, 'modelId', getattr(model_info, 'id', '')).lower()
            if 'gpt2-medium' in model_name or 'gpt-2-medium' in model_name:
                size_gb = 1.4  # GPT2-medium is ~1.4GB
            elif 'dialogpt-medium' in model_name or 'dialagpt-medium' in model_name:
                size_gb = 1.4  # DialoGPT-medium is ~1.4GB
            elif 'distilgpt2' in model_name or 'distil-gpt2' in model_name:
                size_gb = 0.3  # DistilGPT2 is ~300MB
            elif 'gpt2' in model_name and ('large' in model_name or 'xl' in model_name):
                size_gb = 3.0  # GPT2-large/xl is ~3GB
            elif 'gpt2' in model_name:
                size_gb = 0.5  # Standard GPT2 is ~500MB
            elif 'distil' in model_name:
                size_gb = 0.3  # Other distilled models are typically smaller
            elif 'small' in model_name:
                size_gb = 0.3
            elif 'base' in model_name:
                size_gb = 0.5
            elif 'large' in model_name:
                size_gb = 1.5
            else:
                size_gb = 0.5  # Default estimate
        
        # Extract language from tags
        language = None
        if model_info.tags:
            for tag in model_info.tags:
                if tag.startswith('language:'):
                    language = tag.replace('language:', '')
                    break
                elif tag in ['en', 'english', 'multilingual', 'zh', 'chinese', 'es', 'spanish']:
                    language = tag
                    break
        
        return ModelMetadata(
            id=model_info.modelId,
            author=model_info.author or "unknown",
            task=getattr(model_info, 'pipeline_tag', None),
            library=self._extract_library(model_info.tags or []),
            language=language,
            license=getattr(model_info, 'license', None),
            downloads=getattr(model_info, 'downloads', 0) or 0,
            likes=getattr(model_info, 'likes', 0) or 0,
            tags=model_info.tags or [],
            size_gb=size_gb,
            created_at=str(model_info.created_at) if hasattr(model_info, 'created_at') and model_info.created_at else None,
            last_modified=str(model_info.lastModified) if hasattr(model_info, 'lastModified') and model_info.lastModified else None,
            description=getattr(model_info, 'description', None),
            pipeline_tag=getattr(model_info, 'pipeline_tag', None),
            model_card_available=hasattr(model_info, 'cardData') and model_info.cardData is not None
        )
    
    def _extract_library(self, tags: List[str]) -> Optional[str]:
        """Extract the primary library from model tags."""
        for tag in tags:
            if tag in self.compatible_libraries:
                return tag
        return None
    
    def _get_safetensors_size(self, model_id: str) -> Optional[float]:
        """Get model size from SafeTensors metadata (most accurate method).
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Size in GB, or None if not available
        """
        try:
            # Get model info including SafeTensors metadata
            model_info = self.api.model_info(model_id, files_metadata=True)
            
            # Look for SafeTensors metadata with parameter count
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                # SafeTensors info has a 'total' attribute with parameter count
                if hasattr(model_info.safetensors, 'total'):
                    param_count = model_info.safetensors.total
                    # Parameters count * 4 bytes (float32) / 1024^3 = GB
                    return (param_count * 4) / (1024**3)
            
            return None
        except Exception:
            # Silently return None if API call fails
            return None
    
    def _meets_criteria(self, metadata: ModelMetadata, criteria: ModelSearchCriteria) -> bool:
        """Check if a model meets the search criteria (alias for _matches_criteria)."""
        return self._matches_criteria(metadata, criteria)
    
    def _matches_criteria(self, metadata: ModelMetadata, criteria: ModelSearchCriteria) -> bool:
        """Check if a model matches the search criteria."""
        
        # Filter by minimum downloads
        if criteria.min_downloads and metadata.downloads < criteria.min_downloads:
            return False
        
        # Filter by maximum size
        if criteria.max_size_gb and metadata.size_gb and metadata.size_gb > criteria.max_size_gb:
            return False
        
        # Filter by language
        if criteria.language and metadata.language:
            if criteria.language.lower() not in metadata.language.lower():
                return False
        
        # Filter by license
        if criteria.license and metadata.license:
            if criteria.license.lower() != metadata.license.lower():
                return False
        
        # Check if it's a text generation model (if no specific task given)
        if not criteria.task and metadata.task:
            if metadata.task not in self.text_generation_tasks:
                return False
        
        return True
    
    def _check_compatibility(self, model_info: ModelInfo) -> tuple[bool, Optional[str]]:
        """Check if a model is compatible with our system."""
        
        if not model_info.tags:
            return True, None
        
        # Check if it uses compatible libraries
        libraries = [tag for tag in model_info.tags if tag in self.compatible_libraries]
        if not libraries:
            return False, "No compatible library found (transformers, pytorch, etc.)"
        
        # Check for known incompatible tags
        incompatible_tags = {'ggml', 'gguf', 'cpp'}  # These require special handling
        if any(tag in model_info.tags for tag in incompatible_tags):
            return False, "Requires special runtime (GGML/GGUF format)"
        
        # Check if it's a text generation model
        if hasattr(model_info, 'pipeline_tag') and model_info.pipeline_tag:
            if model_info.pipeline_tag not in self.text_generation_tasks:
                return False, f"Not a text generation model (task: {model_info.pipeline_tag})"
        
        return True, None
    
    def save_search_results(self, results: List[ModelMetadata], filename: str) -> Path:
        """Save search results to a JSON file.
        
        Args:
            results: List of model metadata
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        output_path = self.cache_dir / filename
        
        # Convert to dict for JSON serialization
        results_dict = {
            "models": [
                {
                    "id": model.id,
                    "author": model.author,
                    "task": model.task,
                    "library": model.library,
                    "language": model.language,
                    "license": model.license,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "size_gb": model.size_gb,
                    "created_at": model.created_at,
                    "last_modified": model.last_modified,
                    "description": model.description,
                    "pipeline_tag": model.pipeline_tag,
                    "model_card_available": model.model_card_available,
                    "compatible": model.compatible,
                    "compatibility_notes": model.compatibility_notes
                }
                for model in results
            ],
            "search_timestamp": str(Path().resolve()),
            "total_results": len(results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} models to {output_path}")
        return output_path
    
    def load_search_results(self, filename: str) -> List[ModelMetadata]:
        """Load previously saved search results.
        
        Args:
            filename: Input filename
            
        Returns:
            List of model metadata
        """
        input_path = self.cache_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Search results file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        models = []
        for model_dict in data.get("models", []):
            model = ModelMetadata(**model_dict)
            models.append(model)
        
        logger.info(f"Loaded {len(models)} models from {input_path}")
        return models