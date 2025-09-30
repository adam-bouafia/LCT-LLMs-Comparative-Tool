"""Data Manager Module

This module handles downloading and organizing models and datasets
into the data/ directory structure.
"""

import os
import logging
import shutil
from typing import Optional, Dict, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataManager:
    """Manages model and dataset downloads to organized folder structure."""
    
    def __init__(self, data_root: str = None):
        """Initialize the data manager.
        
        Args:
            data_root: Root directory for data storage (defaults to ./data)
        """
        if data_root is None:
            # Get project root (parent of app/src/core)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            data_root = project_root / "data"
        
        self.data_root = Path(data_root)
        self.models_dir = self.data_root / "models"
        self.datasets_dir = self.data_root / "datasets"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataManager initialized - Models: {self.models_dir}, Datasets: {self.datasets_dir}")
    
    def get_model_cache_dir(self, model_id: str) -> Path:
        """Get the cache directory for a specific model.
        
        Args:
            model_id: HuggingFace model ID (e.g., "gpt2", "microsoft/DialoGPT-medium")
            
        Returns:
            Path to model's cache directory
        """
        # Clean model ID for directory name (replace slashes with dashes)
        clean_name = model_id.replace("/", "--")
        model_dir = self.models_dir / clean_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_dataset_cache_dir(self, dataset_name: str) -> Path:
        """Get the cache directory for a specific dataset.
        
        Args:
            dataset_name: Dataset name (e.g., "xsum", "cnn_dailymail")
            
        Returns:
            Path to dataset's cache directory
        """
        # Clean dataset name for directory name
        clean_name = dataset_name.replace("/", "--").replace("_", "-")
        dataset_dir = self.datasets_dir / clean_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    
    def set_huggingface_cache_dirs(self):
        """Set HuggingFace environment variables to use our organized cache dirs."""
        # Set main HuggingFace cache directory
        os.environ["HF_HOME"] = str(self.data_root)
        os.environ["TRANSFORMERS_CACHE"] = str(self.models_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.datasets_dir)
        
        logger.info("Set HuggingFace cache directories:")
        logger.info(f"  HF_HOME: {self.data_root}")
        logger.info(f"  TRANSFORMERS_CACHE: {self.models_dir}")
        logger.info(f"  HF_DATASETS_CACHE: {self.datasets_dir}")
    
    def download_model(self, model_id: str, model_type: str = "causal") -> Optional[Path]:
        """Download a model to the organized cache structure.
        
        Args:
            model_id: HuggingFace model ID
            model_type: Type of model ("causal", "seq2seq", "encoder")
            
        Returns:
            Path to downloaded model directory, or None if failed
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            
            model_cache_dir = self.get_model_cache_dir(model_id)
            
            logger.info(f"📥 Downloading model {model_id} to {model_cache_dir}")
            print(f"📥 DOWNLOADING MODEL: {model_id}")
            print(f"   📁 Cache directory: {model_cache_dir}")
            
            # Download tokenizer
            print(f"   📝 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(model_cache_dir),
                trust_remote_code=True
            )
            print(f"   ✅ Tokenizer downloaded (vocab size: {tokenizer.vocab_size})")
            
            # Download model
            print(f"   🤖 Downloading model weights...")
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    cache_dir=str(model_cache_dir),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=str(model_cache_dir),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            print(f"   ✅ Model downloaded successfully!")
            
            # Save metadata
            metadata = {
                "model_id": model_id,
                "model_type": model_type,
                "download_date": str(Path(__file__).stat().st_mtime),
                "cache_dir": str(model_cache_dir),
                "config": {
                    "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else None,
                    "model_type": getattr(model.config, 'model_type', 'unknown'),
                    "architectures": getattr(model.config, 'architectures', [])
                }
            }
            
            with open(model_cache_dir / "download_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model {model_id} downloaded to {model_cache_dir}")
            return model_cache_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_id}: {e}")
            print(f"❌ Error downloading {model_id}: {e}")
            return None
    
    def download_dataset(self, dataset_name: str, config_name: Optional[str] = None) -> Optional[Path]:
        """Download a dataset to the organized cache structure.
        
        Args:
            dataset_name: Dataset name from HuggingFace
            config_name: Optional config name for the dataset
            
        Returns:
            Path to downloaded dataset directory, or None if failed
        """
        try:
            from datasets import load_dataset
            
            dataset_cache_dir = self.get_dataset_cache_dir(dataset_name)
            
            logger.info(f"📥 Downloading dataset {dataset_name} to {dataset_cache_dir}")
            print(f"📥 DOWNLOADING DATASET: {dataset_name}")
            if config_name:
                print(f"   ⚙️  Config: {config_name}")
            print(f"   📁 Cache directory: {dataset_cache_dir}")
            
            # Download dataset
            print(f"   📊 Downloading dataset files...")
            dataset = load_dataset(
                dataset_name,
                config_name,
                cache_dir=str(dataset_cache_dir)
            )
            
            print(f"   ✅ Dataset downloaded successfully!")
            
            # Save metadata
            metadata = {
                "dataset_name": dataset_name,
                "config_name": config_name,
                "download_date": str(Path(__file__).stat().st_mtime),
                "cache_dir": str(dataset_cache_dir),
                "splits": list(dataset.keys()) if hasattr(dataset, 'keys') else [],
                "features": str(dataset[list(dataset.keys())[0]].features) if dataset and hasattr(dataset, 'keys') and dataset.keys() else None
            }
            
            with open(dataset_cache_dir / "download_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Dataset {dataset_name} downloaded to {dataset_cache_dir}")
            return dataset_cache_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download dataset {dataset_name}: {e}")
            print(f"❌ Error downloading {dataset_name}: {e}")
            return None
    
    def list_downloaded_models(self) -> List[Dict]:
        """List all downloaded models with metadata."""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "download_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        metadata["local_path"] = str(model_dir)
                        models.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {model_dir}: {e}")
                else:
                    # Directory exists but no metadata
                    models.append({
                        "model_id": model_dir.name.replace("--", "/"),
                        "local_path": str(model_dir),
                        "metadata_missing": True
                    })
        
        return models
    
    def list_downloaded_datasets(self) -> List[Dict]:
        """List all downloaded datasets with metadata."""
        datasets = []
        
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "download_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        metadata["local_path"] = str(dataset_dir)
                        datasets.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {dataset_dir}: {e}")
                else:
                    # Directory exists but no metadata
                    datasets.append({
                        "dataset_name": dataset_dir.name.replace("--", "/"),
                        "local_path": str(dataset_dir),
                        "metadata_missing": True
                    })
        
        return datasets
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage information for data directories.
        
        Returns:
            Dictionary with usage info in GB
        """
        def get_size(path):
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total += os.path.getsize(fp)
                        except (OSError, IOError):
                            pass
            except (OSError, IOError):
                pass
            return total / (1024**3)  # Convert to GB
        
        return {
            "models_gb": get_size(self.models_dir),
            "datasets_gb": get_size(self.datasets_dir),
            "total_gb": get_size(self.data_root)
        }
    
    def cleanup_model(self, model_id: str) -> bool:
        """Remove a downloaded model from cache.
        
        Args:
            model_id: Model ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_cache_dir = self.get_model_cache_dir(model_id)
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)
                logger.info(f"🗑️ Removed model {model_id} from cache")
                return True
            else:
                logger.warning(f"Model {model_id} not found in cache")
                return False
        except Exception as e:
            logger.error(f"Failed to remove model {model_id}: {e}")
            return False
    
    def cleanup_dataset(self, dataset_name: str) -> bool:
        """Remove a downloaded dataset from cache.
        
        Args:
            dataset_name: Dataset name to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dataset_cache_dir = self.get_dataset_cache_dir(dataset_name)
            if dataset_cache_dir.exists():
                shutil.rmtree(dataset_cache_dir)
                logger.info(f"🗑️ Removed dataset {dataset_name} from cache")
                return True
            else:
                logger.warning(f"Dataset {dataset_name} not found in cache")
                return False
        except Exception as e:
            logger.error(f"Failed to remove dataset {dataset_name}: {e}")
            return False


# Global data manager instance
_data_manager = None

def get_data_manager() -> DataManager:
    """Get the global data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
        _data_manager.set_huggingface_cache_dirs()
    return _data_manager