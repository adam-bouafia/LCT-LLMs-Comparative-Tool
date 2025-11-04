"""
API Key Management for LLM Experiment Runner

Handles storage and retrieval of API keys for various services.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API keys for various services."""

    def __init__(self):
        """Initialize API key manager."""
        self.config_dir = Path.home() / ".lct"
        self.config_file = self.config_dir / "api_keys.json"
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if self.config_file.exists():
            try:
                self.config_file.chmod(0o600)
            except Exception as e:
                logger.warning(f"Could not set file permissions: {e}")

    def set_api_key(self, service: str, api_key: str) -> None:
        """
        Store API key for a service.

        Args:
            service: Service name (e.g., 'openai', 'huggingface')
            api_key: API key to store
        """
        keys = self._load_keys()
        keys[service.lower()] = api_key
        self._save_keys(keys)
        logger.info(f"API key for {service} stored successfully")

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a service.

        Args:
            service: Service name (e.g., 'openai', 'huggingface')

        Returns:
            API key if found, None otherwise
        """
        # Check environment variable first
        env_var_name = f"{service.upper()}_API_KEY"
        env_key = os.getenv(env_var_name)
        if env_key:
            return env_key

        # Check stored keys
        keys = self._load_keys()
        return keys.get(service.lower())

    def remove_api_key(self, service: str) -> bool:
        """
        Remove API key for a service.

        Args:
            service: Service name

        Returns:
            True if key was removed, False if not found
        """
        keys = self._load_keys()
        if service.lower() in keys:
            del keys[service.lower()]
            self._save_keys(keys)
            logger.info(f"API key for {service} removed")
            return True
        return False

    def list_services(self) -> list:
        """
        List all services with stored API keys.

        Returns:
            List of service names
        """
        keys = self._load_keys()
        return list(keys.keys())

    def has_key(self, service: str) -> bool:
        """
        Check if API key exists for a service.

        Args:
            service: Service name

        Returns:
            True if key exists, False otherwise
        """
        return self.get_api_key(service) is not None

    def _load_keys(self) -> Dict[str, str]:
        """Load API keys from config file."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}

    def _save_keys(self, keys: Dict[str, str]) -> None:
        """Save API keys to config file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(keys, f, indent=2)

            self.config_file.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            raise


# Global instance
_api_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
