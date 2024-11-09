import os
import random
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from threading import Lock
import time
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    api_key: str
    base_url: str
    weight: int = 1
    current_requests: int = 0
    total_requests: int = 0
    errors: int = 0
    last_error_time: float = 0

class KeyManager:
    def __init__(self):
        self._configs: Dict[str, List[APIConfig]] = {
            "openai": [],
            "anthropic": [],
            "gemini": [],
            "x": []
        }
        self._locks: Dict[str, Lock] = {
            "openai": Lock(),
            "anthropic": Lock(),
            "gemini": Lock(),
            "x": Lock()
        }
        self._load_configs()

    def _load_configs(self) -> None:
        """Load API configurations from environment variables."""
        # Try to load from ENV_FILE if set, otherwise try default .env
        env_file = os.getenv('ENV_FILE', '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Load OpenAI configs
        i = 1
        while True:
            key = os.getenv(f"OPENAI_API_KEY_{i}")
            base_url = os.getenv(f"OPENAI_API_BASE_{i}")
            if not key:
                break
            self._configs["openai"].append(APIConfig(
                api_key=key,
                base_url=base_url or "https://api.openai.com/v1"
            ))
            i += 1

        # Load Anthropic configs
        i = 1
        while True:
            key = os.getenv(f"ANTHROPIC_API_KEY_{i}")
            base_url = os.getenv(f"ANTHROPIC_API_BASE_{i}")
            if not key:
                break
            self._configs["anthropic"].append(APIConfig(
                api_key=key,
                base_url=base_url or "https://api.anthropic.com"
            ))
            i += 1
        
        # Load single keys for other services
        if gemini_key := os.getenv("GEMINI_API_KEY"):
            self._configs["gemini"].append(APIConfig(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta"
            ))

        if x_key := os.getenv("XAI_API_KEY"):
            self._configs["x"].append(APIConfig(
                api_key=x_key,
                base_url="https://api.x.ai/v1"
            ))

    def get_config(self, service: str) -> Tuple[str, Optional[str]]:
        """
        Get an API key and base URL using weighted round-robin selection.
        Returns (api_key, base_url)
        """
        with self._locks[service]:
            configs = self._configs[service]
            if not configs:
                raise ValueError(f"No API keys configured for {service}")

            # Sort by current_requests/weight ratio for load balancing
            configs.sort(key=lambda x: (x.current_requests/x.weight if x.weight > 0 else float('inf')))
            selected = configs[0]
            
            selected.current_requests += 1
            selected.total_requests += 1
            
            return selected.api_key, selected.base_url

    def release_config(self, service: str, api_key: str) -> None:
        """Release the API key after use."""
        with self._locks[service]:
            for config in self._configs[service]:
                if config.api_key == api_key:
                    config.current_requests -= 1
                    break

    def report_error(self, service: str, api_key: str) -> None:
        """Report an error for a specific API key."""
        with self._locks[service]:
            for config in self._configs[service]:
                if config.api_key == api_key:
                    config.errors += 1
                    config.last_error_time = time.time()
                    break

    def get_stats(self) -> Dict[str, List[Dict]]:
        """Get statistics for all API keys."""
        stats = {}
        for service, configs in self._configs.items():
            stats[service] = [
                {
                    "current_requests": c.current_requests,
                    "total_requests": c.total_requests,
                    "errors": c.errors,
                    "weight": c.weight
                }
                for c in configs
            ]
        return stats

# Global instance
key_manager = KeyManager() 