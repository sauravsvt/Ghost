"""
Configuration Manager for Ghost-1
Loads settings from config.yaml and .env
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Load .env file
        load_dotenv()
        
        # Load YAML config
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.
        Example: config.get('model.n_ctx')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return os.getenv(key, default)
    
    # Convenience accessors
    @property
    def model_path(self) -> str:
        """Get model path (env override or config)."""
        return self.get_env('MODEL_PATH') or self.get('model.path')
    
    @property
    def max_steps(self) -> int:
        return self.get('agent.max_steps', 20)
    
    @property
    def action_delay(self) -> float:
        return self.get('agent.action_delay', 0.5)
    
    @property
    def stuck_threshold(self) -> int:
        return self.get('agent.stuck_threshold', 3)
    
    @property
    def loop_threshold(self) -> int:
        return self.get('agent.loop_detection_threshold', 3)
    
    @property
    def dataset_path(self) -> str:
        return self.get('paths.dataset', 'dataset.jsonl')
    
    @property
    def logs_dir(self) -> str:
        return self.get('paths.logs_dir', 'logs')
    
    @property
    def models_dir(self) -> str:
        return self.get('paths.models_dir', 'models')

# Global config instance
_config: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config
