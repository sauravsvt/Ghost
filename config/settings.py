"""
Settings - Configuration Constants

Central configuration for Ghost-1 system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScreenSettings:
    """Screen capture and resolution settings."""
    default_width: int = 1920
    default_height: int = 1080
    capture_interval_ms: int = 100  # Minimum time between captures
    downscale_factor: float = 0.5   # Reduce resolution for model input


@dataclass
class SafetySettings:
    """Safety and failsafe configuration."""
    failsafe_enabled: bool = True    # PyAutoGUI failsafe (top-left corner)
    max_actions_per_minute: int = 60  # Rate limiting
    require_confirmation_for_delete: bool = True
    blocked_urls: tuple = (
        "bank",
        "paypal",
        "admin",
        "root"
    )
    

@dataclass
class ModelSettings:
    """Model and inference settings."""
    model_path: str = "models/ghost-bimamba-3b-1.58bit.gguf"
    context_length: int = 1_000_000
    temperature: float = 0.2
    top_p: float = 0.9
    use_calm_decoding: bool = True
    calm_draft_tokens: int = 4
    

@dataclass  
class TrainingSettings:
    """GRPO training configuration."""
    enabled: bool = False  # Disabled by default
    learning_rate: float = 1e-5
    batch_size: int = 4
    group_size: int = 8
    checkpoint_dir: str = "checkpoints/"
    log_trajectories: bool = True


# Global settings instance
SCREEN = ScreenSettings()
SAFETY = SafetySettings()
MODEL = ModelSettings()
TRAINING = TrainingSettings()


# Version info
VERSION = "1.0.0"
CODENAME = "Terminal Edition"
