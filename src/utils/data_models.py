"""Core data models and configuration classes."""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class Transition:
    """Single environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 32
    eval_frequency: int = 100
    num_eval_episodes: int = 10
    random_seed: int = 42
    checkpoint_frequency: int = 500
    log_frequency: int = 10
    early_stopping_patience: int = 50
    save_dir: str = "checkpoints"
