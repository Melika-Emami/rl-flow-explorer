"""Base agent interface for all RL agents."""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action given observation.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode (affects exploration)

        Returns:
            Selected action as integer
        """
        pass

    @abstractmethod
    def update(self, transition: "Transition") -> Dict[str, float]:
        """
        Update agent from experience.

        Args:
            transition: Experience tuple containing (state, action, reward, next_state, done, info)

        Returns:
            Dictionary of metrics (e.g., loss, q_values, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: File path to save agent state
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: File path to load agent state from
        """
        pass
