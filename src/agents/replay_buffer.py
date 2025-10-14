"""Experience replay buffer for DQN agent."""

import numpy as np
from typing import List, Tuple
from src.utils.data_models import Transition


class ReplayBuffer:
    """
    Circular buffer for storing and sampling experience transitions.

    Implements efficient storage using NumPy arrays and circular indexing
    to avoid memory reallocation when buffer is full.
    """

    def __init__(self, capacity: int, observation_dim: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store (10k-100k typical)
            observation_dim: Dimension of observation space
        """
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.position = 0
        self.size = 0

        # Pre-allocate arrays for efficient storage
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, transition: Transition) -> None:
        """
        Add a transition to the buffer.

        Uses circular indexing: when buffer is full, oldest transitions
        are overwritten.

        Args:
            transition: Experience tuple to store
        """
        # Store transition components in pre-allocated arrays
        self.states[self.position] = transition.state
        self.actions[self.position] = transition.action
        self.rewards[self.position] = transition.reward
        self.next_states[self.position] = transition.next_state
        self.dones[self.position] = transition.done

        # Update circular buffer position
        self.position = (self.position + 1) % self.capacity

        # Track actual size (up to capacity)
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as NumPy arrays

        Raises:
            ValueError: If batch_size > current buffer size
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with only {self.size} transitions"
            )

        # Sample random indices without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Return sampled transitions
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for a batch.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer contains at least batch_size transitions
        """
        return self.size >= batch_size
