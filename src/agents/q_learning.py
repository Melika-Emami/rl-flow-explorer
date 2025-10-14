"""Tabular Q-learning agent implementation."""

import pickle
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np

from src.agents.base import Agent
from src.utils.data_models import Transition


class QLearningAgent(Agent):
    """
    Tabular Q-learning agent with epsilon-greedy exploration.

    Uses a dictionary-based Q-table for discrete state-action spaces.
    Supports epsilon decay and state discretization for continuous observations.
    """

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        discretization_bins: int = 10,
    ):
        """
        Initialize Q-learning agent.

        Args:
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            discretization_bins: Number of bins for discretizing continuous observations
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discretization_bins = discretization_bins

        # Q-table as dictionary mapping (state, action) to Q-values
        # Using defaultdict for automatic initialization to 0.0
        self.q_table: Dict[Tuple, float] = defaultdict(float)

        # Track number of updates for metrics
        self.num_updates = 0

    def _discretize_state(self, observation: np.ndarray) -> Tuple:
        """
        Discretize continuous observation into a hashable state representation.

        Args:
            observation: Continuous observation array

        Returns:
            Tuple representing discretized state (hashable for dict keys)
        """
        # For continuous observations, discretize into bins
        # For already discrete observations (e.g., one-hot), convert to tuple
        if observation.dtype == np.float32 or observation.dtype == np.float64:
            # Normalize to [0, 1] range and discretize
            obs_min = observation.min()
            obs_max = observation.max()
            if obs_max - obs_min > 0:
                normalized = (observation - obs_min) / (obs_max - obs_min)
            else:
                normalized = observation
            discretized = (normalized * self.discretization_bins).astype(int)
            discretized = np.clip(discretized, 0, self.discretization_bins - 1)
            return tuple(discretized)
        else:
            # Already discrete, just convert to tuple
            return tuple(observation.astype(int))

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode (affects exploration)

        Returns:
            Selected action as integer
        """
        state = self._discretize_state(observation)

        # Epsilon-greedy exploration (only during training)
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_space_size)
        else:
            # Exploit: greedy action based on Q-values
            q_values = [self.q_table[(state, a)] for a in range(self.action_space_size)]
            max_q = max(q_values)

            # Handle ties by randomly selecting among actions with max Q-value
            best_actions = [
                a for a in range(self.action_space_size) if q_values[a] == max_q
            ]
            return np.random.choice(best_actions)

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Update Q-values using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        Args:
            transition: Experience tuple containing (state, action, reward, next_state, done, info)

        Returns:
            Dictionary of metrics (loss, q_value, etc.)
        """
        state = self._discretize_state(transition.state)
        next_state = self._discretize_state(transition.next_state)
        action = transition.action
        reward = transition.reward
        done = transition.done

        # Current Q-value
        current_q = self.q_table[(state, action)]

        # Compute target Q-value
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: add discounted max future Q-value
            next_q_values = [
                self.q_table[(next_state, a)] for a in range(self.action_space_size)
            ]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q

        # TD error
        td_error = target_q - current_q

        # Q-learning update
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

        self.num_updates += 1

        # Return metrics for logging
        return {
            "q_value": current_q,
            "td_error": abs(td_error),
            "target_q": target_q,
            "epsilon": self.epsilon,
        }

    def decay_epsilon(self) -> None:
        """
        Decay epsilon according to decay schedule.

        Should be called at the end of each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: File path to save agent state
        """
        state = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "discretization_bins": self.discretization_bins,
            "num_updates": self.num_updates,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: File path to load agent state from
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(float, state["q_table"])
        self.action_space_size = state["action_space_size"]
        self.learning_rate = state["learning_rate"]
        self.discount_factor = state["discount_factor"]
        self.epsilon = state["epsilon"]
        self.epsilon_min = state["epsilon_min"]
        self.epsilon_decay = state["epsilon_decay"]
        self.discretization_bins = state["discretization_bins"]
        self.num_updates = state["num_updates"]

    def get_q_table_size(self) -> int:
        """
        Get the number of state-action pairs in the Q-table.

        Returns:
            Number of entries in Q-table
        """
        return len(self.q_table)

    def get_state_action_value(self, observation: np.ndarray, action: int) -> float:
        """
        Get Q-value for a specific state-action pair.

        Args:
            observation: State observation
            action: Action index

        Returns:
            Q-value for the state-action pair
        """
        state = self._discretize_state(observation)
        return self.q_table[(state, action)]
