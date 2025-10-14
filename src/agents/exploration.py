"""Exploration strategies for reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ExplorationStrategy(ABC):
    """
    Abstract base class for exploration strategies.

    Exploration strategies compute intrinsic reward bonuses to encourage
    agents to explore novel or uncertain states.
    """

    @abstractmethod
    def compute_bonus(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[dict] = None,
    ) -> float:
        """
        Compute intrinsic reward bonus for exploration.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
            info: Optional additional information from environment

        Returns:
            Intrinsic reward bonus (non-negative float)
        """
        pass

    def reset(self) -> None:
        """
        Reset exploration strategy state (e.g., at episode start).

        Override this method if the strategy maintains episode-specific state.
        """
        pass

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """
        Update exploration strategy based on observed transition.

        Override this method if the strategy needs to learn or track statistics.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
        """
        pass


class EpsilonGreedy(ExplorationStrategy):
    """
    Epsilon-greedy exploration strategy.

    With probability epsilon, selects a random action (exploration).
    With probability 1-epsilon, selects the greedy action (exploitation).

    Supports epsilon decay over time to reduce exploration as learning progresses.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        decay_mode: str = "episode",
    ):
        """
        Initialize epsilon-greedy exploration strategy.

        Args:
            epsilon: Initial exploration rate (probability of random action)
            epsilon_min: Minimum exploration rate (lower bound)
            epsilon_decay: Decay factor applied to epsilon
            decay_mode: When to decay epsilon ('episode' or 'step')
                       'episode': decay after each episode
                       'step': decay after each step
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError(f"epsilon_min must be in [0, 1], got {epsilon_min}")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")
        if decay_mode not in ["episode", "step"]:
            raise ValueError(
                f"decay_mode must be 'episode' or 'step', got {decay_mode}"
            )

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_mode = decay_mode
        self.initial_epsilon = epsilon

    def compute_bonus(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[dict] = None,
    ) -> float:
        """
        Compute intrinsic reward bonus.

        Epsilon-greedy doesn't provide intrinsic rewards; exploration is
        handled through action selection. Returns 0.0.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
            info: Optional additional information from environment

        Returns:
            Always returns 0.0 (no intrinsic reward)
        """
        return 0.0

    def should_explore(self) -> bool:
        """
        Determine whether to explore (random action) or exploit (greedy action).

        Returns:
            True if should explore, False if should exploit
        """
        return np.random.random() < self.epsilon

    def decay(self) -> None:
        """
        Decay epsilon according to decay schedule.

        Should be called after each episode (if decay_mode='episode')
        or after each step (if decay_mode='step').
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """
        Update exploration strategy based on observed transition.

        If decay_mode is 'step', decays epsilon after each update.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
        """
        if self.decay_mode == "step":
            self.decay()

    def reset(self) -> None:
        """
        Reset exploration strategy state.

        Does not reset epsilon value (exploration rate persists across episodes).
        """
        pass

    def reset_epsilon(self) -> None:
        """
        Reset epsilon to initial value.

        Useful for restarting training or evaluation.
        """
        self.epsilon = self.initial_epsilon

    def get_epsilon(self) -> float:
        """
        Get current epsilon value.

        Returns:
            Current exploration rate
        """
        return self.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """
        Set epsilon to a specific value.

        Args:
            epsilon: New exploration rate (must be in [0, 1])
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon


class CountBasedBonus(ExplorationStrategy):
    """
    Count-based exploration bonus strategy.

    Provides intrinsic rewards inversely proportional to state visit counts,
    encouraging exploration of novel or rarely-visited states.

    The bonus is computed as: bonus = beta / sqrt(count(state))
    where count(state) is the number of times the state has been visited.
    """

    def __init__(
        self,
        beta: float = 1.0,
        state_hash_fn: Optional[callable] = None,
        count_mode: str = "state",
    ):
        """
        Initialize count-based exploration bonus strategy.

        Args:
            beta: Scaling factor for intrinsic reward bonus
            state_hash_fn: Optional function to hash states for counting.
                          If None, uses tuple(state.flatten()) as hash.
            count_mode: What to count ('state' or 'state_action')
                       'state': count visits to states
                       'state_action': count visits to state-action pairs
        """
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if count_mode not in ["state", "state_action"]:
            raise ValueError(
                f"count_mode must be 'state' or 'state_action', got {count_mode}"
            )

        self.beta = beta
        self.state_hash_fn = state_hash_fn or self._default_hash
        self.count_mode = count_mode

        # Visit count tables
        self.state_counts: Dict[Tuple, int] = {}
        self.state_action_counts: Dict[Tuple, int] = {}

        # Statistics for logging
        self.total_bonuses = 0.0
        self.num_bonuses = 0
        self.max_bonus = 0.0
        self.min_bonus = float("inf")

    def _default_hash(self, state: np.ndarray) -> Tuple:
        """
        Default hash function for states.

        Converts state array to tuple for use as dictionary key.

        Args:
            state: State observation array

        Returns:
            Tuple representation of state
        """
        return tuple(state.flatten())

    def compute_bonus(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[dict] = None,
    ) -> float:
        """
        Compute intrinsic reward bonus based on visit counts.

        The bonus is inversely proportional to the square root of the visit count,
        encouraging exploration of novel or rarely-visited states.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
            info: Optional additional information from environment

        Returns:
            Intrinsic reward bonus (non-negative float)
        """
        # Hash the next state (we reward visiting novel next states)
        state_hash = self.state_hash_fn(next_state)

        if self.count_mode == "state":
            # Count based on state visits
            count = self.state_counts.get(state_hash, 0)
        else:  # state_action
            # Count based on state-action pair visits
            state_action_hash = (self.state_hash_fn(state), action)
            count = self.state_action_counts.get(state_action_hash, 0)

        # Compute bonus: beta / sqrt(count + 1)
        # Add 1 to avoid division by zero on first visit
        bonus = self.beta / np.sqrt(count + 1)

        # Update statistics
        self.total_bonuses += bonus
        self.num_bonuses += 1
        self.max_bonus = max(self.max_bonus, bonus)
        self.min_bonus = min(self.min_bonus, bonus)

        return bonus

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """
        Update visit counts based on observed transition.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
        """
        # Hash the next state
        next_state_hash = self.state_hash_fn(next_state)

        # Update state visit count
        if next_state_hash not in self.state_counts:
            self.state_counts[next_state_hash] = 0
        self.state_counts[next_state_hash] += 1

        # Update state-action visit count if needed
        if self.count_mode == "state_action":
            state_action_hash = (self.state_hash_fn(state), action)
            if state_action_hash not in self.state_action_counts:
                self.state_action_counts[state_action_hash] = 0
            self.state_action_counts[state_action_hash] += 1

    def reset(self) -> None:
        """
        Reset episode-specific state.

        Note: Does not reset visit counts (they persist across episodes).
        """
        pass

    def reset_counts(self) -> None:
        """
        Reset all visit counts.

        Useful for starting a new training run or evaluation.
        """
        self.state_counts.clear()
        self.state_action_counts.clear()
        self.total_bonuses = 0.0
        self.num_bonuses = 0
        self.max_bonus = 0.0
        self.min_bonus = float("inf")

    def get_statistics(self) -> Dict[str, float]:
        """
        Get intrinsic reward statistics for logging.

        Returns:
            Dictionary containing:
                - mean_bonus: Average intrinsic reward
                - max_bonus: Maximum intrinsic reward
                - min_bonus: Minimum intrinsic reward
                - num_states_visited: Number of unique states visited
                - total_bonuses: Sum of all bonuses
        """
        stats = {
            "mean_bonus": (
                self.total_bonuses / self.num_bonuses if self.num_bonuses > 0 else 0.0
            ),
            "max_bonus": self.max_bonus if self.num_bonuses > 0 else 0.0,
            "min_bonus": self.min_bonus if self.num_bonuses > 0 else 0.0,
            "num_states_visited": len(self.state_counts),
            "total_bonuses": self.total_bonuses,
        }

        if self.count_mode == "state_action":
            stats["num_state_actions_visited"] = len(self.state_action_counts)

        return stats

    def get_state_count(self, state: np.ndarray) -> int:
        """
        Get visit count for a specific state.

        Args:
            state: State observation

        Returns:
            Number of times state has been visited
        """
        state_hash = self.state_hash_fn(state)
        return self.state_counts.get(state_hash, 0)

    def get_state_action_count(self, state: np.ndarray, action: int) -> int:
        """
        Get visit count for a specific state-action pair.

        Args:
            state: State observation
            action: Action

        Returns:
            Number of times state-action pair has been visited
        """
        state_action_hash = (self.state_hash_fn(state), action)
        return self.state_action_counts.get(state_action_hash, 0)


class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model that predicts next state from current state and action.

    Used by curiosity-based exploration to compute prediction error as intrinsic reward.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ):
        """
        Initialize forward dynamics model.

        Args:
            state_dim: Dimension of state observation
            action_dim: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        input_dim = state_dim + action_dim  # Concatenate state and one-hot action

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer predicts next state
        layers.append(nn.Linear(input_dim, state_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state from current state and action.

        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim) - one-hot encoded

        Returns:
            Predicted next state (batch_size, state_dim)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class CuriosityBonus(ExplorationStrategy):
    """
    Curiosity-based exploration bonus strategy.

    Uses a forward dynamics model to predict next states. The prediction error
    serves as an intrinsic reward bonus, encouraging exploration of states that
    are difficult to predict (i.e., novel or surprising states).

    The forward model is trained alongside the agent to continuously improve
    its predictions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        beta: float = 0.1,
        learning_rate: float = 1e-3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        batch_size: int = 32,
        train_frequency: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize curiosity-based exploration bonus strategy.

        Args:
            state_dim: Dimension of state observation
            action_dim: Number of possible actions
            beta: Scaling factor for intrinsic reward bonus
            learning_rate: Learning rate for forward model training
            hidden_dims: Tuple of hidden layer dimensions for forward model
            batch_size: Batch size for training forward model
            train_frequency: Train forward model every N transitions
            device: Device to run model on ('cpu' or 'cuda')
        """
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if train_frequency <= 0:
            raise ValueError(f"train_frequency must be positive, got {train_frequency}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.device = torch.device(device)

        # Initialize forward dynamics model
        self.forward_model = ForwardDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Optimizer for training forward model
        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)

        # Loss function (MSE for next state prediction)
        self.criterion = nn.MSELoss()

        # Experience buffer for training forward model
        self.experience_buffer = []
        self.max_buffer_size = 10000

        # Training statistics
        self.num_updates = 0
        self.total_loss = 0.0

        # Intrinsic reward statistics
        self.total_bonuses = 0.0
        self.num_bonuses = 0
        self.max_bonus = 0.0
        self.min_bonus = float("inf")

    def _action_to_onehot(self, action: int) -> np.ndarray:
        """
        Convert action index to one-hot encoding.

        Args:
            action: Action index

        Returns:
            One-hot encoded action array
        """
        onehot = np.zeros(self.action_dim)
        onehot[action] = 1.0
        return onehot

    def compute_bonus(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[dict] = None,
    ) -> float:
        """
        Compute intrinsic reward bonus based on forward model prediction error.

        The bonus is proportional to the prediction error (MSE between predicted
        and actual next state), encouraging exploration of surprising transitions.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
            info: Optional additional information from environment

        Returns:
            Intrinsic reward bonus (non-negative float)
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_onehot = self._action_to_onehot(action)
        action_tensor = torch.FloatTensor(action_onehot).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Predict next state
        with torch.no_grad():
            predicted_next_state = self.forward_model(state_tensor, action_tensor)

        # Compute prediction error (MSE)
        prediction_error = torch.mean(
            (predicted_next_state - next_state_tensor) ** 2
        ).item()

        # Scale by beta to get intrinsic reward bonus
        bonus = self.beta * prediction_error

        # Update statistics
        self.total_bonuses += bonus
        self.num_bonuses += 1
        self.max_bonus = max(self.max_bonus, bonus)
        self.min_bonus = min(self.min_bonus, bonus)

        return bonus

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """
        Update forward model based on observed transition.

        Stores transition in experience buffer and trains forward model
        periodically on batches of transitions.

        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation after taking action
        """
        # Store transition in buffer
        self.experience_buffer.append((state, action, next_state))

        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        # Train forward model if enough data and at training frequency
        if (
            len(self.experience_buffer) >= self.batch_size
            and len(self.experience_buffer) % self.train_frequency == 0
        ):
            self._train_forward_model()

    def _train_forward_model(self) -> None:
        """
        Train forward model on a batch of transitions from experience buffer.
        """
        # Sample batch from experience buffer
        if len(self.experience_buffer) <= self.batch_size:
            batch = self.experience_buffer
        else:
            indices = np.random.choice(
                len(self.experience_buffer), self.batch_size, replace=False
            )
            batch = [self.experience_buffer[i] for i in indices]

        # Prepare batch tensors
        states = []
        actions = []
        next_states = []

        for state, action, next_state in batch:
            states.append(state)
            actions.append(self._action_to_onehot(action))
            next_states.append(next_state)

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)

        # Forward pass
        predicted_next_states = self.forward_model(states_tensor, actions_tensor)

        # Compute loss
        loss = self.criterion(predicted_next_states, next_states_tensor)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update statistics
        self.num_updates += 1
        self.total_loss += loss.item()

    def reset(self) -> None:
        """
        Reset episode-specific state.

        Note: Does not reset forward model or experience buffer.
        """
        pass

    def reset_model(self) -> None:
        """
        Reset forward model and experience buffer.

        Useful for starting a new training run or evaluation.
        """
        # Reinitialize forward model
        self.forward_model = ForwardDynamicsModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=(128, 128),
        ).to(self.device)

        # Reinitialize optimizer
        self.optimizer = optim.Adam(
            self.forward_model.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )

        # Clear experience buffer
        self.experience_buffer.clear()

        # Reset statistics
        self.num_updates = 0
        self.total_loss = 0.0
        self.total_bonuses = 0.0
        self.num_bonuses = 0
        self.max_bonus = 0.0
        self.min_bonus = float("inf")

    def get_statistics(self) -> Dict[str, float]:
        """
        Get intrinsic reward and forward model statistics for logging.

        Returns:
            Dictionary containing:
                - mean_bonus: Average intrinsic reward
                - max_bonus: Maximum intrinsic reward
                - min_bonus: Minimum intrinsic reward
                - total_bonuses: Sum of all bonuses
                - mean_forward_loss: Average forward model training loss
                - num_forward_updates: Number of forward model updates
                - buffer_size: Current size of experience buffer
        """
        stats = {
            "mean_bonus": (
                self.total_bonuses / self.num_bonuses if self.num_bonuses > 0 else 0.0
            ),
            "max_bonus": self.max_bonus if self.num_bonuses > 0 else 0.0,
            "min_bonus": self.min_bonus if self.num_bonuses > 0 else 0.0,
            "total_bonuses": self.total_bonuses,
            "mean_forward_loss": (
                self.total_loss / self.num_updates if self.num_updates > 0 else 0.0
            ),
            "num_forward_updates": self.num_updates,
            "buffer_size": len(self.experience_buffer),
        }

        return stats

    def save(self, path: str) -> None:
        """
        Save forward model state.

        Args:
            path: Path to save model checkpoint
        """
        torch.save(
            {
                "forward_model_state_dict": self.forward_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_updates": self.num_updates,
                "total_loss": self.total_loss,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load forward model state.

        Args:
            path: Path to load model checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.forward_model.load_state_dict(checkpoint["forward_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_updates = checkpoint["num_updates"]
        self.total_loss = checkpoint["total_loss"]
