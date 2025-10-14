"""Deep Q-Network (DQN) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from pathlib import Path

from src.agents.base import Agent
from src.agents.dqn_network import DQNNetwork
from src.agents.replay_buffer import ReplayBuffer
from src.utils.data_models import Transition


class DQNAgent(Agent):
    """
    Deep Q-Network agent with experience replay and target network.

    Implements DQN algorithm with:
    - Experience replay buffer for breaking correlation between samples
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration with decay
    - Huber loss for robust training
    - Gradient clipping for stability
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 50000,
        batch_size: int = 32,
        target_update_frequency: int = 1000,
        hidden_dims: tuple = (256, 256),
        gradient_clip_value: float = 1.0,
        device: str = None,
    ):
        """
        Initialize DQN agent.

        Args:
            observation_dim: Dimension of observation space
            action_space_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            buffer_capacity: Capacity of experience replay buffer
            batch_size: Batch size for training
            target_update_frequency: Steps between target network updates
            hidden_dims: Tuple of hidden layer dimensions
            gradient_clip_value: Maximum gradient norm for clipping
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.gradient_clip_value = gradient_clip_value

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize policy network (online network)
        self.policy_network = DQNNetwork(
            observation_dim, action_space_size, hidden_dims
        ).to(self.device)

        # Initialize target network (copy of policy network)
        self.target_network = DQNNetwork(
            observation_dim, action_space_size, hidden_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, observation_dim)

        # Track training steps for target network updates
        self.training_steps = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode (affects exploration)

        Returns:
            Selected action as integer
        """
        # Epsilon-greedy exploration (only during training)
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_space_size)
        else:
            # Exploit: greedy action based on Q-values
            with torch.no_grad():
                # Convert observation to tensor
                state_tensor = (
                    torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                )

                # Get Q-values from policy network
                q_values = self.policy_network(state_tensor)

                # Select action with highest Q-value
                action = q_values.argmax(dim=1).item()

            return action

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Update agent from experience using DQN algorithm.

        Stores transition in replay buffer and performs training update
        if buffer has enough samples.

        Args:
            transition: Experience tuple containing (state, action, reward, next_state, done, info)

        Returns:
            Dictionary of metrics (loss, q_value, etc.)
        """
        # Add transition to replay buffer
        self.replay_buffer.add(transition)

        # Only train if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return {"loss": 0.0, "q_value": 0.0}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = (
            self.policy_network(states_tensor)
            .gather(1, actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(dim=1)[0]
            target_q_values = (
                rewards_tensor
                + (1 - dones_tensor) * self.discount_factor * next_q_values
            )

        # Compute Huber loss (smooth L1 loss)
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.gradient_clip_value
        )

        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        # Return metrics for logging
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
            "target_q": target_q_values.mean().item(),
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
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_dim": self.observation_dim,
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_frequency": self.target_update_frequency,
            "gradient_clip_value": self.gradient_clip_value,
            "training_steps": self.training_steps,
            "hidden_dims": self.policy_network.hidden_dims,
        }

        torch.save(state, path)

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: File path to load agent state from
        """
        state = torch.load(path, map_location=self.device)

        # Restore hyperparameters
        self.observation_dim = state["observation_dim"]
        self.action_space_size = state["action_space_size"]
        self.learning_rate = state["learning_rate"]
        self.discount_factor = state["discount_factor"]
        self.epsilon = state["epsilon"]
        self.epsilon_min = state["epsilon_min"]
        self.epsilon_decay = state["epsilon_decay"]
        self.batch_size = state["batch_size"]
        self.target_update_frequency = state["target_update_frequency"]
        self.gradient_clip_value = state["gradient_clip_value"]
        self.training_steps = state["training_steps"]

        # Restore network weights
        self.policy_network.load_state_dict(state["policy_network_state_dict"])
        self.target_network.load_state_dict(state["target_network_state_dict"])

        # Restore optimizer state
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
