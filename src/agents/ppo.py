"""Proximal Policy Optimization (PPO) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
from pathlib import Path

from src.agents.base import Agent
from src.agents.ppo_network import ActorCriticNetwork
from src.utils.data_models import Transition


class PPOAgent(Agent):
    """
    Proximal Policy Optimization agent with actor-critic architecture.

    Implements PPO algorithm with:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization for exploration
    - Mini-batch updates over collected trajectories
    - Value function clipping for stability
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        batch_size: int = 64,
        hidden_dims: tuple = (128, 128),
        device: str = None,
    ):
        """
        Initialize PPO agent.

        Args:
            observation_dim: Dimension of observation space
            action_space_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma) for future rewards
            gae_lambda: Lambda parameter for GAE
            clip_epsilon: Clipping parameter for PPO objective
            value_loss_coef: Coefficient for value loss in total loss
            entropy_coef: Coefficient for entropy bonus in total loss
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs to train on each batch of trajectories
            batch_size: Mini-batch size for updates
            hidden_dims: Tuple of hidden layer dimensions
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize actor-critic network
        self.network = ActorCriticNetwork(
            observation_dim, action_space_size, hidden_dims
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Storage for trajectory data
        self.trajectory_buffer: List[Transition] = []

        # Track training steps
        self.training_steps = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action by sampling from policy distribution.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode

        Returns:
            Selected action as integer
        """
        with torch.no_grad():
            # Convert observation to tensor
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Get action distribution from policy
            dist = self.network.get_action_distribution(state_tensor)

            # Sample action from distribution
            action = dist.sample()

            return action.item()

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Store transition and perform PPO update when enough data is collected.

        PPO collects trajectories and performs mini-batch updates.
        This method stores transitions and triggers updates periodically.

        Args:
            transition: Experience tuple containing (state, action, reward, next_state, done, info)

        Returns:
            Dictionary of metrics (loss, policy_loss, value_loss, entropy)
        """
        # Add transition to trajectory buffer
        self.trajectory_buffer.append(transition)

        # Only update when episode is done or buffer is full
        # For simplicity, we update at episode boundaries
        if not transition.done:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Perform PPO update on collected trajectory
        metrics = self._ppo_update()

        # Clear trajectory buffer
        self.trajectory_buffer = []

        return metrics

    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectories.

        Returns:
            Dictionary of training metrics
        """
        if len(self.trajectory_buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Extract data from trajectory buffer
        states = np.array([t.state for t in self.trajectory_buffer])
        actions = np.array([t.action for t in self.trajectory_buffer])
        rewards = np.array([t.reward for t in self.trajectory_buffer])
        dones = np.array([t.done for t in self.trajectory_buffer])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(states_tensor, rewards, dones)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old log probabilities and values
        with torch.no_grad():
            old_log_probs, old_values, _ = self.network.evaluate_actions(
                states_tensor, actions_tensor
            )
            old_log_probs = old_log_probs.detach()
            old_values = old_values.squeeze(-1).detach()

        # Perform multiple epochs of mini-batch updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            # Generate random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Evaluate actions with current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                values = values.squeeze(-1)

                # Compute ratio for PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )

                # Policy loss: negative of clipped surrogate objective
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: MSE between predicted and target values
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus for exploration
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                num_updates += 1

        self.training_steps += 1

        # Return average metrics
        return {
            "loss": total_loss / num_updates if num_updates > 0 else 0.0,
            "policy_loss": total_policy_loss / num_updates if num_updates > 0 else 0.0,
            "value_loss": total_value_loss / num_updates if num_updates > 0 else 0.0,
            "entropy": total_entropy / num_updates if num_updates > 0 else 0.0,
        }

    def _compute_gae(
        self, states: torch.Tensor, rewards: np.ndarray, dones: np.ndarray
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            states: Tensor of states [trajectory_length, observation_dim]
            rewards: Array of rewards [trajectory_length]
            dones: Array of done flags [trajectory_length]

        Returns:
            Tuple of (advantages, returns) as tensors
        """
        with torch.no_grad():
            # Get value estimates for all states
            _, values = self.network.forward(states)
            values = values.squeeze(-1).cpu().numpy()

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = (
                rewards[t]
                + self.discount_factor * next_value * (1 - dones[t])
                - values[t]
            )

            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            advantages[t] = last_gae = (
                delta
                + self.discount_factor * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        # Returns are advantages + values
        returns = advantages + values

        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        return advantages_tensor, returns_tensor

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: File path to save agent state
        """
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_dim": self.observation_dim,
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "training_steps": self.training_steps,
            "hidden_dims": self.network.hidden_dims,
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
        self.gae_lambda = state["gae_lambda"]
        self.clip_epsilon = state["clip_epsilon"]
        self.value_loss_coef = state["value_loss_coef"]
        self.entropy_coef = state["entropy_coef"]
        self.max_grad_norm = state["max_grad_norm"]
        self.num_epochs = state["num_epochs"]
        self.batch_size = state["batch_size"]
        self.training_steps = state["training_steps"]

        # Restore network weights
        self.network.load_state_dict(state["network_state_dict"])

        # Restore optimizer state
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
