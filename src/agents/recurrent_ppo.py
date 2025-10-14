"""Recurrent Proximal Policy Optimization (PPO) agent with LSTM for partial observability.

This module implements a recurrent version of PPO that uses LSTM to handle partial observability
in environments where the current observation alone is insufficient to determine the optimal action.

Key Features:
- LSTM-based actor-critic network for temporal processing
- Hidden state tracking across episode steps
- Automatic hidden state reset at episode boundaries
- Truncated backpropagation through time (TBPTT) for efficient training
- Proper sequence handling during both action selection and training

The agent maintains a hidden state throughout an episode, allowing it to remember past observations
and make better decisions in partially observable environments. The hidden state is automatically
reset when an episode ends (done=True).

During training, the agent uses truncated BPTT to process long episodes in manageable chunks,
preventing memory issues and improving training stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.agents.base import Agent
from src.agents.recurrent_ppo_network import RecurrentActorCriticNetwork
from src.utils.data_models import Transition


class RecurrentPPOAgent(Agent):
    """
    Recurrent PPO agent with LSTM for handling partial observability.

    Extends PPO with:
    - LSTM-based actor-critic network for temporal processing
    - Hidden state tracking across episode steps
    - Truncated backpropagation through time
    - Episode boundary handling in sequence processing
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
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        truncation_length: int = 32,
        device: str = None,
    ):
        """
        Initialize Recurrent PPO agent.

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
            hidden_dim: Dimension of input embedding layer
            lstm_hidden_dim: Dimension of LSTM hidden state (64-128 units)
            truncation_length: Length for truncated backpropagation through time
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
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.truncation_length = truncation_length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize recurrent actor-critic network
        self.network = RecurrentActorCriticNetwork(
            observation_dim, action_space_size, hidden_dim, lstm_hidden_dim
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Storage for trajectory data
        self.trajectory_buffer: List[Transition] = []

        # Hidden state tracking for current episode
        self.current_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Track training steps
        self.training_steps = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action by sampling from policy distribution while maintaining hidden state.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode

        Returns:
            Selected action as integer
        """
        with torch.no_grad():
            # Convert observation to tensor
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Get action distribution and update hidden state
            dist, self.current_hidden_state = self.network.get_action_distribution(
                state_tensor, self.current_hidden_state
            )

            # Sample action from distribution
            action = dist.sample()

            return action.item()

    def reset_hidden_state(self) -> None:
        """
        Reset hidden state at the beginning of a new episode.

        Should be called when starting a new episode.
        """
        self.current_hidden_state = None

    def update(self, transition: Transition) -> Dict[str, float]:
        """
        Store transition and perform PPO update when episode is done.

        Args:
            transition: Experience tuple containing (state, action, reward, next_state, done, info)

        Returns:
            Dictionary of metrics (loss, policy_loss, value_loss, entropy)
        """
        # Add transition to trajectory buffer
        self.trajectory_buffer.append(transition)

        # Reset hidden state at episode boundaries
        if transition.done:
            self.reset_hidden_state()

            # Perform PPO update on collected trajectory
            metrics = self._ppo_update()

            # Clear trajectory buffer
            self.trajectory_buffer = []

            return metrics

        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectories with truncated BPTT.

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

        # Normalize advantages for stability (only if we have more than one sample)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # For single sample, just center it
            advantages = advantages - advantages.mean()

        # Get old log probabilities and values with fresh hidden state
        with torch.no_grad():
            old_log_probs, old_values = self._evaluate_trajectory(
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
            # Process trajectory in truncated sequences
            for start_idx in range(0, len(states), self.truncation_length):
                end_idx = min(start_idx + self.truncation_length, len(states))

                # Get sequence data
                seq_states = states_tensor[start_idx:end_idx]
                seq_actions = actions_tensor[start_idx:end_idx]
                seq_advantages = advantages[start_idx:end_idx]
                seq_returns = returns[start_idx:end_idx]
                seq_old_log_probs = old_log_probs[start_idx:end_idx]

                # Evaluate actions with current policy (detach hidden state between sequences)
                log_probs, values, entropy = self._evaluate_sequence(
                    seq_states, seq_actions
                )
                values = values.squeeze(-1)

                # Compute ratio for PPO objective
                ratio = torch.exp(log_probs - seq_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * seq_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * seq_advantages
                )

                # Policy loss: negative of clipped surrogate objective
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: MSE between predicted and target values
                value_loss = nn.functional.mse_loss(values, seq_returns)

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

    def _evaluate_trajectory(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate entire trajectory with LSTM processing.

        Args:
            states: Tensor of states [trajectory_length, observation_dim]
            actions: Tensor of actions [trajectory_length]

        Returns:
            Tuple of (log_probs, values)
        """
        # Add batch dimension and process as sequence
        states_seq = states.unsqueeze(0)  # [1, seq_len, obs_dim]
        actions_seq = actions.unsqueeze(0)  # [1, seq_len]

        # Initialize hidden state
        hidden_state = self.network.init_hidden_state(1, self.device)

        # Evaluate actions
        log_probs, values, entropy, _ = self.network.evaluate_actions(
            states_seq, actions_seq, hidden_state
        )

        # Remove batch dimension
        log_probs = log_probs.squeeze(0)  # [seq_len]
        values = values.squeeze(0)  # [seq_len, 1]

        return log_probs, values

    def _evaluate_sequence(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a sequence of states and actions with fresh hidden state.

        Used during training updates with truncated BPTT.

        Args:
            states: Tensor of states [seq_len, observation_dim]
            actions: Tensor of actions [seq_len]

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        # Add batch dimension
        states_seq = states.unsqueeze(0)  # [1, seq_len, obs_dim]
        actions_seq = actions.unsqueeze(0)  # [1, seq_len]

        # Initialize fresh hidden state for this sequence
        hidden_state = self.network.init_hidden_state(1, self.device)

        # Evaluate actions
        log_probs, values, entropy, _ = self.network.evaluate_actions(
            states_seq, actions_seq, hidden_state
        )

        # Remove batch dimension
        log_probs = log_probs.squeeze(0)  # [seq_len]
        values = values.squeeze(0)  # [seq_len, 1]
        entropy = entropy.squeeze(0)  # [seq_len]

        return log_probs, values, entropy

    def _compute_gae(
        self, states: torch.Tensor, rewards: np.ndarray, dones: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) using LSTM value function.

        Args:
            states: Tensor of states [trajectory_length, observation_dim]
            rewards: Array of rewards [trajectory_length]
            dones: Array of done flags [trajectory_length]

        Returns:
            Tuple of (advantages, returns) as tensors
        """
        with torch.no_grad():
            # Get value estimates for all states using LSTM
            # Add batch dimension and process as sequence
            states_seq = states.unsqueeze(0)  # [1, seq_len, obs_dim]

            # Initialize hidden state
            hidden_state = self.network.init_hidden_state(1, self.device)

            # Get values from network
            _, values, _ = self.network.forward(states_seq, hidden_state)

            # Remove batch dimension
            values = values.squeeze(0).squeeze(-1).cpu().numpy()  # [seq_len]

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
            "hidden_dim": self.hidden_dim,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "truncation_length": self.truncation_length,
            "training_steps": self.training_steps,
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
        self.hidden_dim = state["hidden_dim"]
        self.lstm_hidden_dim = state["lstm_hidden_dim"]
        self.truncation_length = state["truncation_length"]
        self.training_steps = state["training_steps"]

        # Recreate network with loaded parameters
        self.network = RecurrentActorCriticNetwork(
            self.observation_dim,
            self.action_space_size,
            self.hidden_dim,
            self.lstm_hidden_dim,
        ).to(self.device)

        # Restore network weights
        self.network.load_state_dict(state["network_state_dict"])

        # Restore optimizer state
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Reset hidden state
        self.reset_hidden_state()
