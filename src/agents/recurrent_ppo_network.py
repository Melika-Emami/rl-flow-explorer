"""Recurrent Actor-Critic network architecture for PPO agent with LSTM."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RecurrentActorCriticNetwork(nn.Module):
    """
    Recurrent Actor-Critic network with LSTM for partial observability.

    Architecture:
    - Input embedding layer
    - LSTM layer for temporal processing
    - Policy head (actor): outputs action probabilities
    - Value head (critic): outputs state value estimate
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
    ):
        """
        Initialize recurrent actor-critic network.

        Args:
            observation_dim: Dimension of observation space
            action_space_size: Number of possible actions
            hidden_dim: Dimension of input embedding layer
            lstm_hidden_dim: Dimension of LSTM hidden state (64-128 units)
        """
        super(RecurrentActorCriticNetwork, self).__init__()

        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Input embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU()
        )

        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Policy head (actor): outputs action logits
        self.policy_head = nn.Linear(lstm_hidden_dim, action_space_size)

        # Value head (critic): outputs state value estimate
        self.value_head = nn.Linear(lstm_hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
            elif "weight" in name and "lstm" not in name:
                nn.init.orthogonal_(param.data, gain=1.0)

    def forward(
        self,
        observation: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network with hidden state management.

        Args:
            observation: Batch of observations
                - Shape: [batch_size, observation_dim] for single step
                - Shape: [batch_size, seq_len, observation_dim] for sequences
            hidden_state: Optional tuple of (h, c) LSTM hidden states
                - h: [1, batch_size, lstm_hidden_dim]
                - c: [1, batch_size, lstm_hidden_dim]

        Returns:
            Tuple of (action_logits, state_values, new_hidden_state)
            - action_logits: [batch_size, action_space_size] or [batch_size, seq_len, action_space_size]
            - state_values: [batch_size, 1] or [batch_size, seq_len, 1]
            - new_hidden_state: Tuple of (h, c) for next step
        """
        # Handle single step vs sequence input
        is_single_step = len(observation.shape) == 2
        if is_single_step:
            # Add sequence dimension: [batch_size, observation_dim] -> [batch_size, 1, observation_dim]
            observation = observation.unsqueeze(1)

        batch_size = observation.shape[0]

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size, observation.device)

        # Input embedding
        embedded = self.embedding(observation)

        # LSTM forward pass
        lstm_out, new_hidden_state = self.lstm(embedded, hidden_state)

        # Policy head: action logits
        action_logits = self.policy_head(lstm_out)

        # Value head: state value estimate
        state_values = self.value_head(lstm_out)

        # Remove sequence dimension if input was single step
        if is_single_step:
            action_logits = action_logits.squeeze(1)  # [batch_size, action_space_size]
            state_values = state_values.squeeze(1)  # [batch_size, 1]

        return action_logits, state_values, new_hidden_state

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h, c) initialized to zeros
            - h: [1, batch_size, lstm_hidden_dim]
            - c: [1, batch_size, lstm_hidden_dim]
        """
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return (h, c)

    def get_action_distribution(
        self,
        observation: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.distributions.Categorical, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action probability distribution for given observation.

        Args:
            observation: Batch of observations [batch_size, observation_dim]
            hidden_state: Optional LSTM hidden state

        Returns:
            Tuple of (distribution, new_hidden_state)
            - distribution: Categorical distribution over actions
            - new_hidden_state: Updated LSTM hidden state
        """
        action_logits, _, new_hidden_state = self.forward(observation, hidden_state)
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist, new_hidden_state

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Evaluate actions for given observations with hidden state.

        Used during training to compute log probabilities and entropy.

        Args:
            observation: Batch of observations
                - Shape: [batch_size, observation_dim] or [batch_size, seq_len, observation_dim]
            actions: Batch of actions
                - Shape: [batch_size] or [batch_size, seq_len]
            hidden_state: Optional LSTM hidden state

        Returns:
            Tuple of (log_probs, state_values, entropy, new_hidden_state)
            - log_probs: Log probabilities of actions
            - state_values: State value estimates
            - entropy: Entropy of action distribution
            - new_hidden_state: Updated LSTM hidden state
        """
        action_logits, state_values, new_hidden_state = self.forward(
            observation, hidden_state
        )

        # Handle sequence vs single step
        is_sequence = len(actions.shape) > 1

        if is_sequence:
            # Reshape for sequence processing
            batch_size, seq_len = actions.shape
            action_logits_flat = action_logits.view(-1, self.action_space_size)
            actions_flat = actions.view(-1)

            # Create categorical distribution
            dist = torch.distributions.Categorical(logits=action_logits_flat)

            # Compute log probabilities and entropy
            log_probs = dist.log_prob(actions_flat).view(batch_size, seq_len)
            entropy = dist.entropy().view(batch_size, seq_len)
        else:
            # Single step processing
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_probs, state_values, entropy, new_hidden_state
