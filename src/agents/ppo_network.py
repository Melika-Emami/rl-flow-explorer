"""Actor-Critic network architecture for PPO agent."""

import torch
import torch.nn as nn
from typing import Tuple


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared feature extractor.

    Architecture:
    - Shared MLP backbone (feature extractor)
    - Policy head (actor): outputs action probabilities
    - Value head (critic): outputs state value estimate
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ):
        """
        Initialize actor-critic network.

        Args:
            observation_dim: Dimension of observation space
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions for shared backbone
        """
        super(ActorCriticNetwork, self).__init__()

        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.hidden_dims = hidden_dims

        # Shared feature extractor (MLP backbone)
        layers = []
        input_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_backbone = nn.Sequential(*layers)

        # Policy head (actor): outputs action logits
        self.policy_head = nn.Linear(hidden_dims[-1], action_space_size)

        # Value head (critic): outputs state value estimate
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            observation: Batch of observations [batch_size, observation_dim]

        Returns:
            Tuple of (action_logits, state_values)
            - action_logits: [batch_size, action_space_size]
            - state_values: [batch_size, 1]
        """
        # Shared feature extraction
        features = self.shared_backbone(observation)

        # Policy head: action logits (unnormalized log probabilities)
        action_logits = self.policy_head(features)

        # Value head: state value estimate
        state_values = self.value_head(features)

        return action_logits, state_values

    def get_action_distribution(
        self, observation: torch.Tensor
    ) -> torch.distributions.Categorical:
        """
        Get action probability distribution for given observation.

        Args:
            observation: Batch of observations [batch_size, observation_dim]

        Returns:
            Categorical distribution over actions
        """
        action_logits, _ = self.forward(observation)
        return torch.distributions.Categorical(logits=action_logits)

    def evaluate_actions(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.

        Used during training to compute log probabilities and entropy.

        Args:
            observation: Batch of observations [batch_size, observation_dim]
            actions: Batch of actions [batch_size]

        Returns:
            Tuple of (log_probs, state_values, entropy)
            - log_probs: Log probabilities of actions [batch_size]
            - state_values: State value estimates [batch_size, 1]
            - entropy: Entropy of action distribution [batch_size]
        """
        action_logits, state_values = self.forward(observation)

        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)

        # Compute log probabilities of taken actions
        log_probs = dist.log_prob(actions)

        # Compute entropy for exploration bonus
        entropy = dist.entropy()

        return log_probs, state_values, entropy
