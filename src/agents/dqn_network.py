"""Neural network architecture for DQN agent."""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Multi-layer perceptron for Deep Q-Network.

    Architecture:
    - Input layer: observation_dim
    - Hidden layers: 2-3 layers with 128-256 units each
    - ReLU activations between layers
    - Output layer: action_space_size (Q-values for each action)
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        hidden_dims: tuple = (256, 256),
    ):
        """
        Initialize DQN network.

        Args:
            observation_dim: Dimension of observation space
            action_space_size: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions (default: (256, 256))
        """
        super(DQNNetwork, self).__init__()

        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.hidden_dims = hidden_dims

        # Build network layers
        layers = []
        input_dim = observation_dim

        # Add hidden layers with ReLU activations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Add output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_dim, action_space_size))

        # Create sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, observation_dim)

        Returns:
            Q-values tensor of shape (batch_size, action_space_size)
        """
        return self.network(x)
