"""
Training script for RL agents on flow environments.

This script provides a command-line interface for training RL agents
with configurable hyperparameters, environments, and logging.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

from src.environment.graph_generator import (
    GraphGenerator,
    GraphConfig as GraphGenConfig,
)
from src.environment.flow_environment import FlowEnvironment, EnvConfig as FlowEnvConfig
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.recurrent_ppo import RecurrentPPOAgent
from src.agents.exploration import (
    ExplorationStrategy,
    EpsilonGreedy,
    CountBasedBonus,
    CuriosityBonus,
)
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import TrainingConfig
from src.utils.reproducibility import (
    set_global_seeds,
    save_experiment_config,
    save_hyperparameters,
    create_experiment_metadata,
)


def create_exploration_strategy(
    exploration_type: str,
    observation_dim: int,
    action_dim: int,
    config: Dict[str, Any],
) -> ExplorationStrategy:
    """
    Create an exploration strategy based on type and configuration.

    Args:
        exploration_type: Type of exploration ('epsilon_greedy', 'count_based', 'curiosity')
        observation_dim: Dimension of observation space
        action_dim: Dimension of action space
        config: Exploration configuration dictionary

    Returns:
        Initialized exploration strategy
    """
    if exploration_type == "epsilon_greedy":
        return EpsilonGreedy(
            epsilon=config.get("epsilon", 1.0),
            epsilon_min=config.get("epsilon_min", 0.01),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            decay_mode=config.get("decay_mode", "episode"),
        )
    elif exploration_type == "count_based":
        return CountBasedBonus(
            beta=config.get("beta", 1.0),
            count_mode=config.get("count_mode", "state"),
        )
    elif exploration_type == "curiosity":
        return CuriosityBonus(
            state_dim=observation_dim,
            action_dim=action_dim,
            beta=config.get("beta", 0.1),
            learning_rate=config.get("learning_rate", 1e-3),
            hidden_dims=tuple(config.get("hidden_dims", [128, 128])),
            batch_size=config.get("batch_size", 32),
            train_frequency=config.get("train_frequency", 1),
            device=config.get("device", "cpu"),
        )
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")


def create_agent(
    agent_type: str,
    observation_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    exploration_strategy: ExplorationStrategy = None,
):
    """
    Create an agent based on type and configuration.

    Args:
        agent_type: Type of agent ('qlearning', 'dqn', 'ppo', 'recurrent_ppo')
        observation_dim: Dimension of observation space
        action_dim: Dimension of action space
        config: Agent configuration dictionary
        exploration_strategy: Optional exploration strategy to use

    Returns:
        Initialized agent
    """
    if agent_type == "qlearning":
        agent = QLearningAgent(
            action_space_size=action_dim,
            learning_rate=config.get("learning_rate", 0.1),
            discount_factor=config.get("discount_factor", 0.99),
            epsilon=config.get("epsilon", 1.0),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            epsilon_min=config.get("epsilon_min", 0.01),
        )
    elif agent_type == "dqn":
        agent = DQNAgent(
            observation_dim=observation_dim,
            action_space_size=action_dim,
            learning_rate=config.get("learning_rate", 0.001),
            discount_factor=config.get("discount_factor", 0.99),
            epsilon=config.get("epsilon", 1.0),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            epsilon_min=config.get("epsilon_min", 0.01),
            buffer_capacity=config.get("buffer_size", 10000),
            batch_size=config.get("batch_size", 32),
            target_update_frequency=config.get("target_update_frequency", 100),
            hidden_dims=tuple(config.get("hidden_dims", [128, 128])),
        )
    elif agent_type == "ppo":
        agent = PPOAgent(
            observation_dim=observation_dim,
            action_space_size=action_dim,
            learning_rate=config.get("learning_rate", 0.0003),
            discount_factor=config.get("discount_factor", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_epsilon=config.get("clip_epsilon", 0.2),
            entropy_coef=config.get("entropy_coef", 0.01),
            value_loss_coef=config.get("value_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            hidden_dims=tuple(config.get("hidden_dims", [128, 128])),
            num_epochs=config.get("ppo_epochs", 4),
            batch_size=config.get("batch_size", 64),
        )
    elif agent_type == "recurrent_ppo":
        agent = RecurrentPPOAgent(
            observation_dim=observation_dim,
            action_space_size=action_dim,
            learning_rate=config.get("learning_rate", 0.0003),
            discount_factor=config.get("discount_factor", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_epsilon=config.get("clip_epsilon", 0.2),
            entropy_coef=config.get("entropy_coef", 0.01),
            value_loss_coef=config.get("value_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            hidden_dim=config.get("hidden_dims", [128])[0],
            lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
            num_epochs=config.get("ppo_epochs", 4),
            batch_size=config.get("batch_size", 64),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Set exploration strategy if provided
    if exploration_strategy is not None:
        agent.exploration_strategy = exploration_strategy

    return agent


def create_environment(env_config: FlowEnvConfig, graph_config: GraphGenConfig):
    """
    Create a flow environment with generated graph.

    Args:
        env_config: Environment configuration
        graph_config: Graph generation configuration

    Returns:
        FlowEnvironment instance
    """
    generator = GraphGenerator()
    graph = generator.generate(graph_config)
    return FlowEnvironment(graph, env_config)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on flow environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Agent configuration
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        help="Type of agent to train",
    )

    # Environment configuration
    parser.add_argument(
        "--num-nodes", type=int, default=10, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--branching-factor",
        type=float,
        default=2.0,
        help="Branching factor for graph generation",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5, help="Maximum depth of the graph"
    )
    parser.add_argument(
        "--popup-probability",
        type=float,
        default=0.1,
        help="Probability of popup failures",
    )
    parser.add_argument(
        "--num-dependencies", type=int, default=2, help="Number of hidden dependencies"
    )

    # Training configuration
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--discount-factor", type=float, default=0.99, help="Discount factor (gamma)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval-frequency", type=int, default=100, help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory for logs and checkpoints"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name for this experiment"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=500,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--log-frequency", type=int, default=10, help="Log metrics every N episodes"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (overrides other args)",
    )

    # Agent-specific hyperparameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy exploration",
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Replay buffer size (DQN)"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer dimensions",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config file if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        config_dict = load_config_file(args.config)

        # Handle nested config structure from ablation scripts
        if "environment" in config_dict:
            for key, value in config_dict["environment"].items():
                if hasattr(args, key):
                    setattr(args, key, value)

        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(args, key):
                    setattr(args, key, value)

        if "agent_config" in config_dict:
            args.agent_config = config_dict["agent_config"]

        # Override args with top-level config file values
        for key, value in config_dict.items():
            if key not in [
                "environment",
                "training",
                "agent_config",
                "exploration_config",
            ]:
                if hasattr(args, key):
                    setattr(args, key, value)

        # Handle exploration config
        if "exploration" in config_dict:
            args.exploration = config_dict["exploration"]
        if "exploration_config" in config_dict:
            args.exploration_config = config_dict["exploration_config"]

    # Set random seeds for reproducibility
    print(f"Setting random seed to {args.seed}")
    set_global_seeds(args.seed, deterministic=True)

    # Create configurations
    # Use fixed_max_actions to ensure consistent observation space across environments
    env_config = FlowEnvConfig(
        max_steps=args.max_steps,
        random_seed=args.seed,
        fixed_max_actions=5,  # Fixed to handle graphs with varying out-degrees
    )

    graph_config = GraphGenConfig(
        num_nodes=args.num_nodes,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        popup_probability=getattr(args, "popup_probability", 0.1),
        num_dependencies=getattr(args, "num_dependencies", 2),
        random_seed=args.seed,
    )

    training_config = TrainingConfig(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        random_seed=args.seed,
        checkpoint_frequency=args.checkpoint_frequency,
        log_frequency=args.log_frequency,
    )

    # Create environment
    print("Creating training environment...")
    env = create_environment(env_config, graph_config)

    # Create validation environments
    print("Creating validation environments...")
    val_envs = []
    for i in range(3):
        val_graph_config = GraphGenConfig(
            num_nodes=args.num_nodes,
            branching_factor=args.branching_factor,
            max_depth=args.max_depth,
            popup_probability=getattr(args, "popup_probability", 0.1),
            num_dependencies=getattr(args, "num_dependencies", 2),
            random_seed=args.seed + 1000 + i,
        )
        val_env = create_environment(env_config, val_graph_config)
        val_envs.append(val_env)

    # Get observation and action dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    # Create exploration strategy if specified
    exploration_strategy = None
    exploration_config = {}

    if hasattr(args, "exploration") and args.exploration:
        print(f"Creating {args.exploration} exploration strategy...")

        # Get exploration config from args or config file
        if hasattr(args, "exploration_config"):
            exploration_config = args.exploration_config

        exploration_strategy = create_exploration_strategy(
            args.exploration,
            observation_dim,
            action_dim,
            exploration_config,
        )

    # Create agent
    print(f"Creating {args.agent} agent...")
    agent_config = {
        "learning_rate": args.learning_rate,
        "discount_factor": args.discount_factor,
        "batch_size": args.batch_size,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "buffer_size": args.buffer_size,
        "hidden_dims": args.hidden_dims,
    }

    # Merge agent_config from config file if present
    if hasattr(args, "agent_config"):
        agent_config.update(args.agent_config)

    agent = create_agent(
        args.agent,
        observation_dim,
        action_dim,
        agent_config,
        exploration_strategy,
    )

    # Create log directory
    log_dir = Path(args.log_dir)
    if args.experiment_name:
        log_dir = log_dir / args.experiment_name
    else:
        log_dir = log_dir / f"{args.agent}_seed{args.seed}"

    # Save experiment configuration and hyperparameters
    print("Saving experiment configuration...")
    experiment_metadata = create_experiment_metadata(
        agent_type=args.agent,
        env_config=env_config.__dict__,
        training_config=training_config.__dict__,
        agent_config=agent_config,
    )

    # Add exploration config to metadata if present
    if exploration_strategy is not None:
        experiment_metadata["exploration"] = {
            "type": args.exploration if hasattr(args, "exploration") else "none",
            "config": exploration_config,
        }

    save_experiment_config(experiment_metadata, str(log_dir))
    save_hyperparameters(agent_config, str(log_dir))

    # Create training orchestrator
    print(f"Initializing training orchestrator...")
    print(f"Logs will be saved to: {log_dir}")
    orchestrator = TrainingOrchestrator(
        agent=agent,
        env=env,
        config=training_config,
        val_envs=val_envs,
        log_dir=str(log_dir),
        experiment_name=args.experiment_name,
        exploration_strategy=exploration_strategy,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    try:
        results = orchestrator.train()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total episodes: {results['total_episodes']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Best validation reward: {results['best_val_reward']:.2f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Results saved to: {log_dir}")
        print("=" * 60 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
