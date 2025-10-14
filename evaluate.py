"""
Evaluation script for trained RL agents.

This script provides a command-line interface for evaluating trained agents
on test environments with configurable parameters.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

from src.environment.graph_generator import GraphGenerator, GraphConfig
from src.environment.flow_environment import FlowEnvironment, EnvConfig
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.recurrent_ppo import RecurrentPPOAgent
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import MetricsCollector
from src.utils.reproducibility import set_global_seeds, save_experiment_config


def load_agent(
    agent_type: str, checkpoint_path: str, observation_dim: int, action_dim: int
):
    """
    Load a trained agent from checkpoint.

    Args:
        agent_type: Type of agent ('qlearning', 'dqn', 'ppo', 'recurrent_ppo')
        checkpoint_path: Path to checkpoint file
        observation_dim: Dimension of observation space
        action_dim: Dimension of action space

    Returns:
        Loaded agent
    """
    # Create agent with default config (will be overwritten by load)
    if agent_type == "qlearning":
        agent = QLearningAgent(observation_dim, action_dim)
    elif agent_type == "dqn":
        agent = DQNAgent(observation_dim, action_dim)
    elif agent_type == "ppo":
        agent = PPOAgent(observation_dim, action_dim)
    elif agent_type == "recurrent_ppo":
        agent = RecurrentPPOAgent(observation_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Load checkpoint
    print(f"Loading agent from {checkpoint_path}")
    agent.load(checkpoint_path)

    return agent


def create_test_environments(
    base_config: EnvConfig,
    graph_config: GraphConfig,
    num_envs: int,
    seed_offset: int = 10000,
) -> List[FlowEnvironment]:
    """
    Create multiple test environments with different random seeds.

    Args:
        base_config: Base environment configuration
        graph_config: Base graph configuration
        num_envs: Number of environments to create
        seed_offset: Offset for random seeds

    Returns:
        List of FlowEnvironment instances
    """
    envs = []
    generator = GraphGenerator()

    for i in range(num_envs):
        # Create graph config with different seed
        test_graph_config = GraphConfig(
            num_nodes=graph_config.num_nodes,
            branching_factor=graph_config.branching_factor,
            max_depth=graph_config.max_depth,
            popup_probability=graph_config.popup_probability,
            num_dependencies=graph_config.num_dependencies,
            random_seed=graph_config.random_seed + seed_offset + i,
        )

        # Generate graph and create environment
        graph = generator.generate(test_graph_config)
        env = FlowEnvironment(graph, base_config)
        envs.append(env)

    return envs


def create_ood_environments(
    base_config: EnvConfig,
    graph_config: GraphConfig,
    num_envs: int,
    shift_type: str,
    seed_offset: int = 20000,
) -> List[FlowEnvironment]:
    """
    Create out-of-distribution test environments with distribution shifts.

    Args:
        base_config: Base environment configuration
        graph_config: Base graph configuration
        num_envs: Number of environments to create
        shift_type: Type of distribution shift ('deeper', 'higher_popup', 'different_topology')
        seed_offset: Offset for random seeds

    Returns:
        List of FlowEnvironment instances
    """
    envs = []
    generator = GraphGenerator()

    for i in range(num_envs):
        # Modify config based on shift type
        if shift_type == "deeper":
            test_graph_config = GraphConfig(
                num_nodes=graph_config.num_nodes * 2,
                branching_factor=graph_config.branching_factor,
                max_depth=graph_config.max_depth * 2,
                popup_probability=graph_config.popup_probability,
                num_dependencies=graph_config.num_dependencies,
                random_seed=graph_config.random_seed + seed_offset + i,
            )
        elif shift_type == "higher_popup":
            test_graph_config = GraphConfig(
                num_nodes=graph_config.num_nodes,
                branching_factor=graph_config.branching_factor,
                max_depth=graph_config.max_depth,
                popup_probability=min(graph_config.popup_probability * 2, 0.5),
                num_dependencies=graph_config.num_dependencies,
                random_seed=graph_config.random_seed + seed_offset + i,
            )
        elif shift_type == "different_topology":
            test_graph_config = GraphConfig(
                num_nodes=graph_config.num_nodes,
                branching_factor=graph_config.branching_factor * 1.5,
                max_depth=graph_config.max_depth,
                popup_probability=graph_config.popup_probability,
                num_dependencies=graph_config.num_dependencies * 2,
                random_seed=graph_config.random_seed + seed_offset + i,
            )
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")

        # Generate graph and create environment
        graph = generator.generate(test_graph_config)
        env = FlowEnvironment(graph, base_config)
        envs.append(env)

    return envs


def save_results(results, output_path: str):
    """Save evaluation results to JSON file."""
    output_dict = {
        "agent_name": results.agent_name,
        "environment_name": results.environment_name,
        "num_episodes": results.num_episodes,
        "metrics": {
            "success_rate": float(results.metrics.success_rate),
            "success_rate_std": float(results.metrics.success_rate_std),
            "success_rate_ci": [float(x) for x in results.metrics.success_rate_ci],
            "mean_steps_to_success": float(results.metrics.mean_steps_to_success),
            "std_steps_to_success": float(results.metrics.std_steps_to_success),
            "failure_rate": float(results.metrics.failure_rate),
            "mean_coverage": float(results.metrics.mean_coverage),
            "std_coverage": float(results.metrics.std_coverage),
            "mean_reward": float(results.metrics.mean_reward),
            "std_reward": float(results.metrics.std_reward),
            "mean_episode_length": float(results.metrics.mean_episode_length),
            "num_episodes": results.metrics.num_episodes,
            "num_successes": results.metrics.num_successes,
            "num_failures": results.metrics.num_failures,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"Results saved to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Agent configuration
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        help="Type of agent",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to agent checkpoint"
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
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--step-penalty", type=float, default=-0.01, help="Penaly for each taken step"
    )
    parser.add_argument(
        "--success-reward", type=float, default=1.0, help="Reward of success"
    )
    parser.add_argument(
        "--failure-penalty", type=float, default=-1.0, help="Penalty of failure"
    )
    parser.add_argument(
        "--partial-observability",
        type=bool,
        default=True,
        help="Limit observation to local info",
    )
    parser.add_argument(
        "--fixed-max-actions",
        type=int,
        default=5,
        help="Fixed max actions for consistent obs space",
    )

    # Evaluation configuration
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--num-test-envs", type=int, default=5, help="Number of test environments"
    )

    # Out-of-distribution evaluation
    parser.add_argument(
        "--eval-ood",
        action="store_true",
        help="Evaluate on out-of-distribution environments",
    )
    parser.add_argument(
        "--ood-shift",
        type=str,
        default="deeper",
        choices=["deeper", "higher_popup", "different_topology"],
        help="Type of distribution shift for OOD evaluation",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name for this evaluation"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seeds for reproducibility
    print(f"Setting random seed to {args.seed}")
    set_global_seeds(args.seed, deterministic=True)

    # Create configurations
    env_config = EnvConfig(
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
        max_steps=args.max_steps,
        partial_observability=args.partial_observability,
        random_seed=args.seed,
        fixed_max_actions=args.fixed_max_actions,
    )

    graph_config = GraphConfig(
        num_nodes=args.num_nodes,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        popup_probability=args.popup_probability,
        num_dependencies=args.num_dependencies,
        random_seed=args.seed,
    )

    # Create a temporary environment to get dimensions
    print("Creating temporary environment to get dimensions...")
    generator = GraphGenerator()
    temp_graph = generator.generate(graph_config)
    temp_env = FlowEnvironment(temp_graph, env_config)
    observation_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n

    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    # Load agent
    agent = load_agent(args.agent, args.checkpoint, observation_dim, action_dim)

    # Create test environments
    print(f"\nCreating {args.num_test_envs} test environments...")
    test_envs = create_test_environments(env_config, graph_config, args.num_test_envs)

    # Create evaluator
    evaluator = Evaluator(verbose=args.verbose)

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.experiment_name:
        output_dir = output_dir / args.experiment_name
    else:
        output_dir = output_dir / f"{args.agent}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation configuration
    eval_config = {
        "agent_type": args.agent,
        "checkpoint": args.checkpoint,
        "environment": env_config.__dict__,
        "num_episodes": args.num_episodes,
        "num_test_envs": args.num_test_envs,
        "eval_ood": args.eval_ood,
        "ood_shift": args.ood_shift if args.eval_ood else None,
        "seed": args.seed,
    }
    save_experiment_config(eval_config, str(output_dir), "evaluation_config.json")

    # Evaluate on in-distribution test set
    print("\n" + "=" * 60)
    print("EVALUATING ON IN-DISTRIBUTION TEST SET")
    print("=" * 60 + "\n")

    results = evaluator.evaluate(
        agent=agent,
        test_envs=test_envs,
        num_episodes=args.num_episodes,
        agent_name=args.agent,
        environment_name="in_distribution",
    )

    # Save results
    save_results(results, str(output_dir / "in_distribution_results.json"))

    # Evaluate on out-of-distribution test set if requested
    if args.eval_ood:
        print("\n" + "=" * 60)
        print(f"EVALUATING ON OUT-OF-DISTRIBUTION TEST SET ({args.ood_shift})")
        print("=" * 60 + "\n")

        ood_envs = create_ood_environments(
            env_config, graph_config, args.num_test_envs, args.ood_shift
        )

        ood_results = evaluator.evaluate(
            agent=agent,
            test_envs=ood_envs,
            num_episodes=args.num_episodes,
            agent_name=args.agent,
            environment_name=f"ood_{args.ood_shift}",
        )

        # Save OOD results
        save_results(
            ood_results, str(output_dir / f"ood_{args.ood_shift}_results.json")
        )

        # Compute and display performance degradation
        print("\n" + "=" * 60)
        print("PERFORMANCE DEGRADATION")
        print("=" * 60)

        in_dist_success = results.metrics.success_rate
        ood_success = ood_results.metrics.success_rate
        degradation = (
            ((in_dist_success - ood_success) / in_dist_success) * 100
            if in_dist_success > 0
            else 0
        )

        print(f"In-Distribution Success Rate: {in_dist_success:.2%}")
        print(f"Out-of-Distribution Success Rate: {ood_success:.2%}")
        print(f"Performance Degradation: {degradation:.1f}%")
        print("=" * 60 + "\n")

    print(f"\nEvaluation complete! Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
