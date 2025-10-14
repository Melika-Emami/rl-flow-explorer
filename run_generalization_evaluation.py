"""
Run Generalization Evaluation

This script evaluates all trained agents on unseen graphs with distribution shifts.
It loads trained models from baseline, exploration, and memory ablation experiments,
tests them on in-distribution and out-of-distribution test sets, and measures
performance degradation.

Usage:
    # Evaluate all trained agents
    python run_generalization_evaluation.py

    # Evaluate specific agents
    python run_generalization_evaluation.py --agents dqn ppo

    # Use custom test configuration
    python run_generalization_evaluation.py --num-test-graphs 20 --episodes-per-env 20

    # Save detailed results
    python run_generalization_evaluation.py --save-trajectories
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.environment.graph_generator import GraphGenerator, GraphConfig
from src.environment.flow_environment import FlowEnvironment, EnvConfig
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.recurrent_ppo import RecurrentPPOAgent
from src.evaluation.generalization import GeneralizationEvaluator, GeneralizationResults
from src.visualization.plot_generator import PlotGenerator


def find_trained_models(base_dirs: List[str]) -> Dict[str, List[Path]]:
    """
    Find all trained model checkpoints in the specified directories.

    Args:
        base_dirs: List of base directories to search

    Returns:
        Dictionary mapping agent names to lists of checkpoint paths
    """
    models = {}

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        # Search for checkpoint directories
        for root, dirs, files in os.walk(base_dir):
            if "checkpoints" in dirs:
                checkpoint_dir = Path(root) / "checkpoints"

                # Look for final model
                final_model = checkpoint_dir / "final_model.pt"
                if final_model.exists():
                    # Extract agent name from path
                    agent_name = extract_agent_name(root)
                    if agent_name not in models:
                        models[agent_name] = []
                    models[agent_name].append(final_model)

    return models


def extract_agent_name(path: str) -> str:
    """
    Extract agent name from checkpoint path.

    Args:
        path: Path string containing agent information

    Returns:
        Agent name (e.g., 'dqn', 'ppo', 'recurrent_ppo', 'qlearning')
    """
    path_lower = path.lower()

    if "recurrent_ppo" in path_lower or "rppo" in path_lower:
        return "recurrent_ppo"
    elif "ppo" in path_lower:
        return "ppo"
    elif "dqn" in path_lower:
        return "dqn"
    elif "qlearning" in path_lower or "q_learning" in path_lower:
        return "qlearning"
    else:
        return "unknown"


def load_agent(checkpoint_path: Path, agent_type: str, env: FlowEnvironment):
    """
    Load a trained agent from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        agent_type: Type of agent ('dqn', 'ppo', 'recurrent_ppo', 'qlearning')
        env: Environment instance for getting dimensions

    Returns:
        Loaded agent instance
    """
    obs_dim = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    # Initialize agent based on type
    if agent_type == "qlearning":
        agent = QLearningAgent(
            action_space_size=action_space_size,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.0,  # No exploration during evaluation
        )
    elif agent_type == "dqn":
        agent = DQNAgent(
            observation_dim=obs_dim,
            action_space_size=action_space_size,
            hidden_dims=[128, 128],
            learning_rate=0.001,
            epsilon=0.0,  # No exploration during evaluation
        )
    elif agent_type == "ppo":
        agent = PPOAgent(
            observation_dim=obs_dim,
            action_space_size=action_space_size,
            hidden_dims=[128, 128],
            learning_rate=0.0003,
        )
    elif agent_type == "recurrent_ppo":
        agent = RecurrentPPOAgent(
            observation_dim=obs_dim,
            action_space_size=action_space_size,
            hidden_dim=128,
            lstm_hidden_dim=128,
            learning_rate=0.0003,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Load checkpoint
    try:
        agent.load(str(checkpoint_path))
        print(f"  ✓ Loaded {agent_type} from {checkpoint_path}")
    except Exception as e:
        print(f"  ✗ Failed to load {agent_type} from {checkpoint_path}: {e}")
        return None

    return agent


def create_test_splits(
    base_config: GraphConfig,
    num_test_in_dist: int,
    num_test_per_shift: int,
    random_seed: int,
) -> Dict[str, List]:
    """
    Create test splits with distribution shifts.

    Args:
        base_config: Base graph configuration
        num_test_in_dist: Number of in-distribution test graphs
        num_test_per_shift: Number of graphs per distribution shift
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of test splits
    """
    generator = GraphGenerator(random_seed=random_seed)

    splits = generator.create_standard_splits(
        base_config=base_config,
        num_train=0,  # We don't need training graphs
        num_test_in_dist=num_test_in_dist,
        num_test_per_shift=num_test_per_shift,
    )

    # Remove train split
    if "train" in splits:
        del splits["train"]

    return splits


def evaluate_agent_generalization(
    agent,
    agent_name: str,
    test_splits: Dict[str, List],
    env_config: EnvConfig,
    num_episodes_per_env: int,
    verbose: bool = True,
) -> GeneralizationResults:
    """
    Evaluate a single agent's generalization performance.

    Args:
        agent: Trained agent to evaluate
        agent_name: Name of the agent
        test_splits: Dictionary of test graph splits
        env_config: Environment configuration
        num_episodes_per_env: Number of episodes per environment
        verbose: Whether to print progress

    Returns:
        GeneralizationResults
    """
    evaluator = GeneralizationEvaluator(verbose=verbose)

    results = evaluator.evaluate_zero_shot(
        agent=agent,
        graph_splits=test_splits,
        env_config=env_config,
        num_episodes_per_env=num_episodes_per_env,
        agent_name=agent_name,
    )

    return results


def save_results(
    results: Dict[str, GeneralizationResults],
    output_dir: str,
):
    """
    Save generalization results to JSON files.

    Args:
        results: Dictionary mapping agent names to GeneralizationResults
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)

    for agent_name, gen_results in results.items():
        # Prepare results dictionary
        results_dict = {
            "agent_name": agent_name,
            "in_distribution": {
                "success_rate": gen_results.in_dist_results.metrics.success_rate,
                "mean_steps": gen_results.in_dist_results.metrics.mean_steps_to_success,
                "mean_reward": gen_results.in_dist_results.metrics.mean_reward,
                "mean_coverage": gen_results.in_dist_results.metrics.mean_coverage,
                "failure_rate": gen_results.in_dist_results.metrics.failure_rate,
            },
            "out_of_distribution": {},
            "degradation_metrics": {},
        }

        # Add out-of-distribution results
        for shift_name, out_results in gen_results.out_dist_results.items():
            results_dict["out_of_distribution"][shift_name] = {
                "success_rate": out_results.metrics.success_rate,
                "mean_steps": out_results.metrics.mean_steps_to_success,
                "mean_reward": out_results.metrics.mean_reward,
                "mean_coverage": out_results.metrics.mean_coverage,
                "failure_rate": out_results.metrics.failure_rate,
            }

        # Add degradation metrics
        for shift_name, degradation in gen_results.degradation_metrics.items():
            results_dict["degradation_metrics"][shift_name] = {
                "success_rate_drop": degradation.success_rate_drop,
                "success_rate_drop_pct": degradation.success_rate_drop_pct,
                "steps_increase": degradation.steps_increase,
                "steps_increase_pct": degradation.steps_increase_pct,
                "coverage_drop": degradation.coverage_drop,
                "coverage_drop_pct": degradation.coverage_drop_pct,
                "reward_drop": degradation.reward_drop,
                "reward_drop_pct": degradation.reward_drop_pct,
            }

        # Save to file
        output_file = Path(output_dir) / f"{agent_name}_generalization.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"  ✓ Saved results to {output_file}")


def generate_visualizations(
    results: Dict[str, GeneralizationResults],
    output_dir: str,
):
    """
    Generate visualization plots for generalization results.

    Args:
        results: Dictionary mapping agent names to GeneralizationResults
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_gen = PlotGenerator(output_dir=output_dir)

    # Prepare data for plotting
    agent_names = list(results.keys())
    shift_names = []
    if results:
        first_result = next(iter(results.values()))
        shift_names = list(first_result.out_dist_results.keys())

    # 1. Plot generalization comparison across agents
    print("\n  Generating generalization comparison plot...")
    in_dist_success = []
    out_dist_success = {}

    for agent_name, gen_results in results.items():
        in_dist_success.append(gen_results.in_dist_results.metrics.success_rate)

        for shift_name in shift_names:
            if shift_name not in out_dist_success:
                out_dist_success[shift_name] = []
            out_dist_success[shift_name].append(
                gen_results.out_dist_results[shift_name].metrics.success_rate
            )

    # Create comparison plot
    plot_gen.plot_generalization_comparison(
        agent_names=agent_names,
        in_dist_success_rates=in_dist_success,
        out_dist_success_rates=out_dist_success,
        shift_names=shift_names,
    )

    # 2. Plot degradation heatmap
    print("  Generating degradation heatmap...")
    degradation_matrix = []
    for agent_name in agent_names:
        agent_degradations = []
        for shift_name in shift_names:
            degradation = results[agent_name].degradation_metrics[shift_name]
            agent_degradations.append(degradation.success_rate_drop_pct)
        degradation_matrix.append(agent_degradations)

    plot_gen.plot_degradation_heatmap(
        agent_names=agent_names,
        shift_names=shift_names,
        degradation_matrix=degradation_matrix,
    )

    print(f"\n  ✓ Saved plots to {output_dir}")


def print_summary(results: Dict[str, GeneralizationResults]):
    """
    Print summary of generalization results.

    Args:
        results: Dictionary mapping agent names to GeneralizationResults
    """
    print("\n" + "=" * 80)
    print("GENERALIZATION EVALUATION SUMMARY")
    print("=" * 80)

    for agent_name, gen_results in results.items():
        print(f"\n{agent_name.upper()}")
        print("-" * 80)

        # In-distribution performance
        in_dist = gen_results.in_dist_results.metrics
        print(f"\nIn-Distribution Performance:")
        print(f"  Success Rate: {in_dist.success_rate:.2%}")
        print(f"  Mean Steps:   {in_dist.mean_steps_to_success:.2f}")
        print(f"  Coverage:     {in_dist.mean_coverage:.2%}")
        print(f"  Mean Reward:  {in_dist.mean_reward:.3f}")

        # Out-of-distribution performance
        print(f"\nOut-of-Distribution Performance:")
        for shift_name, degradation in gen_results.degradation_metrics.items():
            print(f"\n  {shift_name}:")
            print(f"    Success Rate: {degradation.success_rate_out_dist:.2%}")
            print(
                f"    Degradation:  {degradation.success_rate_drop:+.2%} "
                f"({degradation.success_rate_drop_pct:+.1f}%)"
            )
            if degradation.steps_out_dist > 0:
                print(
                    f"    Steps:        {degradation.steps_out_dist:.2f} "
                    f"({degradation.steps_increase_pct:+.1f}%)"
                )

        # Average degradation
        avg_degradation = np.mean(
            [d.success_rate_drop_pct for d in gen_results.degradation_metrics.values()]
        )
        print(f"\n  Average Degradation: {avg_degradation:+.1f}%")

        # Generalization assessment
        if abs(avg_degradation) < 10:
            assessment = "✓ Excellent generalization"
        elif abs(avg_degradation) < 25:
            assessment = "○ Good generalization"
        else:
            assessment = "✗ Poor generalization"
        print(f"  Assessment: {assessment}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent generalization on unseen graphs"
    )

    # Model selection
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=["baseline_logs", "memory_logs", "test_logs"],
        help="Directories to search for trained models",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        help="Specific agents to evaluate (default: all found)",
    )

    # Test configuration
    parser.add_argument(
        "--num-test-in-dist",
        type=int,
        default=10,
        help="Number of in-distribution test graphs",
    )
    parser.add_argument(
        "--num-test-per-shift",
        type=int,
        default=5,
        help="Number of graphs per distribution shift",
    )
    parser.add_argument(
        "--episodes-per-env",
        type=int,
        default=10,
        help="Number of episodes per test environment",
    )

    # Graph configuration
    parser.add_argument(
        "--num-nodes", type=int, default=10, help="Number of nodes in test graphs"
    )
    parser.add_argument(
        "--branching-factor",
        type=float,
        default=2.0,
        help="Branching factor for test graphs",
    )
    parser.add_argument("--max_depth", type=int, default=5, help="Depth of test graphs")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generalization_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="generalization_plots",
        help="Output directory for plots",
    )

    # Execution options
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("GENERALIZATION EVALUATION")
    print("=" * 80)

    # Set random seed
    np.random.seed(args.random_seed)

    # 1. Find trained models
    print("\n1. Searching for trained models...")
    trained_models = find_trained_models(args.model_dirs)

    if not trained_models:
        print("\n✗ No trained models found!")
        print("\nPlease run baseline experiments first:")
        print("  python run_baseline_experiments.py")
        return

    # Filter by specified agents
    if args.agents:
        trained_models = {k: v for k, v in trained_models.items() if k in args.agents}

    print(f"\nFound trained models:")
    for agent_name, checkpoints in trained_models.items():
        print(f"  {agent_name}: {len(checkpoints)} checkpoint(s)")

    # 2. Create test splits
    print("\n2. Creating test splits with distribution shifts...")
    base_config = GraphConfig(
        num_nodes=args.num_nodes,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
    )

    test_splits = create_test_splits(
        base_config=base_config,
        num_test_in_dist=args.num_test_in_dist,
        num_test_per_shift=args.num_test_per_shift,
        random_seed=args.random_seed,
    )

    print(f"\nTest splits created:")
    for split_name, graphs in test_splits.items():
        print(f"  {split_name}: {len(graphs)} graphs")

    # 3. Create environment configuration
    env_config = EnvConfig(
        step_penalty=-0.01,
        success_reward=1.0,
        failure_penalty=-1.0,
        max_steps=100,
        fixed_max_actions=5,  # Fixed to handle graphs with varying out-degrees
    )

    # 4. Evaluate each agent
    print("\n3. Evaluating agents on test splits...")
    all_results = {}

    for agent_name, checkpoints in trained_models.items():
        print(f"\n--- Evaluating {agent_name.upper()} ---")

        # Use the first checkpoint (or could average over multiple)
        checkpoint_path = checkpoints[0]

        # Create dummy environment for loading agent
        dummy_graph = test_splits["test_in_dist"][0]
        dummy_env = FlowEnvironment(dummy_graph, env_config)

        # Load agent
        agent = load_agent(checkpoint_path, agent_name, dummy_env)
        if agent is None:
            print(f"  Skipping {agent_name} due to loading error")
            continue

        # Evaluate generalization
        try:
            results = evaluate_agent_generalization(
                agent=agent,
                agent_name=agent_name,
                test_splits=test_splits,
                env_config=env_config,
                num_episodes_per_env=args.episodes_per_env,
                verbose=args.verbose,
            )
            all_results[agent_name] = results
        except Exception as e:
            print(f"  ✗ Error evaluating {agent_name}: {e}")
            continue

    if not all_results:
        print("\n✗ No agents were successfully evaluated!")
        return

    # 5. Save results
    print("\n4. Saving results...")
    save_results(all_results, args.output_dir)

    # 6. Generate visualizations
    print("\n5. Generating visualizations...")
    generate_visualizations(all_results, args.plot_dir)

    # 7. Print summary
    print_summary(all_results)

    print("=" * 80)
    print("GENERALIZATION EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Plots saved to: {args.plot_dir}")
    print("\n")


if __name__ == "__main__":
    main()
