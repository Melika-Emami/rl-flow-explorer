"""
Run exploration strategy ablation experiments.

This script runs experiments comparing different exploration strategies
(epsilon-greedy, count-based, curiosity-based) with DQN and PPO agents.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np


def load_exploration_config(config_path: str) -> Dict[str, Any]:
    """
    Load exploration comparison configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_experiment_config(
    base_config: Dict[str, Any],
    experiment: Dict[str, Any],
    seed: int,
    output_dir: str,
) -> str:
    """
    Create a temporary configuration file for a single experiment.

    Args:
        base_config: Base configuration with environment and training settings
        experiment: Experiment-specific configuration
        seed: Random seed
        output_dir: Output directory for config file

    Returns:
        Path to created config file
    """
    # Merge configurations
    config = {
        "experiment_name": f"{experiment['name']}_seed{seed}",
        "agent": experiment["agent"],
        "seed": seed,
        "environment": base_config["base_environment"].copy(),
        "training": base_config["base_training"].copy(),
        "agent_config": experiment["agent_config"].copy(),
    }

    # Update training config with experiment-specific settings
    if "training" in experiment:
        config["training"].update(experiment["training"])

    # Add exploration configuration
    if "exploration" in experiment:
        config["exploration"] = experiment["exploration"]

    if "exploration_config" in experiment:
        config["exploration_config"] = experiment["exploration_config"]

    # Save config file
    output_path = Path(output_dir) / f"{experiment['name']}_seed{seed}_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return str(output_path)


def run_experiment(
    config_path: str,
    log_dir: str,
) -> Dict[str, Any]:
    """
    Run a single exploration ablation experiment.

    Args:
        config_path: Path to experiment configuration file
        log_dir: Base directory for logs

    Returns:
        Dictionary with experiment results
    """
    # Load config to get experiment name
    with open(config_path, "r") as f:
        config = json.load(f)

    experiment_name = config["experiment_name"]
    agent = config["agent"]
    seed = config["seed"]

    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--config",
        config_path,
        "--log-dir",
        log_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Agent: {agent}, Seed: {seed}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"\n{'='*60}")
        print(f"Completed experiment: {experiment_name}")
        print(f"{'='*60}\n")

        # Load results
        log_path = Path(log_dir) / experiment_name
        metrics_path = log_path / "training_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            return {
                "experiment_name": experiment_name,
                "agent": agent,
                "seed": seed,
                "success": True,
                "metrics": metrics,
                "log_dir": str(log_path),
            }
        else:
            return {
                "experiment_name": experiment_name,
                "agent": agent,
                "seed": seed,
                "success": False,
                "error": "Metrics file not found",
            }

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"FAILED experiment: {experiment_name}")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        print(f"{'='*60}\n")

        return {
            "experiment_name": experiment_name,
            "agent": agent,
            "seed": seed,
            "success": False,
            "error": str(e),
            "stderr": e.stderr if e.stderr else None,
        }


def aggregate_exploration_results(
    results: List[Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate results across seeds for each exploration strategy.

    Args:
        results: List of result dictionaries from individual runs
        experiments: List of experiment configurations

    Returns:
        Dictionary with aggregated statistics per exploration strategy
    """
    aggregated = {}

    for experiment in experiments:
        exp_name = experiment["name"]
        agent = experiment["agent"]
        exploration = experiment.get("exploration", "epsilon_greedy")

        # Filter results for this experiment
        exp_results = [
            r for r in results if r["success"] and exp_name in r["experiment_name"]
        ]

        if not exp_results:
            aggregated[exp_name] = {
                "agent": agent,
                "exploration": exploration,
                "num_seeds": 0,
                "num_successful": 0,
                "error": "No successful runs",
            }
            continue

        # Extract final metrics from each seed
        final_rewards = []
        final_success_rates = []
        final_steps = []
        final_coverage = []
        overall_success_rates = []

        for result in exp_results:
            metrics = result["metrics"]["metrics"]
            if metrics:
                # Get last episode metrics
                last_episode = metrics[-1]
                final_rewards.append(last_episode.get("reward", 0))
                final_success_rates.append(
                    1.0 if last_episode.get("success", False) else 0.0
                )
                final_steps.append(last_episode.get("steps", 0))

                # Get coverage if available
                if "coverage" in last_episode:
                    final_coverage.append(last_episode["coverage"])

                # Compute overall success rate across all episodes
                total_successes = sum(1 for ep in metrics if ep.get("success", False))
                overall_success_rate = (
                    total_successes / len(metrics) if metrics else 0.0
                )
                overall_success_rates.append(overall_success_rate)

        # Compute statistics
        aggregated[exp_name] = {
            "agent": agent,
            "exploration": exploration,
            "num_seeds": len(exp_results),
            "num_successful": len(exp_results),
            "seeds": [r["seed"] for r in exp_results],
        }

        if final_rewards:
            aggregated[exp_name]["final_reward"] = {
                "mean": float(np.mean(final_rewards)),
                "std": float(np.std(final_rewards)),
                "min": float(np.min(final_rewards)),
                "max": float(np.max(final_rewards)),
                "values": final_rewards,
            }

        if final_success_rates:
            aggregated[exp_name]["final_episode_success_rate"] = {
                "mean": float(np.mean(final_success_rates)),
                "std": float(np.std(final_success_rates)),
                "values": final_success_rates,
            }

        if overall_success_rates:
            aggregated[exp_name]["overall_success_rate"] = {
                "mean": float(np.mean(overall_success_rates)),
                "std": float(np.std(overall_success_rates)),
                "values": overall_success_rates,
            }

        if final_steps:
            aggregated[exp_name]["final_steps"] = {
                "mean": float(np.mean(final_steps)),
                "std": float(np.std(final_steps)),
                "min": float(np.min(final_steps)),
                "max": float(np.max(final_steps)),
                "values": final_steps,
            }

        if final_coverage:
            aggregated[exp_name]["final_coverage"] = {
                "mean": float(np.mean(final_coverage)),
                "std": float(np.std(final_coverage)),
                "values": final_coverage,
            }

    return aggregated


def print_comparison_table(aggregated: Dict[str, Any]):
    """
    Print comparison table of exploration strategies.

    Args:
        aggregated: Aggregated results dictionary
    """
    print("\n" + "=" * 100)
    print("EXPLORATION STRATEGY COMPARISON")
    print("=" * 100)

    # Print header
    print(
        f"\n{'Experiment':<30} {'Agent':<10} {'Exploration':<15} {'Overall Success':<20} {'Final Success':<20} {'Reward':<20}"
    )
    print("-" * 115)

    # Print results for each experiment
    for exp_name, results in aggregated.items():
        if results["num_successful"] == 0:
            print(f"{exp_name:<30} {'FAILED':<10}")
            continue

        agent = results["agent"]
        exploration = results["exploration"]

        overall_success = results.get("overall_success_rate", {})
        final_success = results.get("final_episode_success_rate", {})
        reward = results.get("final_reward", {})

        overall_str = (
            f"{overall_success['mean']:.2%} ± {overall_success['std']:.2%}"
            if overall_success
            else "N/A"
        )
        final_str = (
            f"{final_success['mean']:.2%} ± {final_success['std']:.2%}"
            if final_success
            else "N/A"
        )
        reward_str = f"{reward['mean']:.2f} ± {reward['std']:.2f}" if reward else "N/A"

        print(
            f"{exp_name:<30} {agent:<10} {exploration:<15} {overall_str:<20} {final_str:<20} {reward_str:<20}"
        )

    print("=" * 100)

    # Print comparison by agent
    print("\n" + "=" * 100)
    print("COMPARISON BY AGENT")
    print("=" * 100)

    agents = set(r["agent"] for r in aggregated.values() if r["num_successful"] > 0)

    for agent in sorted(agents):
        print(f"\n{agent.upper()}:")
        print("-" * 100)

        agent_results = {
            name: results
            for name, results in aggregated.items()
            if results.get("agent") == agent and results["num_successful"] > 0
        }

        for exp_name, results in agent_results.items():
            exploration = results["exploration"]
            overall_success = results.get("overall_success_rate", {})
            final_success = results.get("final_episode_success_rate", {})
            reward = results.get("final_reward", {})

            overall_str = (
                f"{overall_success['mean']:.2%} ± {overall_success['std']:.2%}"
                if overall_success
                else "N/A"
            )
            final_str = (
                f"{final_success['mean']:.2%} ± {final_success['std']:.2%}"
                if final_success
                else "N/A"
            )
            reward_str = (
                f"{reward['mean']:.2f} ± {reward['std']:.2f}" if reward else "N/A"
            )

            print(
                f"  {exploration:<20} Overall: {overall_str:<20} Final: {final_str:<20} Reward: {reward_str}"
            )

    print("=" * 100 + "\n")


def save_results(
    aggregated: Dict[str, Any],
    output_dir: str,
    config_name: str,
):
    """
    Save aggregated results to JSON file.

    Args:
        aggregated: Aggregated results dictionary
        output_dir: Output directory
        config_name: Name of the configuration
    """
    output_path = Path(output_dir) / f"{config_name}_results.json"

    # Add metadata
    full_results = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "results": aggregated,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run exploration strategy ablation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/exploration_comparison.json",
        help="Path to exploration comparison configuration file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="exploration_logs",
        help="Base directory for experiment logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exploration_results",
        help="Directory for aggregated results",
    )
    parser.add_argument(
        "--temp-config-dir",
        type=str,
        default="temp_configs",
        help="Directory for temporary config files",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (by name). If not provided, runs all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_exploration_config(args.config)

    # Create output directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.temp_config_dir).mkdir(parents=True, exist_ok=True)

    # Filter experiments if specified
    experiments = config["experiments"]
    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]
        if not experiments:
            print(f"Error: No experiments found matching {args.experiments}")
            return 1

    seeds = config.get("seeds", [42, 142, 242, 342, 442])

    print(f"\n{'='*60}")
    print(f"EXPLORATION STRATEGY ABLATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Configuration: {config.get('experiment_name', 'exploration_comparison')}")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(experiments) * len(seeds)}")
    print(f"{'='*60}\n")

    # Run experiments
    all_results = []

    for experiment in experiments:
        exp_name = experiment["name"]
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"Agent: {experiment['agent']}")
        print(f"Exploration: {experiment.get('exploration', 'epsilon_greedy')}")
        print(f"{'='*60}\n")

        for seed in seeds:
            # Check if results already exist
            if args.skip_existing:
                log_path = Path(args.log_dir) / f"{exp_name}_seed{seed}"
                metrics_path = log_path / "training_metrics.json"
                if metrics_path.exists():
                    print(f"Skipping {exp_name} (seed={seed}) - results already exist")

                    # Load existing results
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)

                    all_results.append(
                        {
                            "experiment_name": f"{exp_name}_seed{seed}",
                            "agent": experiment["agent"],
                            "seed": seed,
                            "success": True,
                            "metrics": metrics,
                            "log_dir": str(log_path),
                        }
                    )
                    continue

            # Create experiment config
            config_path = create_experiment_config(
                config,
                experiment,
                seed,
                args.temp_config_dir,
            )

            # Run experiment
            result = run_experiment(config_path, args.log_dir)
            all_results.append(result)

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60 + "\n")

    aggregated = aggregate_exploration_results(all_results, experiments)

    # Print comparison table
    print_comparison_table(aggregated)

    # Save results
    config_name = config.get("experiment_name", "exploration_comparison")
    save_results(aggregated, args.output_dir, config_name)

    # Check if all experiments succeeded
    num_successful = sum(1 for r in aggregated.values() if r["num_successful"] > 0)
    num_total = len(experiments)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {num_total}")
    print(f"Successful: {num_successful}")
    print(f"Failed: {num_total - num_successful}")
    print(f"{'='*60}\n")

    return 0 if num_successful == num_total else 1


if __name__ == "__main__":
    sys.exit(main())
