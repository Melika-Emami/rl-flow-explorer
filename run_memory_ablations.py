"""
Run memory ablation experiments.

This script runs experiments comparing PPO vs Recurrent PPO agents
on environments with varying levels of partial observability to isolate
the contribution of memory (LSTM) to performance.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np


def load_memory_config(config_path: str) -> Dict[str, Any]:
    """
    Load memory ablation configuration.

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
        base_config: Base configuration with training settings
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
        "environment": experiment["environment"].copy(),
        "training": base_config["base_training"].copy(),
        "agent_config": experiment["agent_config"].copy(),
    }

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
    Run a single memory ablation experiment.

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


def aggregate_memory_results(
    results: List[Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate results across seeds for each memory configuration.

    Args:
        results: List of result dictionaries from individual runs
        experiments: List of experiment configurations

    Returns:
        Dictionary with aggregated statistics per configuration
    """
    aggregated = {}

    for experiment in experiments:
        exp_name = experiment["name"]
        agent = experiment["agent"]
        popup_prob = experiment["environment"]["popup_probability"]
        num_deps = experiment["environment"]["num_dependencies"]

        # Filter results for this experiment
        exp_results = [
            r for r in results if r["success"] and exp_name in r["experiment_name"]
        ]

        if not exp_results:
            aggregated[exp_name] = {
                "agent": agent,
                "popup_probability": popup_prob,
                "num_dependencies": num_deps,
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

        # Compute statistics
        aggregated[exp_name] = {
            "agent": agent,
            "popup_probability": popup_prob,
            "num_dependencies": num_deps,
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
            aggregated[exp_name]["final_success_rate"] = {
                "mean": float(np.mean(final_success_rates)),
                "std": float(np.std(final_success_rates)),
                "values": final_success_rates,
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


def compute_memory_contribution(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the contribution of memory (LSTM) by comparing PPO vs Recurrent PPO.

    Args:
        aggregated: Aggregated results dictionary

    Returns:
        Dictionary with memory contribution analysis
    """
    memory_analysis = {}

    # Group by observability level
    observability_levels = {
        "low": 0.1,
        "medium": 0.3,
        "high": 0.5,
    }

    for level_name, popup_prob in observability_levels.items():
        # Find PPO and Recurrent PPO results for this level
        ppo_key = f"ppo_{level_name}_observability"
        recurrent_ppo_key = f"recurrent_ppo_{level_name}_observability"

        if ppo_key not in aggregated or recurrent_ppo_key not in aggregated:
            continue

        ppo_results = aggregated[ppo_key]
        recurrent_results = aggregated[recurrent_ppo_key]

        if (
            ppo_results["num_successful"] == 0
            or recurrent_results["num_successful"] == 0
        ):
            continue

        # Compute improvement from adding memory
        ppo_success = ppo_results.get("final_success_rate", {}).get("mean", 0)
        recurrent_success = recurrent_results.get("final_success_rate", {}).get(
            "mean", 0
        )
        success_improvement = recurrent_success - ppo_success
        success_improvement_pct = (
            (success_improvement / ppo_success * 100) if ppo_success > 0 else 0
        )

        ppo_reward = ppo_results.get("final_reward", {}).get("mean", 0)
        recurrent_reward = recurrent_results.get("final_reward", {}).get("mean", 0)
        reward_improvement = recurrent_reward - ppo_reward
        reward_improvement_pct = (
            (reward_improvement / abs(ppo_reward) * 100) if ppo_reward != 0 else 0
        )

        ppo_steps = ppo_results.get("final_steps", {}).get("mean", 0)
        recurrent_steps = recurrent_results.get("final_steps", {}).get("mean", 0)
        steps_reduction = ppo_steps - recurrent_steps
        steps_reduction_pct = (
            (steps_reduction / ppo_steps * 100) if ppo_steps > 0 else 0
        )

        memory_analysis[level_name] = {
            "popup_probability": popup_prob,
            "ppo": {
                "success_rate": ppo_success,
                "reward": ppo_reward,
                "steps": ppo_steps,
            },
            "recurrent_ppo": {
                "success_rate": recurrent_success,
                "reward": recurrent_reward,
                "steps": recurrent_steps,
            },
            "improvement": {
                "success_rate_absolute": success_improvement,
                "success_rate_percent": success_improvement_pct,
                "reward_absolute": reward_improvement,
                "reward_percent": reward_improvement_pct,
                "steps_reduction_absolute": steps_reduction,
                "steps_reduction_percent": steps_reduction_pct,
            },
        }

    return memory_analysis


def print_comparison_table(aggregated: Dict[str, Any]):
    """
    Print comparison table of PPO vs Recurrent PPO.

    Args:
        aggregated: Aggregated results dictionary
    """
    print("\n" + "=" * 120)
    print("MEMORY ABLATION: PPO vs RECURRENT PPO")
    print("=" * 120)

    # Print header
    print(
        f"\n{'Experiment':<35} {'Agent':<15} {'Popup%':<10} {'Success Rate':<20} {'Reward':<20} {'Steps':<15}"
    )
    print("-" * 120)

    # Print results for each experiment
    for exp_name, results in sorted(aggregated.items()):
        if results["num_successful"] == 0:
            print(f"{exp_name:<35} {'FAILED':<15}")
            continue

        agent = results["agent"]
        popup_prob = results["popup_probability"]

        success_rate = results.get("final_success_rate", {})
        reward = results.get("final_reward", {})
        steps = results.get("final_steps", {})

        success_str = (
            f"{success_rate['mean']:.2%} ± {success_rate['std']:.2%}"
            if success_rate
            else "N/A"
        )
        reward_str = f"{reward['mean']:.2f} ± {reward['std']:.2f}" if reward else "N/A"
        steps_str = f"{steps['mean']:.1f} ± {steps['std']:.1f}" if steps else "N/A"

        print(
            f"{exp_name:<35} {agent:<15} {popup_prob*100:>5.0f}%     {success_str:<20} {reward_str:<20} {steps_str:<15}"
        )

    print("=" * 120)


def print_memory_contribution(memory_analysis: Dict[str, Any]):
    """
    Print analysis of memory contribution.

    Args:
        memory_analysis: Memory contribution analysis dictionary
    """
    print("\n" + "=" * 120)
    print("MEMORY CONTRIBUTION ANALYSIS")
    print("=" * 120)
    print("\nImprovement from adding LSTM memory (Recurrent PPO vs PPO):\n")

    print(
        f"{'Observability':<20} {'Popup%':<10} {'Success Δ':<20} {'Reward Δ':<20} {'Steps Δ':<20}"
    )
    print("-" * 120)

    for level_name, analysis in sorted(memory_analysis.items()):
        popup_prob = analysis["popup_probability"]
        improvement = analysis["improvement"]

        success_delta = f"+{improvement['success_rate_absolute']:.2%} ({improvement['success_rate_percent']:+.1f}%)"
        reward_delta = f"{improvement['reward_absolute']:+.2f} ({improvement['reward_percent']:+.1f}%)"
        steps_delta = f"{improvement['steps_reduction_absolute']:+.1f} ({improvement['steps_reduction_percent']:+.1f}%)"

        print(
            f"{level_name.capitalize():<20} {popup_prob*100:>5.0f}%     {success_delta:<20} {reward_delta:<20} {steps_delta:<20}"
        )

    print("=" * 120)

    # Print interpretation
    print("\nINTERPRETATION:")
    print("-" * 120)

    for level_name, analysis in sorted(memory_analysis.items()):
        improvement = analysis["improvement"]
        popup_prob = analysis["popup_probability"]

        print(
            f"\n{level_name.capitalize()} Partial Observability ({popup_prob*100:.0f}% popup rate):"
        )

        if improvement["success_rate_percent"] > 5:
            print(
                f"  ✓ Memory provides SIGNIFICANT benefit (+{improvement['success_rate_percent']:.1f}% success rate)"
            )
        elif improvement["success_rate_percent"] > 0:
            print(
                f"  ~ Memory provides MODEST benefit (+{improvement['success_rate_percent']:.1f}% success rate)"
            )
        else:
            print(
                f"  ✗ Memory provides NO benefit ({improvement['success_rate_percent']:+.1f}% success rate)"
            )

        if improvement["reward_percent"] > 5:
            print(f"  ✓ Reward improvement: {improvement['reward_percent']:+.1f}%")
        elif improvement["reward_percent"] > 0:
            print(
                f"  ~ Modest reward improvement: {improvement['reward_percent']:+.1f}%"
            )

        if improvement["steps_reduction_percent"] > 5:
            print(
                f"  ✓ Efficiency gain: {improvement['steps_reduction_percent']:+.1f}% fewer steps"
            )

    print("\n" + "=" * 120)


def save_results(
    aggregated: Dict[str, Any],
    memory_analysis: Dict[str, Any],
    output_dir: str,
    config_name: str,
):
    """
    Save aggregated results to JSON file.

    Args:
        aggregated: Aggregated results dictionary
        memory_analysis: Memory contribution analysis
        output_dir: Output directory
        config_name: Name of the configuration
    """
    output_path = Path(output_dir) / f"{config_name}_results.json"

    # Add metadata
    full_results = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "results": aggregated,
        "memory_contribution": memory_analysis,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run memory ablation experiments (PPO vs Recurrent PPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/memory_ablation.json",
        help="Path to memory ablation configuration file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="memory_logs",
        help="Base directory for experiment logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="memory_results",
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
    config = load_memory_config(args.config)

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
    print(f"MEMORY ABLATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Configuration: {config.get('experiment_name', 'memory_ablation')}")
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
        print(f"Popup probability: {experiment['environment']['popup_probability']}")
        print(f"Dependencies: {experiment['environment']['num_dependencies']}")
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

    aggregated = aggregate_memory_results(all_results, experiments)

    # Compute memory contribution
    memory_analysis = compute_memory_contribution(aggregated)

    # Print comparison table
    print_comparison_table(aggregated)

    # Print memory contribution analysis
    print_memory_contribution(memory_analysis)

    # Save results
    config_name = config.get("experiment_name", "memory_ablation")
    save_results(aggregated, memory_analysis, args.output_dir, config_name)

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
