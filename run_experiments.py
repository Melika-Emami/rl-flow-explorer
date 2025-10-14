"""
Experiment runner for executing training across multiple seeds.

This script runs experiments with multiple random seeds, optionally in parallel,
and aggregates results with summary statistics.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np


def run_single_experiment(
    agent: str,
    config_path: Optional[str],
    seed: int,
    base_log_dir: str,
    experiment_name: str,
    additional_args: List[str] = None,
) -> Dict[str, Any]:
    """
    Run a single training experiment with a specific seed.

    Args:
        agent: Type of agent to train
        config_path: Path to configuration file (optional)
        seed: Random seed
        base_log_dir: Base directory for logs
        experiment_name: Name of the experiment
        additional_args: Additional command-line arguments

    Returns:
        Dictionary with experiment results
    """
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--agent",
        agent,
        "--seed",
        str(seed),
        "--log-dir",
        base_log_dir,
        "--experiment-name",
        f"{experiment_name}_seed{seed}",
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    if additional_args:
        cmd.extend(additional_args)

    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name} (seed={seed})")
    print(f"Command: {' '.join(cmd)}")
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
        print(f"Completed experiment: {experiment_name} (seed={seed})")
        print(f"{'='*60}\n")

        # Load results
        log_dir = Path(base_log_dir) / f"{experiment_name}_seed{seed}"
        metrics_path = log_dir / "training_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            return {
                "seed": seed,
                "success": True,
                "metrics": metrics,
                "log_dir": str(log_dir),
            }
        else:
            return {
                "seed": seed,
                "success": False,
                "error": "Metrics file not found",
            }

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"FAILED experiment: {experiment_name} (seed={seed})")
        print(f"Error: {e}")
        print(f"{'='*60}\n")

        return {
            "seed": seed,
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple seeds.

    Args:
        results: List of result dictionaries from individual runs

    Returns:
        Dictionary with aggregated statistics
    """
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {
            "num_seeds": len(results),
            "num_successful": 0,
            "num_failed": len(results),
            "seeds": [],
            "failed_seeds": [r["seed"] for r in results],
            "error": "All experiments failed",
        }

    # Extract final metrics from each seed
    final_rewards = []
    final_success_rates = []
    final_steps = []

    for result in successful_results:
        metrics = result["metrics"]["metrics"]
        if metrics:
            # Get last episode metrics
            last_episode = metrics[-1]
            final_rewards.append(last_episode.get("reward", 0))
            final_success_rates.append(
                1.0 if last_episode.get("success", False) else 0.0
            )
            final_steps.append(last_episode.get("steps", 0))

    # Compute statistics
    aggregated = {
        "num_seeds": len(results),
        "num_successful": len(successful_results),
        "num_failed": len(results) - len(successful_results),
        "seeds": [r["seed"] for r in successful_results],
        "failed_seeds": [r["seed"] for r in results if not r["success"]],
    }

    if final_rewards:
        aggregated["final_reward"] = {
            "mean": float(np.mean(final_rewards)),
            "std": float(np.std(final_rewards)),
            "min": float(np.min(final_rewards)),
            "max": float(np.max(final_rewards)),
            "values": final_rewards,
        }

    if final_success_rates:
        aggregated["final_success_rate"] = {
            "mean": float(np.mean(final_success_rates)),
            "std": float(np.std(final_success_rates)),
            "values": final_success_rates,
        }

    if final_steps:
        aggregated["final_steps"] = {
            "mean": float(np.mean(final_steps)),
            "std": float(np.std(final_steps)),
            "min": float(np.min(final_steps)),
            "max": float(np.max(final_steps)),
            "values": final_steps,
        }

    return aggregated


def save_aggregated_results(
    aggregated: Dict[str, Any],
    output_dir: str,
    experiment_name: str,
):
    """
    Save aggregated results to JSON file.

    Args:
        aggregated: Aggregated results dictionary
        output_dir: Output directory
        experiment_name: Name of the experiment
    """
    output_path = Path(output_dir) / f"{experiment_name}_aggregated.json"

    # Add metadata
    full_results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "results": aggregated,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nAggregated results saved to {output_path}")


def print_summary(aggregated: Dict[str, Any], experiment_name: str):
    """
    Print summary of aggregated results.

    Args:
        aggregated: Aggregated results dictionary
        experiment_name: Name of the experiment
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {experiment_name}")
    print("=" * 60)

    print(f"\nTotal seeds: {aggregated['num_seeds']}")
    print(f"Successful: {aggregated['num_successful']}")
    print(f"Failed: {aggregated['num_failed']}")

    if aggregated["num_successful"] > 0:
        if "final_reward" in aggregated:
            print(f"\nFinal Reward:")
            print(
                f"  Mean: {aggregated['final_reward']['mean']:.3f} ± {aggregated['final_reward']['std']:.3f}"
            )
            print(
                f"  Range: [{aggregated['final_reward']['min']:.3f}, {aggregated['final_reward']['max']:.3f}]"
            )

        if "final_success_rate" in aggregated:
            print(f"\nFinal Success Rate:")
            print(
                f"  Mean: {aggregated['final_success_rate']['mean']:.2%} ± {aggregated['final_success_rate']['std']:.2%}"
            )

        if "final_steps" in aggregated:
            print(f"\nFinal Steps:")
            print(
                f"  Mean: {aggregated['final_steps']['mean']:.1f} ± {aggregated['final_steps']['std']:.1f}"
            )
            print(
                f"  Range: [{aggregated['final_steps']['min']:.0f}, {aggregated['final_steps']['max']:.0f}]"
            )

    if aggregated["failed_seeds"]:
        print(f"\nFailed seeds: {aggregated['failed_seeds']}")

    print("=" * 60 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments across multiple seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment configuration
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        help="Type of agent to train",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name for this experiment"
    )

    # Seed configuration
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of seeds to use (e.g., --seeds 42 43 44)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds to generate (if --seeds not provided)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for generating random seeds",
    )

    # Execution configuration
    parser.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of parallel workers"
    )

    # Output configuration
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Base directory for logs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory for aggregated results",
    )

    # Additional training arguments
    parser.add_argument(
        "--training-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional arguments to pass to train.py",
    )

    return parser.parse_args()


def main():
    """Main experiment runner function."""
    args = parse_args()

    # Generate seeds if not provided
    if args.seeds:
        seeds = args.seeds
    else:
        np.random.seed(args.base_seed)
        seeds = [args.base_seed + i * 100 for i in range(args.num_seeds)]

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENTS: {args.experiment_name}")
    print(f"{'='*60}")
    print(f"Agent: {args.agent}")
    print(f"Seeds: {seeds}")
    print(f"Parallel: {args.parallel}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = []

    if args.parallel:
        # Run in parallel
        print(f"Running experiments in parallel with {args.max_workers} workers...\n")

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []

            for seed in seeds:
                future = executor.submit(
                    run_single_experiment,
                    args.agent,
                    args.config,
                    seed,
                    args.log_dir,
                    args.experiment_name,
                    args.training_args,
                )
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    results.append(
                        {
                            "seed": None,
                            "success": False,
                            "error": str(e),
                        }
                    )
    else:
        # Run sequentially
        print("Running experiments sequentially...\n")

        for seed in seeds:
            result = run_single_experiment(
                args.agent,
                args.config,
                seed,
                args.log_dir,
                args.experiment_name,
                args.training_args,
            )
            results.append(result)

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(results)

    # Save aggregated results
    save_aggregated_results(aggregated, str(output_dir), args.experiment_name)

    # Print summary
    print_summary(aggregated, args.experiment_name)

    # Return success if at least one experiment succeeded
    if aggregated["num_successful"] > 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
