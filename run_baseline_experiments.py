"""
Script to run all baseline experiments.

This script trains:
- Q-learning agent on small graphs
- DQN agent on medium graphs
- PPO agent on medium graphs
- Recurrent PPO agent on medium graphs

Each agent is trained with 5+ random seeds for statistical significance.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_experiment_batch(
    agent: str,
    config_path: str,
    experiment_name: str,
    num_seeds: int = 5,
    base_seed: int = 42,
    parallel: bool = False,
    max_workers: int = 4,
):
    """
    Run a batch of experiments for a specific agent.

    Args:
        agent: Agent type
        config_path: Path to configuration file
        experiment_name: Name for the experiment
        num_seeds: Number of seeds to run
        base_seed: Base seed for generating seeds
        parallel: Whether to run in parallel
        max_workers: Maximum number of parallel workers
    """
    print(f"\n{'='*70}")
    print(f"RUNNING BASELINE EXPERIMENTS: {experiment_name}")
    print(f"{'='*70}")
    print(f"Agent: {agent}")
    print(f"Config: {config_path}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Base seed: {base_seed}")
    print(f"Parallel: {parallel}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        sys.executable,
        "run_experiments.py",
        "--agent",
        agent,
        "--config",
        config_path,
        "--experiment-name",
        experiment_name,
        "--num-seeds",
        str(num_seeds),
        "--base-seed",
        str(base_seed),
        "--log-dir",
        "baseline_logs",
        "--output-dir",
        "baseline_results",
    ]

    if parallel:
        cmd.extend(["--parallel", "--max-workers", str(max_workers)])

    # Run experiment
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Experiment {experiment_name} failed with error: {e}")
        return False


def main():
    """Main function to run all baseline experiments."""
    parser = argparse.ArgumentParser(
        description="Run all baseline experiments for",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds to run for each agent",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for generating random seeds",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        choices=["qlearning", "dqn", "ppo", "recurrent_ppo"],
        help="Which agents to train (default: all)",
    )

    args = parser.parse_args()

    # Create output directories
    Path("baseline_logs").mkdir(exist_ok=True)
    Path("baseline_results").mkdir(exist_ok=True)

    # Define experiments
    experiments = []

    if "qlearning" in args.agents:
        experiments.append(
            {
                "agent": "qlearning",
                "config": "configs/qlearning_small.json",
                "name": "baseline_qlearning_small",
            }
        )

    if "dqn" in args.agents:
        experiments.append(
            {
                "agent": "dqn",
                "config": "configs/dqn_baseline.json",
                "name": "baseline_dqn_medium",
            }
        )

    if "ppo" in args.agents:
        experiments.append(
            {
                "agent": "ppo",
                "config": "configs/ppo_baseline.json",
                "name": "baseline_ppo_medium",
            }
        )

    if "recurrent_ppo" in args.agents:
        experiments.append(
            {
                "agent": "recurrent_ppo",
                "config": "configs/recurrent_ppo_baseline.json",
                "name": "baseline_recurrent_ppo_medium",
            }
        )

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nExperiments to run: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['name']} ({exp['agent']})")
    print(f"\nSeeds per experiment: {args.num_seeds}")
    print(f"Total training runs: {len(experiments) * args.num_seeds}")
    print(f"Parallel execution: {args.parallel}")
    if args.parallel:
        print(f"Max workers: {args.max_workers}")
    print("=" * 70 + "\n")

    # Run experiments
    results = {}
    failed_experiments = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'#'*70}\n")

        success = run_experiment_batch(
            agent=exp["agent"],
            config_path=exp["config"],
            experiment_name=exp["name"],
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            parallel=args.parallel,
            max_workers=args.max_workers,
        )

        results[exp["name"]] = success

        if not success:
            failed_experiments.append(exp["name"])
            print(f"\nWARNING: Experiment {exp['name']} failed!")
        else:
            print(f"\nSUCCESS: Experiment {exp['name']} completed successfully!")

    # Print final summary
    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS SUMMARY")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Successful: {sum(results.values())}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp_name in failed_experiments:
            print(f"  - {exp_name}")

    print(f"\nResults saved to: baseline_results/")
    print(f"Logs saved to: baseline_logs/")
    print("=" * 70 + "\n")

    # Return success if all experiments succeeded
    if failed_experiments:
        print("WARNING: Some experiments failed. Please check the logs.")
        return 1
    else:
        print("SUCCESS: All baseline experiments completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
