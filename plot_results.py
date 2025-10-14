"""
Plotting script for generating visualizations from experiment results.

This script provides a command-line interface for generating all plots
from training and evaluation results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.visualization.plot_generator import PlotGenerator
from src.evaluation.metrics import AggregateMetrics


def set_random_seed(seed: int):
    """Set random seed for reproducible plotting."""
    np.random.seed(seed)


def load_training_results(results_dir: str) -> Dict[str, Any]:
    """
    Load training results from directory.

    Args:
        results_dir: Directory containing training_metrics.json

    Returns:
        Dictionary with training metrics
    """
    metrics_path = Path(results_dir) / "training_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Training metrics not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data


def load_evaluation_results(results_path: str) -> AggregateMetrics:
    """
    Load evaluation results and convert to AggregateMetrics.

    Args:
        results_path: Path to evaluation results JSON file

    Returns:
        AggregateMetrics object
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    metrics_dict = data["metrics"]

    # Create AggregateMetrics object
    return AggregateMetrics(
        success_rate=metrics_dict["success_rate"],
        success_rate_std=metrics_dict["success_rate_std"],
        success_rate_ci=tuple(metrics_dict["success_rate_ci"]),
        mean_steps_to_success=metrics_dict["mean_steps_to_success"],
        std_steps_to_success=metrics_dict["std_steps_to_success"],
        steps_ci=(0, 0),  # Not stored in JSON
        failure_rate=metrics_dict["failure_rate"],
        failure_types={},  # Not stored in JSON
        mean_coverage=metrics_dict["mean_coverage"],
        std_coverage=metrics_dict["std_coverage"],
        coverage_ci=(0, 0),  # Not stored in JSON
        mean_reward=metrics_dict["mean_reward"],
        std_reward=metrics_dict["std_reward"],
        reward_ci=(0, 0),  # Not stored in JSON
        mean_episode_length=metrics_dict["mean_episode_length"],
        std_episode_length=0.0,  # Not stored in JSON
        num_episodes=metrics_dict["num_episodes"],
        num_successes=metrics_dict["num_successes"],
        num_failures=metrics_dict["num_failures"],
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Plot type selection
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=[
            "learning_curves",
            "multiple_metrics",
            "ablation",
            "algorithm_comparison",
            "exploration_comparison",
            "memory_comparison",
            "generalization",
            "degradation",
            "all",
        ],
        help="Type of plot to generate",
    )

    # Input paths
    parser.add_argument(
        "--results-dirs",
        type=str,
        nargs="+",
        help="Directories containing training results (for learning curves)",
    )
    parser.add_argument(
        "--agent-names",
        type=str,
        nargs="+",
        help="Names for agents (must match number of results-dirs)",
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        nargs="+",
        help="Paths to evaluation result JSON files",
    )
    parser.add_argument(
        "--eval-names", type=str, nargs="+", help="Names for evaluation results"
    )

    # Generalization-specific inputs
    parser.add_argument(
        "--in-dist-results",
        type=str,
        nargs="+",
        help="Paths to in-distribution evaluation results",
    )
    parser.add_argument(
        "--ood-results",
        type=str,
        nargs="+",
        help="Paths to out-of-distribution evaluation results",
    )

    # Memory comparison inputs
    parser.add_argument(
        "--recurrent-result", type=str, help="Path to recurrent agent evaluation result"
    )
    parser.add_argument(
        "--non-recurrent-result",
        type=str,
        help="Path to non-recurrent agent evaluation result",
    )

    # Plot configuration
    parser.add_argument(
        "--metric",
        type=str,
        default="reward",
        help="Metric to plot (for single-metric plots)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["reward", "steps", "success"],
        help="Metrics to plot (for multi-metric plots)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Window size for smoothing learning curves",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name for output file (without extension)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")

    # Display configuration
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots (only save)"
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible plotting"
    )

    return parser.parse_args()


def plot_learning_curves(args, plotter):
    """Generate learning curves plot."""
    if not args.results_dirs:
        print("Error: --results-dirs required for learning curves")
        return

    if args.agent_names and len(args.agent_names) != len(args.results_dirs):
        print("Error: Number of agent names must match number of results directories")
        return

    # Load training results
    results = {}
    for i, results_dir in enumerate(args.results_dirs):
        agent_name = args.agent_names[i] if args.agent_names else f"Agent {i+1}"
        try:
            data = load_training_results(results_dir)
            results[agent_name] = data["metrics"]
        except Exception as e:
            print(f"Warning: Could not load results from {results_dir}: {e}")

    if not results:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or f"learning_curves_{args.metric}"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_learning_curves(
        results=results,
        metric=args.metric,
        title=args.title,
        save_path=str(output_path),
        show=not args.no_show,
        smoothing_window=args.smoothing_window,
    )


def plot_multiple_metrics(args, plotter):
    """Generate multiple metrics plot."""
    if not args.results_dirs:
        print("Error: --results-dirs required for multiple metrics plot")
        return

    # Load training results
    results = {}
    for i, results_dir in enumerate(args.results_dirs):
        agent_name = args.agent_names[i] if args.agent_names else f"Agent {i+1}"
        try:
            data = load_training_results(results_dir)
            results[agent_name] = data["metrics"]
        except Exception as e:
            print(f"Warning: Could not load results from {results_dir}: {e}")

    if not results:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or "multiple_metrics"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_multiple_metrics(
        results=results,
        metrics=args.metrics,
        save_path=str(output_path),
        show=not args.no_show,
        smoothing_window=args.smoothing_window,
    )


def plot_ablation(args, plotter):
    """Generate ablation table."""
    if not args.eval_results or not args.eval_names:
        print("Error: --eval-results and --eval-names required for ablation table")
        return

    if len(args.eval_results) != len(args.eval_names):
        print("Error: Number of eval results must match number of eval names")
        return

    # Load evaluation results
    ablation_results = {}
    for name, result_path in zip(args.eval_names, args.eval_results):
        try:
            metrics = load_evaluation_results(result_path)
            ablation_results[name] = metrics
        except Exception as e:
            print(f"Warning: Could not load results from {result_path}: {e}")

    if not ablation_results:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or "ablation_table"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_ablation_table(
        ablation_results=ablation_results,
        save_path=str(output_path),
        show=not args.no_show,
    )


def plot_algorithm_comparison(args, plotter):
    """Generate algorithm comparison plot."""
    if not args.eval_results or not args.eval_names:
        print(
            "Error: --eval-results and --eval-names required for algorithm comparison"
        )
        return

    # Load evaluation results
    results = {}
    for name, result_path in zip(args.eval_names, args.eval_results):
        try:
            metrics = load_evaluation_results(result_path)
            results[name] = metrics
        except Exception as e:
            print(f"Warning: Could not load results from {result_path}: {e}")

    if not results:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or f"algorithm_comparison_{args.metric}"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_algorithm_comparison(
        results=results,
        metric=args.metric,
        title=args.title,
        save_path=str(output_path),
        show=not args.no_show,
    )


def plot_exploration_comparison(args, plotter):
    """Generate exploration strategy comparison plot."""
    if not args.eval_results or not args.eval_names:
        print(
            "Error: --eval-results and --eval-names required for exploration comparison"
        )
        return

    # Load evaluation results
    results = {}
    for name, result_path in zip(args.eval_names, args.eval_results):
        try:
            metrics = load_evaluation_results(result_path)
            results[name] = metrics
        except Exception as e:
            print(f"Warning: Could not load results from {result_path}: {e}")

    if not results:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or "exploration_comparison"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_exploration_comparison(
        results=results,
        save_path=str(output_path),
        show=not args.no_show,
    )


def plot_memory_comparison(args, plotter):
    """Generate memory comparison plot."""
    if not args.recurrent_result or not args.non_recurrent_result:
        print("Error: --recurrent-result and --non-recurrent-result required")
        return

    # Load results
    try:
        recurrent_metrics = load_evaluation_results(args.recurrent_result)
        non_recurrent_metrics = load_evaluation_results(args.non_recurrent_result)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Generate plot
    output_name = args.output_name or "memory_comparison"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_memory_comparison(
        recurrent_results=recurrent_metrics,
        non_recurrent_results=non_recurrent_metrics,
        save_path=str(output_path),
        show=not args.no_show,
    )


def plot_generalization(args, plotter):
    """Generate generalization plot."""
    if not args.in_dist_results or not args.ood_results or not args.eval_names:
        print("Error: --in-dist-results, --ood-results, and --eval-names required")
        return

    if len(args.in_dist_results) != len(args.ood_results) or len(
        args.in_dist_results
    ) != len(args.eval_names):
        print("Error: Number of in-dist, ood, and names must match")
        return

    # Load results
    in_dist_dict = {}
    ood_dict = {}

    for name, in_path, ood_path in zip(
        args.eval_names, args.in_dist_results, args.ood_results
    ):
        try:
            in_dist_dict[name] = load_evaluation_results(in_path)
            ood_dict[name] = load_evaluation_results(ood_path)
        except Exception as e:
            print(f"Warning: Could not load results for {name}: {e}")

    if not in_dist_dict or not ood_dict:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or "generalization"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_generalization(
        in_dist_results=in_dist_dict,
        out_dist_results=ood_dict,
        save_path=str(output_path),
        show=not args.no_show,
    )


def plot_degradation(args, plotter):
    """Generate performance degradation plot."""
    if not args.in_dist_results or not args.ood_results or not args.eval_names:
        print("Error: --in-dist-results, --ood-results, and --eval-names required")
        return

    # Load results
    in_dist_dict = {}
    ood_dict = {}

    for name, in_path, ood_path in zip(
        args.eval_names, args.in_dist_results, args.ood_results
    ):
        try:
            in_dist_dict[name] = load_evaluation_results(in_path)
            ood_dict[name] = load_evaluation_results(ood_path)
        except Exception as e:
            print(f"Warning: Could not load results for {name}: {e}")

    if not in_dist_dict or not ood_dict:
        print("Error: No valid results loaded")
        return

    # Generate plot
    output_name = args.output_name or f"degradation_{args.metric}"
    output_path = Path(args.output_dir) / f"{output_name}.{args.format}"

    plotter.plot_performance_degradation(
        in_dist_results=in_dist_dict,
        out_dist_results=ood_dict,
        metric=args.metric,
        save_path=str(output_path),
        show=not args.no_show,
    )


def main():
    """Main plotting function."""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plot generator
    plotter = PlotGenerator(dpi=args.dpi, random_seed=args.seed)

    # Generate requested plots
    if args.plot_type == "learning_curves":
        plot_learning_curves(args, plotter)
    elif args.plot_type == "multiple_metrics":
        plot_multiple_metrics(args, plotter)
    elif args.plot_type == "ablation":
        plot_ablation(args, plotter)
    elif args.plot_type == "algorithm_comparison":
        plot_algorithm_comparison(args, plotter)
    elif args.plot_type == "exploration_comparison":
        plot_exploration_comparison(args, plotter)
    elif args.plot_type == "memory_comparison":
        plot_memory_comparison(args, plotter)
    elif args.plot_type == "generalization":
        plot_generalization(args, plotter)
    elif args.plot_type == "degradation":
        plot_degradation(args, plotter)
    elif args.plot_type == "all":
        print("Generating all available plots...")
        # Try to generate each plot type if inputs are available
        if args.results_dirs:
            print("\n--- Learning Curves ---")
            plot_learning_curves(args, plotter)
            print("\n--- Multiple Metrics ---")
            plot_multiple_metrics(args, plotter)

        if args.eval_results and args.eval_names:
            print("\n--- Ablation Table ---")
            plot_ablation(args, plotter)
            print("\n--- Algorithm Comparison ---")
            plot_algorithm_comparison(args, plotter)
            print("\n--- Exploration Comparison ---")
            plot_exploration_comparison(args, plotter)

        if args.recurrent_result and args.non_recurrent_result:
            print("\n--- Memory Comparison ---")
            plot_memory_comparison(args, plotter)

        if args.in_dist_results and args.ood_results and args.eval_names:
            print("\n--- Generalization ---")
            plot_generalization(args, plotter)
            print("\n--- Performance Degradation ---")
            plot_degradation(args, plotter)

    print(f"\nPlots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
