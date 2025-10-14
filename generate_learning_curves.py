"""
Script to generate all learning curve plots.

This script creates:
1. Individual learning curves for each agent with confidence bands
2. Comparison plots across all algorithms
3. Multi-metric learning curves
4. High-resolution figures for publication
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.visualization.plot_generator import PlotGenerator


def find_experiment_results(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all experiment results in a directory.

    Args:
        base_dir: Base directory to search

    Returns:
        Dictionary mapping experiment names to list of result directories
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return {}

    results = {}

    # Look for training_metrics.json files
    for metrics_file in base_path.rglob("training_metrics.json"):
        # Get experiment name from parent directory
        exp_dir = metrics_file.parent
        exp_name = exp_dir.name

        # Group by agent type (extract from experiment name)
        agent_type = None
        for agent in ["qlearning", "dqn", "recurrent_ppo", "ppo"]:
            if agent in exp_name.lower():
                agent_type = agent
                break

        if agent_type:
            if agent_type not in results:
                results[agent_type] = []
            results[agent_type].append(str(exp_dir))

    return results


def load_all_seeds(result_dirs: List[str]) -> List[Dict]:
    """
    Load training metrics from multiple seed directories.

    Args:
        result_dirs: List of result directories

    Returns:
        List of training metrics dictionaries
    """
    all_metrics = []

    for result_dir in result_dirs:
        metrics_path = Path(result_dir) / "training_metrics.json"

        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)

                # Handle different JSON formats
                if isinstance(data, dict) and "metrics" in data:
                    all_metrics.append(data["metrics"])
                elif isinstance(data, list):
                    all_metrics.append(data)
            except Exception as e:
                print(f"Warning: Could not load {metrics_path}: {e}")

    return all_metrics


def aggregate_metrics_across_seeds(
    all_seeds: List[List[Dict]], smoothing_window: int = 10
) -> tuple:
    """
    Aggregate metrics across multiple seeds with confidence bands.

    Args:
        all_seeds: List of metric lists (one per seed)
        smoothing_window: Window for smoothing

    Returns:
        Tuple of (episodes, mean_values, std_values) for each metric
    """
    if not all_seeds:
        return {}, np.array([]), {}

    # Find common metrics
    if not all_seeds[0]:
        return {}, np.array([]), {}

    metrics_names = set(all_seeds[0][0].keys()) - {"episode"}

    # Get maximum episode count
    max_episodes = max(len(seed_metrics) for seed_metrics in all_seeds)

    # Aggregate each metric
    aggregated = {}
    episodes = None

    for metric_name in metrics_names:
        # Collect values for this metric across all seeds
        all_values = []

        for seed_metrics in all_seeds:
            values = []
            eps = []

            for m in seed_metrics:
                if "episode" in m and metric_name in m:
                    eps.append(m["episode"])
                    values.append(float(m[metric_name]))

            if values:
                # Apply smoothing
                if smoothing_window > 1:
                    values = moving_average(np.array(values), smoothing_window)
                    eps = eps[: len(values)]

                all_values.append(values)

                if episodes is None:
                    episodes = np.array(eps)

        if all_values:
            # Pad sequences to same length
            max_len = max(len(v) for v in all_values)
            padded_values = []

            for values in all_values:
                if len(values) < max_len:
                    # Pad with last value
                    padded = np.pad(
                        values,
                        (0, max_len - len(values)),
                        mode="edge",
                    )
                else:
                    padded = values
                padded_values.append(padded)

            # Compute mean and std
            values_array = np.array(padded_values)
            mean_values = np.mean(values_array, axis=0)
            std_values = np.std(values_array, axis=0)

            aggregated[metric_name] = {
                "mean": mean_values,
                "std": std_values,
            }

    return aggregated, episodes, metrics_names


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average."""
    if window <= 1:
        return data

    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def create_single_agent_curves(
    agent_name: str,
    result_dirs: List[str],
    output_dir: Path,
    plotter: PlotGenerator,
    smoothing_window: int = 10,
    dpi: int = 300,
):
    """
    Create learning curves for a single agent across all metrics.

    Args:
        agent_name: Name of the agent
        result_dirs: List of result directories for different seeds
        output_dir: Output directory for plots
        plotter: PlotGenerator instance
        smoothing_window: Smoothing window size
        dpi: DPI for output images
    """
    print(f"\nGenerating learning curves for {agent_name}...")

    # Load all seeds
    all_seeds = load_all_seeds(result_dirs)

    if not all_seeds:
        print(f"  Warning: No data found for {agent_name}")
        return

    print(f"  Loaded {len(all_seeds)} seeds")

    # Aggregate metrics
    aggregated, episodes, metric_names = aggregate_metrics_across_seeds(
        all_seeds, smoothing_window
    )

    if not aggregated:
        print(f"  Warning: No metrics to plot for {agent_name}")
        return

    # Create individual plots for key metrics
    key_metrics = ["reward", "success", "steps", "coverage"]

    for metric in key_metrics:
        if metric not in aggregated:
            continue

        # Create plot data
        metric_values = aggregated[metric].copy()
        metric_values["episodes"] = episodes
        results = {agent_name: metric_values}

        # Generate plot
        output_path = output_dir / f"{agent_name}_{metric}_curve.png"

        try:
            plotter.plot_learning_curves(
                results=results,
                metric=metric,
                title=f"{agent_name.upper()} Learning Curve: {metric.title()}",
                save_path=str(output_path),
                show=False,
                smoothing_window=smoothing_window,
            )
            print(f"  ✓ Saved {metric} curve to {output_path}")
        except Exception as e:
            print(f"  ✗ Error creating {metric} plot: {e}")

    # Create multi-metric plot
    output_path = output_dir / f"{agent_name}_all_metrics.png"

    try:
        # Create plot data
        metric_values = aggregated.copy()
        for m_val in metric_values.key():
            metric_values[m_val]["episodes"] = episodes
        results = {agent_name: metric_values}

        results = {agent_name: all_seeds}
        plotter.plot_multiple_metrics(
            results=results,
            metrics=key_metrics,
            save_path=str(output_path),
            show=False,
            smoothing_window=smoothing_window,
        )
        print(f"  ✓ Saved multi-metric plot to {output_path}")
    except Exception as e:
        print(f"  ✗ Error creating multi-metric plot: {e}")


def create_algorithm_comparison_curves(
    all_results: Dict[str, List[str]],
    output_dir: Path,
    plotter: PlotGenerator,
    smoothing_window: int = 10,
    dpi: int = 300,
):
    """
    Create comparison plots across all algorithms.

    Args:
        all_results: Dictionary mapping agent names to result directories
        output_dir: Output directory for plots
        plotter: PlotGenerator instance
        smoothing_window: Smoothing window size
        dpi: DPI for output images
    """
    print("\nGenerating algorithm comparison plots...")

    # Load all agent data
    agent_data = {}

    for agent_name, result_dirs in all_results.items():
        all_seeds = load_all_seeds(result_dirs)
        if all_seeds:
            agent_data[agent_name] = all_seeds
            print(f"  Loaded {agent_name}: {len(all_seeds)} seeds")

    if len(agent_data) < 2:
        print("  Warning: Need at least 2 agents for comparison")
        return

    # Create comparison plots for key metrics
    key_metrics = ["reward", "success", "steps", "coverage"]

    for metric in key_metrics:
        output_path = output_dir / f"algorithm_comparison_{metric}.png"

        try:
            plotter.plot_learning_curves(
                results=agent_data,
                metric=metric,
                title=f"Algorithm Comparison: {metric.title()}",
                save_path=str(output_path),
                show=False,
                smoothing_window=smoothing_window,
            )
            print(f"  ✓ Saved {metric} comparison to {output_path}")
        except Exception as e:
            print(f"  ✗ Error creating {metric} comparison: {e}")

    # Create comprehensive multi-metric comparison
    output_path = output_dir / "algorithm_comparison_all_metrics.png"

    try:
        plotter.plot_multiple_metrics(
            results=agent_data,
            metrics=key_metrics,
            save_path=str(output_path),
            show=False,
            smoothing_window=smoothing_window,
        )
        print(f"  ✓ Saved multi-metric comparison to {output_path}")
    except Exception as e:
        print(f"  ✗ Error creating multi-metric comparison: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate all learning curve plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="baseline_logs",
        help="Directory containing experiment logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="learning_curve_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Window size for smoothing curves",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output images (high-resolution)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible plotting",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help="Specific agents to plot (default: all found)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEARNING CURVE GENERATION")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Smoothing window: {args.smoothing_window}")
    print(f"DPI: {args.dpi}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Find all experiment results
    print("\nSearching for experiment results...")
    all_results = find_experiment_results(args.results_dir)

    if not all_results:
        print(f"\n⚠️  No experiment results found in {args.results_dir}")
        print("\nPlease run baseline experiments first:")
        print("  python run_baseline_experiments.py")
        return 1

    print(f"\nFound results for {len(all_results)} agent types:")
    for agent_name, result_dirs in all_results.items():
        print(f"  - {agent_name}: {len(result_dirs)} runs")

    # Filter agents if specified
    if args.agents:
        all_results = {k: v for k, v in all_results.items() if k in args.agents}
        print(f"\nFiltered to {len(all_results)} agents: {list(all_results.keys())}")

    # Create plot generator
    plotter = PlotGenerator(
        output_dir=str(output_dir),
        dpi=args.dpi,
        random_seed=args.seed,
    )

    # Generate individual agent curves
    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL AGENT LEARNING CURVES")
    print("=" * 70)

    for agent_name, result_dirs in all_results.items():
        create_single_agent_curves(
            agent_name=agent_name,
            result_dirs=result_dirs,
            output_dir=output_dir,
            plotter=plotter,
            smoothing_window=args.smoothing_window,
            dpi=args.dpi,
        )

    # Generate comparison curves
    print("\n" + "=" * 70)
    print("GENERATING ALGORITHM COMPARISON CURVES")
    print("=" * 70)

    create_algorithm_comparison_curves(
        all_results=all_results,
        output_dir=output_dir,
        plotter=plotter,
        smoothing_window=args.smoothing_window,
        dpi=args.dpi,
    )

    # Summary
    print("\n" + "=" * 70)
    print("LEARNING CURVE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  Individual agent curves:")
    for agent_name in all_results.keys():
        print(f"    - {agent_name}_<metric>_curve.png")
        print(f"    - {agent_name}_all_metrics.png")
    print("  Comparison plots:")
    print("    - algorithm_comparison_<metric>.png")
    print("    - algorithm_comparison_all_metrics.png")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
