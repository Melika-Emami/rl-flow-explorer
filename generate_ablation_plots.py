"""
Generate ablation tables and plots from experiment results.

This script creates comprehensive ablation visualizations including:
- Algorithm comparison tables
- Exploration strategy comparison plots
- Memory (recurrent vs non-recurrent) comparison plots
- Statistical significance tests
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats

from src.visualization.plot_generator import PlotGenerator
from src.evaluation.metrics import AggregateMetrics


def load_aggregated_results(results_path: str) -> Dict[str, Any]:
    """
    Load aggregated experiment results from JSON file.

    Args:
        results_path: Path to aggregated results JSON file

    Returns:
        Dictionary containing experiment results
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    return data


def convert_to_aggregate_metrics(data: Dict[str, Any]) -> AggregateMetrics:
    """
    Convert aggregated results to AggregateMetrics object.

    Args:
        data: Dictionary containing aggregated results

    Returns:
        AggregateMetrics object
    """
    results = data["results"]

    # Extract metrics with defaults for missing values
    success_rate = results.get("final_success_rate", {}).get("mean", 0.0)
    success_rate_std = results.get("final_success_rate", {}).get("std", 0.0)

    reward_mean = results.get("final_reward", {}).get("mean", 0.0)
    reward_std = results.get("final_reward", {}).get("std", 0.0)

    steps_mean = results.get("final_steps", {}).get("mean", 0.0)
    steps_std = results.get("final_steps", {}).get("std", 0.0)

    # Calculate confidence intervals (95% CI using t-distribution)
    num_seeds = results.get("num_seeds", 1)
    if num_seeds > 1:
        t_value = stats.t.ppf(0.975, num_seeds - 1)  # 95% CI
        success_ci = (
            success_rate - t_value * success_rate_std / np.sqrt(num_seeds),
            success_rate + t_value * success_rate_std / np.sqrt(num_seeds),
        )
        reward_ci = (
            reward_mean - t_value * reward_std / np.sqrt(num_seeds),
            reward_mean + t_value * reward_std / np.sqrt(num_seeds),
        )
        steps_ci = (
            steps_mean - t_value * steps_std / np.sqrt(num_seeds),
            steps_mean + t_value * steps_std / np.sqrt(num_seeds),
        )
    else:
        success_ci = (success_rate, success_rate)
        reward_ci = (reward_mean, reward_mean)
        steps_ci = (steps_mean, steps_mean)

    return AggregateMetrics(
        success_rate=success_rate,
        success_rate_std=success_rate_std,
        success_rate_ci=success_ci,
        mean_steps_to_success=steps_mean,
        std_steps_to_success=steps_std,
        steps_ci=steps_ci,
        failure_rate=1.0 - success_rate,
        failure_types={},
        mean_coverage=0.0,  # Not available in aggregated results
        std_coverage=0.0,
        coverage_ci=(0.0, 0.0),
        mean_reward=reward_mean,
        std_reward=reward_std,
        reward_ci=reward_ci,
        mean_episode_length=steps_mean,
        std_episode_length=steps_std,
        num_episodes=100,  # Default
        num_successes=int(success_rate * 100),
        num_failures=int((1.0 - success_rate) * 100),
    )


def convert_exploration_to_aggregate_metrics(
    config_data: Dict[str, Any],
) -> AggregateMetrics:
    """
    Convert exploration comparison results to AggregateMetrics object.

    This handles the nested structure of exploration_comparison_results.json
    where each configuration has its own metrics.

    Args:
        config_data: Dictionary containing a single configuration's results
                    (e.g., data for "dqn_epsilon_greedy")

    Returns:
        AggregateMetrics object
    """
    # Extract metrics - exploration results use "final_episode_success_rate"
    success_rate = config_data.get("final_episode_success_rate", {}).get("mean", 0.0)
    success_rate_std = config_data.get("final_episode_success_rate", {}).get("std", 0.0)

    reward_mean = config_data.get("final_reward", {}).get("mean", 0.0)
    reward_std = config_data.get("final_reward", {}).get("std", 0.0)

    steps_mean = config_data.get("final_steps", {}).get("mean", 0.0)
    steps_std = config_data.get("final_steps", {}).get("std", 0.0)

    # Calculate confidence intervals (95% CI using t-distribution)
    num_seeds = config_data.get("num_seeds", 1)
    if num_seeds > 1:
        t_value = stats.t.ppf(0.975, num_seeds - 1)  # 95% CI
        success_ci = (
            success_rate - t_value * success_rate_std / np.sqrt(num_seeds),
            success_rate + t_value * success_rate_std / np.sqrt(num_seeds),
        )
        reward_ci = (
            reward_mean - t_value * reward_std / np.sqrt(num_seeds),
            reward_mean + t_value * reward_std / np.sqrt(num_seeds),
        )
        steps_ci = (
            steps_mean - t_value * steps_std / np.sqrt(num_seeds),
            steps_mean + t_value * steps_std / np.sqrt(num_seeds),
        )
    else:
        success_ci = (success_rate, success_rate)
        reward_ci = (reward_mean, reward_mean)
        steps_ci = (steps_mean, steps_mean)

    return AggregateMetrics(
        success_rate=success_rate,
        success_rate_std=success_rate_std,
        success_rate_ci=success_ci,
        mean_steps_to_success=steps_mean,
        std_steps_to_success=steps_std,
        steps_ci=steps_ci,
        failure_rate=1.0 - success_rate,
        failure_types={},
        mean_coverage=0.0,  # Not available in exploration results
        std_coverage=0.0,
        coverage_ci=(0.0, 0.0),
        mean_reward=reward_mean,
        std_reward=reward_std,
        reward_ci=reward_ci,
        mean_episode_length=steps_mean,
        std_episode_length=steps_std,
        num_episodes=100,  # Default
        num_successes=int(success_rate * 100),
        num_failures=int((1.0 - success_rate) * 100),
    )


def perform_t_test(values1: List[float], values2: List[float]) -> Dict[str, float]:
    """
    Perform independent t-test between two groups.

    Args:
        values1: First group of values
        values2: Second group of values

    Returns:
        Dictionary with t-statistic and p-value
    """
    if len(values1) < 2 or len(values2) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_ind(values1, values2)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def generate_algorithm_comparison_table(
    results: Dict[str, AggregateMetrics], output_path: str
):
    """
    Generate a text table comparing algorithm performance.

    Args:
        results: Dictionary mapping algorithm names to metrics
        output_path: Path to save the table
    """
    lines = []
    lines.append("=" * 100)
    lines.append("ALGORITHM COMPARISON TABLE")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = (
        f"{'Algorithm':<25} {'Success Rate':<20} {'Mean Reward':<20} {'Mean Steps':<20}"
    )
    lines.append(header)
    lines.append("-" * 100)

    # Sort by success rate (descending)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].success_rate, reverse=True
    )

    # Data rows
    for name, metrics in sorted_results:
        success_str = f"{metrics.success_rate:.3f} ± {metrics.success_rate_std:.3f}"
        reward_str = f"{metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}"
        steps_str = (
            f"{metrics.mean_steps_to_success:.2f} ± {metrics.std_steps_to_success:.2f}"
        )

        row = f"{name:<25} {success_str:<20} {reward_str:<20} {steps_str:<20}"
        lines.append(row)

    lines.append("=" * 100)
    lines.append("")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print to console
    print("\n".join(lines))


def generate_statistical_tests(
    results_dict: Dict[str, Dict[str, Any]], output_path: str
):
    """
    Generate statistical significance tests between configurations.

    Args:
        results_dict: Dictionary mapping names to raw results data
        output_path: Path to save the test results
    """
    lines = []
    lines.append("=" * 100)
    lines.append("STATISTICAL SIGNIFICANCE TESTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Pairwise t-tests for Success Rate (α = 0.05)")
    lines.append("-" * 100)

    names = list(results_dict.keys())

    # Extract success rate values for each configuration
    success_values = {}
    for name, data in results_dict.items():
        values = data["results"].get("final_success_rate", {}).get("values", [])
        success_values[name] = values

    # Perform pairwise t-tests
    for i, name1 in enumerate(names):
        for name2 in names[i + 1 :]:
            if len(success_values[name1]) > 1 and len(success_values[name2]) > 1:
                test_result = perform_t_test(
                    success_values[name1], success_values[name2]
                )

                sig_marker = "***" if test_result["significant"] else "n.s."
                lines.append(
                    f"{name1} vs {name2}: t={test_result['t_statistic']:.3f}, "
                    f"p={test_result['p_value']:.4f} {sig_marker}"
                )
            else:
                lines.append(
                    f"{name1} vs {name2}: Insufficient data for t-test (need >1 seed)"
                )

    lines.append("")
    lines.append("Legend: *** = significant (p < 0.05), n.s. = not significant")
    lines.append("=" * 100)
    lines.append("")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print to console
    print("\n".join(lines))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ablation tables and plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input paths
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="baseline_results",
        help="Directory containing baseline experiment results",
    )
    parser.add_argument(
        "--exploration-dir",
        type=str,
        default="exploration_results",
        help="Directory containing exploration ablation results",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory_results",
        help="Directory containing memory ablation results",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation_plots",
        help="Directory to save plots and tables",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")

    # Display configuration
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots (only save)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible plotting"
    )

    return parser.parse_args()


def main():
    """Main function to generate ablation plots and tables."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize plot generator
    plotter = PlotGenerator(
        output_dir=str(output_dir), dpi=args.dpi, random_seed=args.seed
    )

    print("=" * 100)
    print("GENERATING ABLATION TABLES AND PLOTS")
    print("=" * 100)
    print("")

    # ========================================
    # 1. Algorithm Comparison (Baselines)
    # ========================================
    print("\n--- Loading Baseline Results ---")
    baseline_dir = Path(args.baseline_dir)

    if baseline_dir.exists():
        baseline_files = list(baseline_dir.glob("*_aggregated.json"))

        if baseline_files:
            baseline_results = {}
            baseline_raw = {}

            for file_path in baseline_files:
                # Extract algorithm name from filename
                name = file_path.stem.replace("_aggregated", "").replace(
                    "baseline_", ""
                )
                name = name.replace("_", " ").title()

                try:
                    data = load_aggregated_results(str(file_path))
                    baseline_raw[name] = data
                    baseline_results[name] = convert_to_aggregate_metrics(data)
                    print(f"  Loaded: {name}")
                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")

            if baseline_results:
                # Generate algorithm comparison table
                print("\n--- Generating Algorithm Comparison Table ---")
                table_path = output_dir / "algorithm_comparison_table.txt"
                generate_algorithm_comparison_table(baseline_results, str(table_path))

                # Generate algorithm comparison plot
                print("\n--- Generating Algorithm Comparison Plot ---")
                plot_path = output_dir / f"algorithm_comparison.{args.format}"
                plotter.plot_algorithm_comparison(
                    results=baseline_results,
                    metric="success_rate",
                    title="Algorithm Performance Comparison",
                    save_path=str(plot_path),
                    show=not args.no_show,
                )
                print(f"  Saved: {plot_path}")

                # Generate ablation table visualization
                print("\n--- Generating Ablation Table Visualization ---")
                ablation_plot_path = output_dir / f"ablation_table.{args.format}"
                plotter.plot_ablation_table(
                    ablation_results=baseline_results,
                    save_path=str(ablation_plot_path),
                    show=not args.no_show,
                )
                print(f"  Saved: {ablation_plot_path}")

                # Generate statistical tests
                if len(baseline_raw) > 1:
                    print("\n--- Generating Statistical Significance Tests ---")
                    stats_path = output_dir / "statistical_tests.txt"
                    generate_statistical_tests(baseline_raw, str(stats_path))
        else:
            print(f"  No baseline results found in {baseline_dir}")
    else:
        print(f"  Baseline directory not found: {baseline_dir}")

    # ========================================
    # 2. Exploration Strategy Comparison
    # ========================================
    print("\n--- Loading Exploration Ablation Results ---")
    exploration_dir = Path(args.exploration_dir)

    if exploration_dir.exists():
        # Look for the exploration comparison results file
        exploration_file = exploration_dir / "exploration_comparison_results.json"

        if exploration_file.exists():
            try:
                data = load_aggregated_results(str(exploration_file))
                exploration_results = {}

                # The exploration results have a nested structure
                # Each key in data["results"] is a configuration name
                for config_name, config_data in data["results"].items():
                    # Format the name nicely
                    name = config_name.replace("_", " ").title()
                    exploration_results[name] = (
                        convert_exploration_to_aggregate_metrics(config_data)
                    )
                    print(f"  Loaded: {name}")

                if exploration_results:
                    # Generate exploration comparison plot
                    print("\n--- Generating Exploration Strategy Comparison ---")
                    plot_path = output_dir / f"exploration_comparison.{args.format}"
                    plotter.plot_exploration_comparison(
                        results=exploration_results,
                        save_path=str(plot_path),
                        show=not args.no_show,
                    )
                    print(f"  Saved: {plot_path}")
            except Exception as e:
                print(f"  Warning: Could not load exploration results: {e}")
        else:
            print(f"  No exploration comparison results found: {exploration_file}")
    else:
        print(f"  Exploration directory not found: {exploration_dir}")

    # ========================================
    # 3. Memory Comparison (Recurrent vs Non-Recurrent)
    # ========================================
    print("\n--- Loading Memory Ablation Results ---")
    memory_dir = Path(args.memory_dir)

    if memory_dir.exists():
        # Look for recurrent and non-recurrent results
        recurrent_file = memory_dir / "recurrent_ppo_aggregated.json"
        non_recurrent_file = memory_dir / "ppo_aggregated.json"

        # Also check baseline directory for PPO results
        if not non_recurrent_file.exists():
            non_recurrent_file = baseline_dir / "baseline_ppo_medium_aggregated.json"

        if not recurrent_file.exists():
            recurrent_file = (
                baseline_dir / "baseline_recurrent_ppo_medium_aggregated.json"
            )

        if recurrent_file.exists() and non_recurrent_file.exists():
            try:
                recurrent_data = load_aggregated_results(str(recurrent_file))
                non_recurrent_data = load_aggregated_results(str(non_recurrent_file))

                recurrent_metrics = convert_to_aggregate_metrics(recurrent_data)
                non_recurrent_metrics = convert_to_aggregate_metrics(non_recurrent_data)

                print(f"  Loaded: Recurrent PPO")
                print(f"  Loaded: Non-Recurrent PPO")

                # Generate memory comparison plot
                print("\n--- Generating Memory Comparison Plot ---")
                plot_path = output_dir / f"memory_comparison.{args.format}"
                plotter.plot_memory_comparison(
                    recurrent_results=recurrent_metrics,
                    non_recurrent_results=non_recurrent_metrics,
                    save_path=str(plot_path),
                    show=not args.no_show,
                )
                print(f"  Saved: {plot_path}")

            except Exception as e:
                print(f"  Warning: Could not load memory comparison results: {e}")
        else:
            print(f"  Memory comparison files not found")
            if not recurrent_file.exists():
                print(f"    Missing: {recurrent_file}")
            if not non_recurrent_file.exists():
                print(f"    Missing: {non_recurrent_file}")
    else:
        print(f"  Memory directory not found: {memory_dir}")

    print("\n" + "=" * 100)
    print(f"ABLATION PLOTS AND TABLES SAVED TO: {output_dir}")
    print("=" * 100)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
