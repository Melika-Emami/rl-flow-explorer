"""
Generate Generalization Plots

This script generates generalization visualization plots from saved evaluation results.
It creates:
1. In-distribution vs out-of-distribution comparison plots
2. Performance degradation visualizations
3. Coverage heatmaps

Usage:
    # Generate all generalization plots from default directory
    python generate_generalization_plots.py

    # Use custom results directory
    python generate_generalization_plots.py --results-dir generalization_results

    # Generate specific plot types
    python generate_generalization_plots.py --plots comparison degradation

    # Customize output
    python generate_generalization_plots.py --output-dir my_plots --dpi 300
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.evaluation.metrics import AggregateMetrics
from src.visualization.plot_generator import PlotGenerator


def load_generalization_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load generalization results from JSON files.

    Args:
        results_dir: Directory containing generalization result files

    Returns:
        Dictionary mapping agent names to their generalization results
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results = {}

    # Find all generalization result files
    for file_path in results_dir.glob("*_generalization.json"):
        agent_name = file_path.stem.replace("_generalization", "")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                results[agent_name] = data
                print(f"  [OK] Loaded results for {agent_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path}: {e}")

    return results


def extract_aggregate_metrics(results_dict: Dict) -> AggregateMetrics:
    """
    Extract AggregateMetrics from results dictionary.

    Args:
        results_dict: Dictionary containing metric values

    Returns:
        AggregateMetrics object
    """
    return AggregateMetrics(
        success_rate=results_dict.get("success_rate", 0.0),
        mean_steps_to_success=results_dict.get("mean_steps", 0.0),
        mean_reward=results_dict.get("mean_reward", 0.0),
        mean_coverage=results_dict.get("mean_coverage", 0.0),
        failure_rate=results_dict.get("failure_rate", 0.0),
    )


def generate_comparison_plot(
    results: Dict[str, Dict],
    plot_gen: PlotGenerator,
    output_dir: str,
    show: bool = False,
):
    """
    Generate in-distribution vs out-of-distribution comparison plot.

    Args:
        results: Dictionary of generalization results
        plot_gen: PlotGenerator instance
        output_dir: Output directory for plots
        show: Whether to display plots
    """
    print("\n  Generating generalization comparison plot...")

    # Prepare data
    agent_names = list(results.keys())
    shift_names = []

    # Get shift names from first agent
    if results:
        first_agent = next(iter(results.values()))
        if "out_of_distribution" in first_agent:
            shift_names = list(first_agent["out_of_distribution"].keys())

    if not shift_names:
        print("    [ERROR] No out-of-distribution results found")
        return

    # Extract success rates
    in_dist_success = []
    out_dist_success = {shift: [] for shift in shift_names}

    for agent_name in agent_names:
        agent_results = results[agent_name]

        # In-distribution
        in_dist = agent_results.get("in_distribution", {})
        in_dist_success.append(in_dist.get("success_rate", 0.0))

        # Out-of-distribution
        ood_results = agent_results.get("out_of_distribution", {})
        for shift in shift_names:
            shift_data = ood_results.get(shift, {})
            out_dist_success[shift].append(shift_data.get("success_rate", 0.0))

    # Generate plot
    save_path = Path(output_dir) / "generalization_comparison.png"
    plot_gen.plot_generalization_comparison(
        agent_names=agent_names,
        in_dist_success_rates=in_dist_success,
        out_dist_success_rates=out_dist_success,
        shift_names=shift_names,
        save_path=str(save_path),
        show=show,
    )

    print(f"    [OK] Saved to {save_path}")


def generate_degradation_heatmap(
    results: Dict[str, Dict],
    plot_gen: PlotGenerator,
    output_dir: str,
    show: bool = False,
):
    """
    Generate performance degradation heatmap.

    Args:
        results: Dictionary of generalization results
        plot_gen: PlotGenerator instance
        output_dir: Output directory for plots
        show: Whether to display plots
    """
    print("\n  Generating degradation heatmap...")

    # Prepare data
    agent_names = list(results.keys())
    shift_names = []

    # Get shift names from first agent
    if results:
        first_agent = next(iter(results.values()))
        if "degradation_metrics" in first_agent:
            shift_names = list(first_agent["degradation_metrics"].keys())

    if not shift_names:
        print("    [ERROR] No degradation metrics found")
        return

    # Extract degradation percentages
    degradation_matrix = []

    for agent_name in agent_names:
        agent_results = results[agent_name]
        degradation_metrics = agent_results.get("degradation_metrics", {})

        agent_degradations = []
        for shift in shift_names:
            shift_data = degradation_metrics.get(shift, {})
            degradation_pct = shift_data.get("success_rate_drop_pct", 0.0)
            agent_degradations.append(degradation_pct)

        degradation_matrix.append(agent_degradations)

    # Generate plot
    save_path = Path(output_dir) / "degradation_heatmap.png"
    plot_gen.plot_degradation_heatmap(
        agent_names=agent_names,
        shift_names=shift_names,
        degradation_matrix=degradation_matrix,
        save_path=str(save_path),
        show=show,
    )

    print(f"    [OK] Saved to {save_path}")


def generate_coverage_heatmap(
    results: Dict[str, Dict],
    plot_gen: PlotGenerator,
    output_dir: str,
    show: bool = False,
):
    """
    Generate coverage comparison heatmap.

    Args:
        results: Dictionary of generalization results
        plot_gen: PlotGenerator instance
        output_dir: Output directory for plots
        show: Whether to display plots
    """
    print("\n  Generating coverage heatmap...")

    # Prepare data
    agent_names = list(results.keys())
    shift_names = []

    # Get shift names
    if results:
        first_agent = next(iter(results.values()))
        if "out_of_distribution" in first_agent:
            shift_names = ["in_dist"] + list(first_agent["out_of_distribution"].keys())

    if not shift_names:
        print("    [ERROR] No coverage data found")
        return

    # Extract coverage values
    coverage_matrix = []

    for agent_name in agent_names:
        agent_results = results[agent_name]
        agent_coverage = []

        # In-distribution coverage
        in_dist = agent_results.get("in_distribution", {})
        agent_coverage.append(in_dist.get("mean_coverage", 0.0))

        # Out-of-distribution coverage
        ood_results = agent_results.get("out_of_distribution", {})
        for shift in shift_names[1:]:  # Skip "in_dist"
            shift_data = ood_results.get(shift, {})
            agent_coverage.append(shift_data.get("mean_coverage", 0.0))

        coverage_matrix.append(agent_coverage)

    # Convert to numpy array
    coverage_array = np.array(coverage_matrix)

    # Generate heatmap
    save_path = Path(output_dir) / "coverage_heatmap.png"

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6), dpi=plot_gen.dpi)

    # Create heatmap
    im = ax.imshow(
        coverage_array,
        cmap="YlGn",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(shift_names)))
    ax.set_yticks(np.arange(len(agent_names)))
    ax.set_xticklabels(shift_names, rotation=45, ha="right")
    ax.set_yticklabels(agent_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coverage", rotation=270, labelpad=20, fontsize=12)

    # Add text annotations
    for i in range(len(agent_names)):
        for j in range(len(shift_names)):
            value = coverage_array[i, j]
            text_color = "white" if value < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.1%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    # Title and layout
    ax.set_title("Coverage Heatmap: In-Dist vs Out-of-Dist", fontsize=14, pad=20)
    ax.set_xlabel("Test Set", fontsize=12)
    ax.set_ylabel("Agent", fontsize=12)
    plt.tight_layout()

    # Save
    fig.savefig(save_path, dpi=plot_gen.dpi, bbox_inches="tight")
    print(f"    [OK] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_detailed_metrics_plot(
    results: Dict[str, Dict],
    plot_gen: PlotGenerator,
    output_dir: str,
    show: bool = False,
):
    """
    Generate detailed metrics comparison across distribution shifts.

    Args:
        results: Dictionary of generalization results
        plot_gen: PlotGenerator instance
        output_dir: Output directory for plots
        show: Whether to display plots
    """
    print("\n  Generating detailed metrics plot...")

    import matplotlib.pyplot as plt

    # Prepare data
    agent_names = list(results.keys())
    metrics = ["success_rate", "mean_coverage", "mean_reward", "mean_steps"]
    metric_labels = ["Success Rate", "Coverage", "Mean Reward", "Steps to Success"]

    # Get shift names
    shift_names = []
    if results:
        first_agent = next(iter(results.values()))
        if "out_of_distribution" in first_agent:
            shift_names = ["in_dist"] + list(first_agent["out_of_distribution"].keys())

    if not shift_names:
        print("    [ERROR] No data found")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=plot_gen.dpi)
    axes = axes.flatten()

    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]

        # Extract data for this metric
        x_pos = np.arange(len(shift_names))
        width = 0.8 / len(agent_names)

        for agent_idx, agent_name in enumerate(agent_names):
            agent_results = results[agent_name]
            values = []

            # In-distribution
            in_dist = agent_results.get("in_distribution", {})
            values.append(in_dist.get(metric, 0.0))

            # Out-of-distribution
            ood_results = agent_results.get("out_of_distribution", {})
            for shift in shift_names[1:]:
                shift_data = ood_results.get(shift, {})
                values.append(shift_data.get(metric, 0.0))

            # Plot bars
            offset = (agent_idx - len(agent_names) / 2 + 0.5) * width
            bars = ax.bar(
                x_pos + offset,
                values,
                width,
                label=agent_name,
                alpha=0.8,
            )

        # Formatting
        ax.set_xlabel("Test Set", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(shift_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Generalization Metrics Across Distribution Shifts",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    # Save
    save_path = Path(output_dir) / "detailed_metrics_comparison.png"
    fig.savefig(save_path, dpi=plot_gen.dpi, bbox_inches="tight")
    print(f"    [OK] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(results: Dict[str, Dict]):
    """
    Print summary of generalization results.

    Args:
        results: Dictionary of generalization results
    """
    print("\n" + "=" * 80)
    print("GENERALIZATION RESULTS SUMMARY")
    print("=" * 80)

    for agent_name, agent_results in results.items():
        print(f"\n{agent_name.upper()}")
        print("-" * 80)

        # In-distribution
        in_dist = agent_results.get("in_distribution", {})
        print(f"\nIn-Distribution:")
        print(f"  Success Rate: {in_dist.get('success_rate', 0.0):.2%}")
        print(f"  Coverage:     {in_dist.get('mean_coverage', 0.0):.2%}")
        print(f"  Mean Reward:  {in_dist.get('mean_reward', 0.0):.3f}")

        # Out-of-distribution
        degradation_metrics = agent_results.get("degradation_metrics", {})
        if degradation_metrics:
            print(f"\nOut-of-Distribution Degradation:")
            for shift_name, metrics in degradation_metrics.items():
                degradation_pct = metrics.get("success_rate_drop_pct", 0.0)
                print(f"  {shift_name}: {degradation_pct:+.1f}%")

            # Average degradation
            avg_degradation = np.mean(
                [
                    m.get("success_rate_drop_pct", 0.0)
                    for m in degradation_metrics.values()
                ]
            )
            print(f"\n  Average Degradation: {avg_degradation:+.1f}%")

            # Assessment
            if abs(avg_degradation) < 10:
                assessment = "[OK] Excellent generalization"
            elif abs(avg_degradation) < 25:
                assessment = "[GOOD] Good generalization"
            else:
                assessment = "[ERROR] Poor generalization"
            print(f"  Assessment: {assessment}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate generalization visualization plots"
    )

    # Input/output
    parser.add_argument(
        "--results-dir",
        type=str,
        default="generalization_results",
        help="Directory containing generalization result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generalization_plots",
        help="Output directory for plots",
    )

    # Plot selection
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["comparison", "degradation", "coverage", "detailed", "all"],
        default=["all"],
        help="Which plots to generate",
    )

    # Plot options
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("GENERALIZATION PLOT GENERATION")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\n1. Loading generalization results...")
    try:
        results = load_generalization_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nPlease run generalization evaluation first:")
        print("  python run_generalization_evaluation.py")
        return

    if not results:
        print("\n[ERROR] No generalization results found!")
        print(
            f"\nExpected files: <agent_name>_generalization.json in {args.results_dir}"
        )
        return

    print(f"\nLoaded results for {len(results)} agent(s)")

    # Print summary
    print_summary(results)

    # Create plot generator
    plot_gen = PlotGenerator(
        output_dir=str(output_dir),
        dpi=args.dpi,
        random_seed=args.random_seed,
    )

    # Determine which plots to generate
    plot_types = args.plots
    if "all" in plot_types:
        plot_types = ["comparison", "degradation", "coverage", "detailed"]

    # Generate plots
    print("\n2. Generating plots...")

    if "comparison" in plot_types:
        generate_comparison_plot(results, plot_gen, output_dir, args.show)

    if "degradation" in plot_types:
        generate_degradation_heatmap(results, plot_gen, output_dir, args.show)

    if "coverage" in plot_types:
        generate_coverage_heatmap(results, plot_gen, output_dir, args.show)

    if "detailed" in plot_types:
        generate_detailed_metrics_plot(results, plot_gen, output_dir, args.show)

    print("\n" + "=" * 80)
    print("PLOT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")
    print("\n")


if __name__ == "__main__":
    main()
