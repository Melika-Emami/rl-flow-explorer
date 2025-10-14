"""
PlotGenerator: Creates publication-quality plots for RL experiments.

This module provides functionality to generate learning curves, ablation
comparisons, and generalization visualizations with confidence bands.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..evaluation.metrics import AggregateMetrics
from ..evaluation.evaluator import EvaluationResults


class PlotGenerator:
    """
    Generates publication-quality plots for RL experiments.

    This class creates standardized visualizations including learning curves
    with confidence bands, ablation comparisons, and generalization analysis.
    All plots use deterministic styling for reproducibility.
    """

    def __init__(
        self,
        output_dir: str = "plots",
        style: str = "seaborn-v0_8-darkgrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        random_seed: int = 42,
    ):
        """
        Initialize the PlotGenerator.

        Args:
            output_dir: Default output directory for saving plots
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
            dpi: Dots per inch for figure resolution
            random_seed: Random seed for deterministic plotting
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Set style
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            plt.style.use("default")
            sns.set_theme()

        # Define consistent color palette
        self.colors = sns.color_palette("husl", 10)

        # Font sizes
        self.title_fontsize = 14
        self.label_fontsize = 12
        self.tick_fontsize = 10
        self.legend_fontsize = 10

    def plot_learning_curves(
        self,
        results: Dict[str, Union[Dict[str, Any], str]],
        metric: str = "reward",
        xlabel: str = "Episode",
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        smoothing_window: int = 10,
    ) -> Figure:
        """
        Plot learning curves with confidence bands over multiple seeds.

        Args:
            results: Dictionary mapping agent names to either:
                - Training metrics dictionary
                - Path to JSON file containing training metrics
            metric: Metric to plot (e.g., 'reward', 'success', 'steps')
            xlabel: Label for x-axis
            ylabel: Label for y-axis (auto-generated if None)
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot
            smoothing_window: Window size for moving average smoothing

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Auto-generate labels if not provided
        if ylabel is None:
            ylabel = metric.replace("_", " ").title()
        if title is None:
            title = f"Learning Curves: {ylabel}"

        # Plot each agent's learning curve
        for idx, (agent_name, data) in enumerate(results.items()):
            # Load data if path provided
            if isinstance(data, str):
                data = self._load_training_metrics(data)

            # Extract metric values
            # episodes, mean_values, std_values = self._extract_metric_over_time(
            #     data, metric, smoothing_window
            # )

            episodes = data["episodes"]
            mean_values = data["mean"]
            std_values = data["std"]

            if len(episodes) == 0:
                print(f"Warning: No data for {agent_name}")
                continue

            # Get color for this agent
            color = self.colors[idx % len(self.colors)]

            # Plot mean line
            ax.plot(
                episodes,
                mean_values,
                label=agent_name,
                color=color,
                linewidth=2,
            )

            # Plot confidence band (mean ± std)
            ax.fill_between(
                episodes,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
                color=color,
            )

        # Formatting
        ax.set_xlabel(xlabel, fontsize=self.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.label_fontsize)
        ax.set_title(title, fontsize=self.title_fontsize)
        ax.legend(fontsize=self.legend_fontsize, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=self.tick_fontsize)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_multiple_metrics(
        self,
        results: Dict[str, Union[Dict[str, Dict[str, Any]], str]],
        metrics: List[str],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        smoothing_window: int = 10,
    ) -> Figure:
        """
        Plot multiple metrics in subplots.

        Args:
            results: Dictionary mapping agent names to training metrics
            metrics: List of metrics to plot
            titles: List of subplot titles (auto-generated if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot
            smoothing_window: Window size for moving average smoothing

        Returns:
            Matplotlib Figure object
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0] * n_cols / 2, self.figsize[1] * n_rows / 2),
            dpi=self.dpi,
        )

        # Ensure axes is always 2D array
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each metric
        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Generate title
            if titles and idx < len(titles):
                title = titles[idx]
            else:
                title = metric.replace("_", " ").title()

            # Plot each agent
            for agent_idx, (agent_name, data) in enumerate(results.items()):
                # Load data if path provided
                if isinstance(data, str):
                    data = self._load_training_metrics(data)

                # Extract metric values
                # episodes, mean_values, std_values = self._extract_metric_over_time(
                #     data, metric, smoothing_window
                # )
                episodes = data["episodes"]
                mean_values = data["mean"]
                std_values = data["std"]

                if len(episodes) == 0:
                    continue

                # Get color
                color = self.colors[agent_idx % len(self.colors)]

                # Plot
                ax.plot(
                    episodes, mean_values, label=agent_name, color=color, linewidth=2
                )
                ax.fill_between(
                    episodes,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.2,
                    color=color,
                )

            # Formatting
            ax.set_xlabel("Episode", fontsize=self.label_fontsize)
            ax.set_ylabel(title, fontsize=self.label_fontsize)
            ax.set_title(title, fontsize=self.title_fontsize)
            ax.legend(fontsize=self.legend_fontsize - 2, loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=self.tick_fontsize)

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def _load_training_metrics(self, path: str) -> List[Dict[str, Any]]:
        # TODO: modify output structure
        """
        Load training metrics from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of metric dictionaries
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, dict) and "metrics" in data:
            return data["metrics"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected JSON format in {path}")

    def _extract_metric_over_time(
        self,
        metrics: List[Dict[str, Any]],
        metric_name: str,
        smoothing_window: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract metric values over time with optional smoothing.

        Args:
            metrics: List of metric dictionaries
            metric_name: Name of metric to extract
            smoothing_window: Window size for moving average

        Returns:
            Tuple of (episodes, mean_values, std_values)
        """
        if not metrics:
            return np.array([]), np.array([]), np.array([])

        # Extract episodes and values
        episodes = []
        values = []

        for seed_m in metrics:
            seed_ep = []
        for m in metrics:
            if "episode" in m and metric_name in m:
                episodes.append(m["episode"])
                values.append(float(m[metric_name]))

        if not episodes:
            return np.array([]), np.array([]), np.array([])

        episodes = np.array(episodes)
        values = np.array(values)

        # Apply smoothing if requested
        if smoothing_window > 1:
            values = self._moving_average(values, smoothing_window)
            # Adjust episodes to match smoothed values length
            episodes = episodes[: len(values)]

        # For single seed, std is zero
        std_values = np.zeros_like(values)

        return episodes, values, std_values

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Compute moving average of data.

        Args:
            data: Input data array
            window: Window size

        Returns:
            Smoothed data array
        """
        if window <= 1:
            return data

        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    def plot_ablation_table(
        self,
        ablation_results: Dict[str, AggregateMetrics],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Generate ablation comparison table as a figure.

        Args:
            ablation_results: Dictionary mapping configuration names to AggregateMetrics
            metrics: List of metrics to include (uses default set if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        # Default metrics to display
        if metrics is None:
            metrics = [
                "success_rate",
                "mean_steps_to_success",
                "mean_coverage",
                "mean_reward",
            ]

        # Prepare data for table
        config_names = list(ablation_results.keys())
        table_data = []

        for config_name in config_names:
            agg_metrics = ablation_results[config_name]
            row = [config_name]

            for metric in metrics:
                value = getattr(agg_metrics, metric, None)
                if value is not None:
                    # Format based on metric type
                    if "rate" in metric or "coverage" in metric:
                        row.append(f"{value:.2%}")
                    else:
                        row.append(f"{value:.2f}")
                else:
                    row.append("N/A")

            table_data.append(row)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(config_names) * 0.5 + 2), dpi=self.dpi)
        ax.axis("tight")
        ax.axis("off")

        # Create table
        column_labels = ["Configuration"] + [
            m.replace("_", " ").title() for m in metrics
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.3] + [0.2] * len(metrics),
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(len(column_labels)):
            cell = table[(0, i)]
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(weight="bold", color="white")

        # Alternate row colors
        for i in range(1, len(config_names) + 1):
            for j in range(len(column_labels)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#f0f0f0")

        plt.title("Ablation Study Results", fontsize=self.title_fontsize, pad=20)

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved ablation table to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_algorithm_comparison(
        self,
        results: Dict[str, AggregateMetrics],
        metric: str = "success_rate",
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Create bar plot comparing different algorithms.

        Args:
            results: Dictionary mapping algorithm names to AggregateMetrics
            metric: Metric to compare
            ylabel: Label for y-axis (auto-generated if None)
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Auto-generate labels if not provided
        if ylabel is None:
            ylabel = metric.replace("_", " ").title()
        if title is None:
            title = f"Algorithm Comparison: {ylabel}"

        # Extract data
        algorithm_names = list(results.keys())
        values = []
        errors = []

        for name in algorithm_names:
            agg_metrics = results[name]
            value = getattr(agg_metrics, metric, 0.0)
            values.append(value)

            # Get error (std or CI width)
            std_attr = f"{metric}_std"
            if hasattr(agg_metrics, std_attr):
                error = getattr(agg_metrics, std_attr)
            else:
                # Try to get from CI
                ci_attr = f"{metric}_ci"
                if hasattr(agg_metrics, ci_attr):
                    ci = getattr(agg_metrics, ci_attr)
                    error = (ci[1] - ci[0]) / 2
                else:
                    error = 0.0
            errors.append(error)

        # Create bar plot
        x_pos = np.arange(len(algorithm_names))
        bars = ax.bar(
            x_pos,
            values,
            yerr=errors,
            capsize=5,
            color=self.colors[: len(algorithm_names)],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if "rate" in metric or "coverage" in metric:
                label = f"{value:.1%}"
            else:
                label = f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + errors[i],
                label,
                ha="center",
                va="bottom",
                fontsize=self.tick_fontsize,
            )

        # Formatting
        ax.set_xlabel("Algorithm", fontsize=self.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.label_fontsize)
        ax.set_title(title, fontsize=self.title_fontsize)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithm_names, rotation=45, ha="right")
        ax.tick_params(labelsize=self.tick_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved comparison plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_exploration_comparison(
        self,
        results: Dict[str, AggregateMetrics],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Create comparison plot for different exploration strategies.

        Args:
            results: Dictionary mapping exploration strategy names to AggregateMetrics
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        # Create subplots for multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        axes = axes.flatten()

        metrics = [
            ("success_rate", "Success Rate"),
            ("mean_coverage", "Coverage"),
            ("mean_reward", "Mean Reward"),
            ("mean_steps_to_success", "Steps to Success"),
        ]

        strategy_names = list(results.keys())
        x_pos = np.arange(len(strategy_names))

        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx]

            # Extract data
            values = []
            errors = []

            for name in strategy_names:
                agg_metrics = results[name]
                value = getattr(agg_metrics, metric, 0.0)
                values.append(value)

                # Get error
                std_attr = f"{metric}_std"
                if hasattr(agg_metrics, std_attr):
                    error = getattr(agg_metrics, std_attr)
                else:
                    error = 0.0
                errors.append(error)

            # Create bar plot
            bars = ax.bar(
                x_pos,
                values,
                yerr=errors,
                capsize=5,
                color=self.colors[: len(strategy_names)],
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add value labels
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                if "rate" in metric or "coverage" in metric:
                    text = f"{value:.1%}"
                else:
                    text = f"{value:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + error,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Formatting
            ax.set_ylabel(label, fontsize=self.label_fontsize)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategy_names, rotation=45, ha="right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Exploration Strategy Comparison",
            fontsize=self.title_fontsize + 2,
            y=0.995,
        )
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved exploration comparison to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_memory_comparison(
        self,
        recurrent_results: AggregateMetrics,
        non_recurrent_results: AggregateMetrics,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Visualize contribution of memory (recurrent vs non-recurrent).

        Args:
            recurrent_results: Results from recurrent agent
            non_recurrent_results: Results from non-recurrent agent
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        # Metrics to compare
        metrics = [
            ("success_rate", "Success Rate", True),
            ("mean_coverage", "Coverage", True),
            ("mean_reward", "Mean Reward", False),
            ("mean_steps_to_success", "Steps to Success", False),
        ]

        # Left plot: Bar comparison
        ax = axes[0]
        metric_names = [m[1] for m in metrics]
        recurrent_values = []
        non_recurrent_values = []

        for metric, _, _ in metrics:
            recurrent_values.append(getattr(recurrent_results, metric, 0.0))
            non_recurrent_values.append(getattr(non_recurrent_results, metric, 0.0))

        x_pos = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax.bar(
            x_pos - width / 2,
            recurrent_values,
            width,
            label="Recurrent",
            color=self.colors[0],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x_pos + width / 2,
            non_recurrent_values,
            width,
            label="Non-Recurrent",
            color=self.colors[1],
            alpha=0.8,
        )

        ax.set_ylabel("Value", fontsize=self.label_fontsize)
        ax.set_title("Memory Contribution", fontsize=self.title_fontsize)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        # Right plot: Improvement percentage
        ax = axes[1]
        improvements = []

        for i, (metric, label, is_percentage) in enumerate(metrics):
            rec_val = recurrent_values[i]
            non_rec_val = non_recurrent_values[i]

            if non_rec_val != 0:
                improvement = ((rec_val - non_rec_val) / abs(non_rec_val)) * 100
            else:
                improvement = 0.0

            improvements.append(improvement)

        colors_improvement = [
            self.colors[2] if imp > 0 else self.colors[3] for imp in improvements
        ]
        bars = ax.barh(metric_names, improvements, color=colors_improvement, alpha=0.8)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:+.1f}%",
                ha="left" if imp > 0 else "right",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Improvement (%)", fontsize=self.label_fontsize)
        ax.set_title("Recurrent vs Non-Recurrent", fontsize=self.title_fontsize)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved memory comparison to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_generalization(
        self,
        in_dist_results: Dict[str, AggregateMetrics],
        out_dist_results: Dict[str, AggregateMetrics],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Compare in-distribution vs out-of-distribution performance.

        Args:
            in_dist_results: Dictionary mapping agent names to in-dist AggregateMetrics
            out_dist_results: Dictionary mapping agent names to out-of-dist AggregateMetrics
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        axes = axes.flatten()

        metrics = [
            ("success_rate", "Success Rate"),
            ("mean_coverage", "Coverage"),
            ("mean_reward", "Mean Reward"),
            ("mean_steps_to_success", "Steps to Success"),
        ]

        agent_names = list(in_dist_results.keys())
        x_pos = np.arange(len(agent_names))
        width = 0.35

        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx]

            # Extract in-dist values
            in_dist_values = []
            in_dist_errors = []
            for name in agent_names:
                value = getattr(in_dist_results[name], metric, 0.0)
                in_dist_values.append(value)

                std_attr = f"{metric}_std"
                error = getattr(in_dist_results[name], std_attr, 0.0)
                in_dist_errors.append(error)

            # Extract out-of-dist values
            out_dist_values = []
            out_dist_errors = []
            for name in agent_names:
                value = getattr(out_dist_results[name], metric, 0.0)
                out_dist_values.append(value)

                std_attr = f"{metric}_std"
                error = getattr(out_dist_results[name], std_attr, 0.0)
                out_dist_errors.append(error)

            # Create grouped bar plot
            bars1 = ax.bar(
                x_pos - width / 2,
                in_dist_values,
                width,
                yerr=in_dist_errors,
                label="In-Distribution",
                color=self.colors[0],
                alpha=0.8,
                capsize=3,
            )
            bars2 = ax.bar(
                x_pos + width / 2,
                out_dist_values,
                width,
                yerr=out_dist_errors,
                label="Out-of-Distribution",
                color=self.colors[1],
                alpha=0.8,
                capsize=3,
            )

            # Formatting
            ax.set_ylabel(label, fontsize=self.label_fontsize)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=9)
            ax.legend(fontsize=self.legend_fontsize - 2)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Generalization: In-Distribution vs Out-of-Distribution",
            fontsize=self.title_fontsize + 2,
            y=0.995,
        )
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved generalization plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_performance_degradation(
        self,
        in_dist_results: Dict[str, AggregateMetrics],
        out_dist_results: Dict[str, AggregateMetrics],
        metric: str = "success_rate",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Create bar plot showing performance degradation.

        Args:
            in_dist_results: Dictionary mapping agent names to in-dist AggregateMetrics
            out_dist_results: Dictionary mapping agent names to out-of-dist AggregateMetrics
            metric: Metric to analyze for degradation
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        agent_names = list(in_dist_results.keys())
        degradations = []

        for name in agent_names:
            in_dist_value = getattr(in_dist_results[name], metric, 0.0)
            out_dist_value = getattr(out_dist_results[name], metric, 0.0)

            # Calculate degradation percentage
            if in_dist_value != 0:
                degradation = ((in_dist_value - out_dist_value) / in_dist_value) * 100
            else:
                degradation = 0.0

            degradations.append(degradation)

        # Create bar plot
        x_pos = np.arange(len(agent_names))
        colors_deg = [
            self.colors[3] if deg > 0 else self.colors[2] for deg in degradations
        ]
        bars = ax.bar(
            x_pos,
            degradations,
            color=colors_deg,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels
        for bar, deg in zip(bars, degradations):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{deg:.1f}%",
                ha="center",
                va="bottom" if deg > 0 else "top",
                fontsize=self.tick_fontsize,
            )

        # Formatting
        ax.set_xlabel("Agent", fontsize=self.label_fontsize)
        ax.set_ylabel("Performance Degradation (%)", fontsize=self.label_fontsize)
        ax.set_title(
            f"Performance Degradation: {metric.replace('_', ' ').title()}",
            fontsize=self.title_fontsize,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_names, rotation=45, ha="right")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=self.tick_fontsize)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved degradation plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_coverage_heatmap(
        self,
        state_visitation: np.ndarray,
        state_labels: Optional[List[str]] = None,
        title: str = "State Visitation Heatmap",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Create heatmap showing state visitation patterns.

        Args:
            state_visitation: 2D array of visitation counts (agents x states)
            state_labels: Optional labels for states
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)

        # Normalize visitation counts
        normalized_visitation = state_visitation / (
            state_visitation.sum(axis=1, keepdims=True) + 1e-10
        )

        # Create heatmap
        im = ax.imshow(
            normalized_visitation,
            cmap="YlOrRd",
            aspect="auto",
            interpolation="nearest",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Visitation Frequency", fontsize=self.label_fontsize)

        # Set labels
        ax.set_xlabel("State", fontsize=self.label_fontsize)
        ax.set_ylabel("Agent/Episode", fontsize=self.label_fontsize)
        ax.set_title(title, fontsize=self.title_fontsize)

        # Set ticks
        if state_labels and len(state_labels) <= 20:
            ax.set_xticks(np.arange(len(state_labels)))
            ax.set_xticklabels(state_labels, rotation=90, fontsize=8)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved coverage heatmap to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_coverage_comparison(
        self,
        coverage_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Compare coverage patterns across different agents.

        Args:
            coverage_data: Dictionary mapping agent names to coverage arrays
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        n_agents = len(coverage_data)
        fig, axes = plt.subplots(
            1,
            n_agents,
            figsize=(5 * n_agents, 5),
            dpi=self.dpi,
        )

        # Ensure axes is iterable
        if n_agents == 1:
            axes = [axes]

        for idx, (agent_name, coverage) in enumerate(coverage_data.items()):
            ax = axes[idx]

            # Normalize
            normalized = coverage / (coverage.sum() + 1e-10)

            # Create heatmap
            im = ax.imshow(
                normalized.reshape(-1, 1) if normalized.ndim == 1 else normalized,
                cmap="YlOrRd",
                aspect="auto",
                interpolation="nearest",
            )

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Set title
            ax.set_title(agent_name, fontsize=self.title_fontsize)
            ax.set_xlabel("State", fontsize=self.label_fontsize)

            if idx == 0:
                ax.set_ylabel("Frequency", fontsize=self.label_fontsize)

        fig.suptitle(
            "Coverage Comparison Across Agents",
            fontsize=self.title_fontsize + 2,
            y=1.02,
        )
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved coverage comparison to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_generalization_comparison(
        self,
        agent_names: List[str],
        in_dist_success_rates: List[float],
        out_dist_success_rates: Dict[str, List[float]],
        shift_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Plot comparison of generalization across agents and distribution shifts.

        Creates a grouped bar chart showing in-distribution and multiple
        out-of-distribution success rates for each agent.

        Args:
            agent_names: List of agent names
            in_dist_success_rates: List of in-distribution success rates (one per agent)
            out_dist_success_rates: Dictionary mapping shift names to lists of success rates
            shift_names: List of distribution shift names
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)

        # Number of agents and shifts
        n_agents = len(agent_names)
        n_shifts = len(shift_names) + 1  # +1 for in-dist

        # Set up bar positions
        x_pos = np.arange(n_agents)
        width = 0.8 / n_shifts  # Total width divided by number of bars

        # Plot in-distribution bars
        offset = -0.4 + width / 2
        bars = ax.bar(
            x_pos + offset,
            in_dist_success_rates,
            width,
            label="In-Distribution",
            color=self.colors[0],
            alpha=0.9,
        )
        offset += width

        # Plot out-of-distribution bars for each shift
        for idx, shift_name in enumerate(shift_names):
            success_rates = out_dist_success_rates[shift_name]
            bars = ax.bar(
                x_pos + offset,
                success_rates,
                width,
                label=shift_name.replace("test_", "").replace("_", " ").title(),
                color=self.colors[idx + 1],
                alpha=0.9,
            )
            offset += width

        # Formatting
        ax.set_ylabel("Success Rate", fontsize=self.label_fontsize)
        ax.set_xlabel("Agent", fontsize=self.label_fontsize)
        ax.set_title(
            "Generalization Performance Across Distribution Shifts",
            fontsize=self.title_fontsize,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [name.upper() for name in agent_names],
            fontsize=self.tick_fontsize,
        )
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=self.legend_fontsize, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved generalization comparison to {save_path}")
        else:
            # Auto-save with default name
            default_path = Path(self.output_dir) / "generalization_comparison.png"
            fig.savefig(default_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved generalization comparison to {default_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_degradation_heatmap(
        self,
        agent_names: List[str],
        shift_names: List[str],
        degradation_matrix: List[List[float]],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """
        Plot heatmap of performance degradation across agents and shifts.

        Creates a heatmap showing the percentage drop in success rate
        for each agent on each distribution shift.

        Args:
            agent_names: List of agent names
            shift_names: List of distribution shift names
            degradation_matrix: 2D list of degradation percentages
                               (rows=agents, cols=shifts)
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # Convert to numpy array
        degradation_array = np.array(degradation_matrix)

        # Create heatmap
        im = ax.imshow(
            degradation_array,
            cmap="RdYlGn_r",  # Red for high degradation, green for low
            aspect="auto",
            vmin=-50,  # Cap at -50% degradation
            vmax=0,  # Best case is 0% degradation
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(shift_names)))
        ax.set_yticks(np.arange(len(agent_names)))
        ax.set_xticklabels(
            [
                name.replace("test_", "").replace("_", " ").title()
                for name in shift_names
            ],
            rotation=45,
            ha="right",
            fontsize=self.tick_fontsize,
        )
        ax.set_yticklabels(
            [name.upper() for name in agent_names],
            fontsize=self.tick_fontsize,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            "Success Rate Degradation (%)",
            rotation=270,
            labelpad=20,
            fontsize=self.label_fontsize,
        )

        # Add text annotations
        for i in range(len(agent_names)):
            for j in range(len(shift_names)):
                value = degradation_array[i, j]
                text_color = "white" if value < -25 else "black"
                text = ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        # Title and layout
        ax.set_title(
            "Performance Degradation Heatmap",
            fontsize=self.title_fontsize,
            pad=20,
        )
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved degradation heatmap to {save_path}")
        else:
            # Auto-save with default name
            default_path = Path(self.output_dir) / "degradation_heatmap.png"
            fig.savefig(default_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved degradation heatmap to {default_path}")

        # Show if requested
        if show:
            plt.show()

        return fig
