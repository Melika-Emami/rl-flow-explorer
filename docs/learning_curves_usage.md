# Learning Curves Generation - Usage Guide

## Overview

This guide provides comprehensive documentation for generating learning curves from RL experiment results. Learning curves visualize how agents improve over training episodes and are essential for comparing different algorithms and configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Script Usage](#script-usage)
3. [Output Files](#output-files)
4. [Plot Features](#plot-features)
5. [Advanced Usage](#advanced-usage)
6. [Integration](#integration)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Generate All Learning Curves

```bash
python generate_learning_curves.py
```

This command:
- Searches for experiment results in `baseline_results/`
- Generates individual learning curves for each agent
- Creates comparison plots across all algorithms
- Saves high-resolution plots (300 DPI) to `learning_curve_plots/`

### Prerequisites

You must have training results available. If not, run baseline experiments first:

```bash
python run_baseline_experiments.py --num-seeds 5
```

## Script Usage

### Command-Line Interface

```bash
python generate_learning_curves.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--results-dir` | str | `baseline_results` | Directory containing experiment results |
| `--output-dir` | str | `learning_curve_plots` | Directory to save plots |
| `--smoothing-window` | int | `10` | Window size for moving average smoothing |
| `--dpi` | int | `300` | DPI for output images (high-resolution) |
| `--seed` | int | `42` | Random seed for reproducible plotting |
| `--agents` | list | `None` | Specific agents to plot (default: all found) |

### Examples

#### Custom Output Directory

```bash
python generate_learning_curves.py --output-dir my_plots
```

#### Higher Smoothing

```bash
python generate_learning_curves.py --smoothing-window 20
```

#### Specific Agents Only

```bash
python generate_learning_curves.py --agents dqn ppo
```

#### Ultra High Resolution

```bash
python generate_learning_curves.py --dpi 600
```

#### Complete Custom Configuration

```bash
python generate_learning_curves.py \
    --results-dir baseline_results \
    --output-dir publication_plots \
    --smoothing-window 15 \
    --dpi 300 \
    --seed 42 \
    --agents dqn ppo recurrent_ppo
```

## Output Files

### Directory Structure

```
learning_curve_plots/
├── dqn_reward_curve.png
├── dqn_success_curve.png
├── dqn_steps_curve.png
├── dqn_coverage_curve.png
├── dqn_all_metrics.png
├── ppo_reward_curve.png
├── ppo_success_curve.png
├── ppo_steps_curve.png
├── ppo_coverage_curve.png
├── ppo_all_metrics.png
├── recurrent_ppo_reward_curve.png
├── recurrent_ppo_success_curve.png
├── recurrent_ppo_steps_curve.png
├── recurrent_ppo_coverage_curve.png
├── recurrent_ppo_all_metrics.png
├── algorithm_comparison_reward.png
├── algorithm_comparison_success.png
├── algorithm_comparison_steps.png
├── algorithm_comparison_coverage.png
└── algorithm_comparison_all_metrics.png
```

### File Naming Convention

#### Individual Agent Plots
- `{agent}_{metric}_curve.png` - Single metric learning curve
- `{agent}_all_metrics.png` - All metrics in one figure

#### Comparison Plots
- `algorithm_comparison_{metric}.png` - Compare all agents on one metric
- `algorithm_comparison_all_metrics.png` - All metrics comparison

### Metrics Plotted

1. **Reward** - Cumulative episode reward
2. **Success** - Success rate (0-1)
3. **Steps** - Number of steps to success
4. **Coverage** - State space coverage (0-1)

## Plot Features

### Confidence Bands

All plots include confidence bands showing:
- **Solid line**: Mean value across all seeds
- **Shaded area**: Mean ± standard deviation

This visualization shows both the average performance and the variability across different random seeds.

### Smoothing

Learning curves can be noisy due to:
- Stochastic environments
- Exploration randomness
- Training instabilities

The script applies moving average smoothing to reduce noise while preserving trends. The smoothing window is configurable (default: 10 episodes).

### High Resolution

Plots are generated at 300 DPI by default, ensuring:
- Publication-quality figures
- Clear text and labels
- Smooth lines and curves
- Professional appearance

### Consistent Styling

All plots use:
- Deterministic color palette
- Consistent font sizes
- Grid lines for readability
- Clear axis labels and titles
- Legends for multiple agents

## Advanced Usage

### Using PlotGenerator Directly

For more control, use the `PlotGenerator` class directly:

```python
from src.visualization.plot_generator import PlotGenerator

# Create plotter
plotter = PlotGenerator(
    output_dir="my_plots",
    dpi=300,
    random_seed=42
)

# Load training results
results = {
    "DQN": "baseline_results/dqn_seed0/training_metrics.json",
    "PPO": "baseline_results/ppo_seed0/training_metrics.json",
}

# Generate single metric learning curve
plotter.plot_learning_curves(
    results=results,
    metric="reward",
    title="Algorithm Comparison: Reward",
    save_path="my_plots/reward_comparison.png",
    show=False,
    smoothing_window=10
)

# Generate multi-metric plot
plotter.plot_multiple_metrics(
    results=results,
    metrics=["reward", "success", "steps", "coverage"],
    save_path="my_plots/all_metrics.png",
    show=False,
    smoothing_window=10
)
```

### Custom Metrics

To plot custom metrics, modify the `key_metrics` list in the script:

```python
key_metrics = ["reward", "success", "steps", "coverage", "my_custom_metric"]
```

Ensure your training metrics JSON includes the custom metric.

### Different Result Directories

Generate curves for different experiment types:

```bash
# Exploration ablations
python generate_learning_curves.py \
    --results-dir exploration_results \
    --output-dir exploration_curves

# Memory ablations
python generate_learning_curves.py \
    --results-dir memory_results \
    --output-dir memory_curves

# Generalization experiments
python generate_learning_curves.py \
    --results-dir generalization_results \
    --output-dir generalization_curves
```

## Integration

### With Baseline Experiments

```bash
# Run experiments
python run_baseline_experiments.py --num-seeds 5 --parallel

# Generate curves
python generate_learning_curves.py
```

### With Exploration Ablations

```bash
# Run ablations
python run_exploration_ablations.py --num-seeds 5

# Generate curves
python generate_learning_curves.py \
    --results-dir exploration_results \
    --output-dir exploration_curves
```

### With Memory Ablations

```bash
# Run ablations
python run_memory_ablations.py --num-seeds 5

# Generate curves
python generate_learning_curves.py \
    --results-dir memory_results \
    --output-dir memory_curves
```

### With plot_results.py

The `plot_results.py` script can also generate learning curves with more manual configuration:

```bash
python plot_results.py \
    --plot-type learning_curves \
    --results-dirs baseline_results/dqn_seed0 baseline_results/ppo_seed0 \
    --agent-names "DQN" "PPO" \
    --metric reward \
    --output-dir plots \
    --smoothing-window 10
```

## Troubleshooting

### No Results Found

**Error:**
```
⚠️  No experiment results found in baseline_results
```

**Solution:**
Run baseline experiments first:
```bash
python run_baseline_experiments.py --num-seeds 5
```

### Missing Metrics

**Issue:** Some metrics are missing from plots

**Solution:** Ensure your training logs include all required metrics:
- reward
- success
- steps
- coverage

Check the `training_metrics.json` file format.

### Memory Issues

**Issue:** Out of memory when loading many seeds

**Solutions:**
1. Process agents one at a time:
   ```bash
   python generate_learning_curves.py --agents dqn
   python generate_learning_curves.py --agents ppo
   ```

2. Increase smoothing window to reduce data points:
   ```bash
   python generate_learning_curves.py --smoothing-window 50
   ```

3. Reduce DPI for testing:
   ```bash
   python generate_learning_curves.py --dpi 150
   ```

### Empty Plots

**Issue:** Plots are generated but appear empty

**Solution:** Check that:
1. Training metrics JSON files exist
2. JSON files contain "metrics" array
3. Metrics array has episode data
4. Metric names match expected names

### Warnings from PlotGenerator

**Warning:**
```
Warning: No data for agent_name
```

**Note:** This warning is expected when using list format for metrics. Plots are still generated correctly. The warning comes from the PlotGenerator trying different data formats.

## Data Format

### Expected JSON Structure

```json
{
  "config": {
    "agent": "dqn",
    "seed": 0,
    "num_episodes": 100
  },
  "metrics": [
    {
      "episode": 0,
      "reward": -10.5,
      "success": 0.0,
      "steps": 20,
      "coverage": 0.1
    },
    {
      "episode": 1,
      "reward": -8.2,
      "success": 0.0,
      "steps": 18,
      "coverage": 0.15
    }
  ]
}
```

### Required Fields

Each metric entry must include:
- `episode`: Episode number (int)
- `reward`: Episode reward (float)
- `success`: Success indicator (float, 0-1)
- `steps`: Number of steps (int)
- `coverage`: Coverage metric (float, 0-1)

## Best Practices

1. **Use Multiple Seeds**: Run at least 5 seeds for reliable confidence bands

2. **Consistent Configuration**: Use the same hyperparameters across seeds

3. **Appropriate Smoothing**: 
   - Small window (5-10) for detailed curves
   - Large window (20-50) for trend visualization

4. **High DPI for Publication**: Use 300+ DPI for papers and presentations

5. **Descriptive Names**: Use clear agent names in results directories

6. **Regular Backups**: Save plots and results regularly

7. **Version Control**: Track plot generation scripts and configurations

## Related Documentation

- [LEARNING_CURVES_QUICK_START.md](../LEARNING_CURVES_QUICK_START.md) - Quick start guide
- [TASK_14.1_SUMMARY.md](../TASK_14.1_SUMMARY.md) - Implementation summary
- [visualization_usage.md](visualization_usage.md) - General visualization guide
- [experiment_scripts_usage.md](experiment_scripts_usage.md) - Experiment scripts guide

## Examples

See [example_learning_curves.py](../example_learning_curves.py) for complete examples of:
- Basic usage
- Custom configuration
- Direct PlotGenerator usage
- Integration with experiments
- Different experiment types
