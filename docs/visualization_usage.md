# Visualization Module Usage Guide

This guide explains how to use the `PlotGenerator` class to create publication-quality visualizations for RL experiments.

## Overview

The `PlotGenerator` class provides a comprehensive set of plotting functions for:
- Learning curves with confidence bands
- Ablation study comparisons
- Algorithm performance comparisons
- Exploration strategy analysis
- Memory contribution visualization
- Generalization analysis
- State coverage heatmaps

All plots use deterministic styling for reproducibility and follow publication-quality standards.

## Installation

The visualization module requires the following dependencies:
```bash
pip install matplotlib seaborn numpy scipy
```

## Basic Usage

### Creating a PlotGenerator

```python
from src.visualization import PlotGenerator

# Create plot generator with default settings
plot_gen = PlotGenerator(
    style='seaborn-v0_8-darkgrid',  # Matplotlib style
    figsize=(10, 6),                 # Default figure size
    dpi=100,                         # Resolution
    random_seed=42,                  # For reproducibility
)
```

### 1. Learning Curves

Plot training progress over time with confidence bands:

```python
# From training metrics dictionaries
results = {
    'Q-Learning': [
        {'episode': 0, 'reward': -10.5, 'steps': 25, 'success': False},
        {'episode': 1, 'reward': -8.2, 'steps': 23, 'success': False},
        # ... more episodes
    ],
    'DQN': [...],
    'PPO': [...],
}

# Or from JSON files
results = {
    'Q-Learning': 'logs/q_learning/training_metrics.json',
    'DQN': 'logs/dqn/training_metrics.json',
    'PPO': 'logs/ppo/training_metrics.json',
}

# Plot single metric
fig = plot_gen.plot_learning_curves(
    results,
    metric='reward',
    xlabel='Episode',
    ylabel='Episode Reward',
    title='Training Performance',
    save_path='learning_curves.png',
    show=True,
    smoothing_window=10,  # Moving average window
)

# Plot multiple metrics in subplots
fig = plot_gen.plot_multiple_metrics(
    results,
    metrics=['reward', 'steps', 'success'],
    save_path='multi_metrics.png',
    show=True,
)
```

### 2. Ablation Study

Compare different model configurations:

```python
from src.evaluation.metrics import AggregateMetrics

# Create ablation results
ablation_results = {
    'Baseline': AggregateMetrics(...),
    'With Exploration': AggregateMetrics(...),
    'With Memory': AggregateMetrics(...),
    'Full Model': AggregateMetrics(...),
}

# Create table
fig = plot_gen.plot_ablation_table(
    ablation_results,
    metrics=['success_rate', 'mean_steps_to_success', 'mean_coverage'],
    save_path='ablation_table.png',
    show=True,
)
```

### 3. Algorithm Comparison

Compare final performance across algorithms:

```python
# Bar plot for single metric
fig = plot_gen.plot_algorithm_comparison(
    algorithm_results,
    metric='success_rate',
    ylabel='Success Rate',
    title='Algorithm Performance Comparison',
    save_path='algorithm_comparison.png',
    show=True,
)
```

### 4. Exploration Strategy Comparison

Compare different exploration strategies:

```python
exploration_results = {
    'Epsilon-Greedy': AggregateMetrics(...),
    'Count-Based': AggregateMetrics(...),
    'Curiosity': AggregateMetrics(...),
}

# Creates 2x2 subplot with multiple metrics
fig = plot_gen.plot_exploration_comparison(
    exploration_results,
    save_path='exploration_comparison.png',
    show=True,
)
```

### 5. Memory Contribution

Visualize the contribution of recurrent memory:

```python
recurrent_results = AggregateMetrics(...)
non_recurrent_results = AggregateMetrics(...)

fig = plot_gen.plot_memory_comparison(
    recurrent_results,
    non_recurrent_results,
    save_path='memory_comparison.png',
    show=True,
)
```

### 6. Generalization Analysis

Compare in-distribution vs out-of-distribution performance:

```python
in_dist_results = {
    'Q-Learning': AggregateMetrics(...),
    'DQN': AggregateMetrics(...),
    'PPO': AggregateMetrics(...),
}

out_dist_results = {
    'Q-Learning': AggregateMetrics(...),
    'DQN': AggregateMetrics(...),
    'PPO': AggregateMetrics(...),
}

# Grouped bar plot comparison
fig = plot_gen.plot_generalization(
    in_dist_results,
    out_dist_results,
    save_path='generalization.png',
    show=True,
)

# Performance degradation
fig = plot_gen.plot_performance_degradation(
    in_dist_results,
    out_dist_results,
    metric='success_rate',
    save_path='degradation.png',
    show=True,
)
```

### 7. Coverage Visualization

Visualize state visitation patterns:

```python
import numpy as np

# State visitation matrix (agents/episodes x states)
state_visitation = np.array([
    [10, 5, 20, 15, ...],  # Episode 1
    [12, 8, 18, 14, ...],  # Episode 2
    # ...
])

# Heatmap
fig = plot_gen.plot_coverage_heatmap(
    state_visitation,
    state_labels=['S0', 'S1', 'S2', ...],
    title='State Visitation Patterns',
    save_path='coverage_heatmap.png',
    show=True,
)

# Compare coverage across agents
coverage_data = {
    'Random': np.array([...]),
    'Q-Learning': np.array([...]),
    'DQN': np.array([...]),
}

fig = plot_gen.plot_coverage_comparison(
    coverage_data,
    save_path='coverage_comparison.png',
    show=True,
)
```

## Advanced Features

### Custom Styling

```python
# Use custom matplotlib style
plot_gen = PlotGenerator(
    style='ggplot',
    figsize=(12, 8),
    dpi=150,
)

# Access and modify style attributes
plot_gen.title_fontsize = 16
plot_gen.label_fontsize = 14
plot_gen.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
```

### Saving High-Resolution Figures

```python
# Save with high DPI for publications
plot_gen = PlotGenerator(dpi=300)

fig = plot_gen.plot_learning_curves(
    results,
    metric='reward',
    save_path='figure_high_res.png',
    show=False,
)

# Or save manually with custom format
fig.savefig('figure.pdf', dpi=300, bbox_inches='tight', format='pdf')
```

### Deterministic Plotting

All plots are deterministic when using the same random seed:

```python
# Same seed produces identical plots
plot_gen1 = PlotGenerator(random_seed=42)
plot_gen2 = PlotGenerator(random_seed=42)

# These will be identical
fig1 = plot_gen1.plot_learning_curves(results, ...)
fig2 = plot_gen2.plot_learning_curves(results, ...)
```

## Integration with Training Pipeline

### Example: Complete Workflow

```python
from src.training.orchestrator import TrainingOrchestrator
from src.evaluation.evaluator import Evaluator
from src.visualization import PlotGenerator

# 1. Train agents
orchestrator = TrainingOrchestrator(agent, env, config)
training_results = orchestrator.train()

# 2. Evaluate agents
evaluator = Evaluator()
eval_results = evaluator.evaluate(agent, test_envs, num_episodes=100)

# 3. Create visualizations
plot_gen = PlotGenerator(random_seed=42)

# Learning curves
fig = plot_gen.plot_learning_curves(
    {'My Agent': training_results['training_metrics']},
    metric='reward',
    save_path='results/learning_curve.png',
)

# Final performance
fig = plot_gen.plot_algorithm_comparison(
    {'My Agent': eval_results.metrics},
    metric='success_rate',
    save_path='results/final_performance.png',
)
```

## Tips and Best Practices

1. **Smoothing**: Use `smoothing_window` parameter for noisy learning curves
2. **Confidence Bands**: Ensure you have multiple seeds for meaningful confidence intervals
3. **File Formats**: Use PNG for presentations, PDF for publications
4. **Resolution**: Use DPI 100-150 for screen, 300+ for print
5. **Colors**: The default color palette is colorblind-friendly
6. **Reproducibility**: Always set `random_seed` for consistent results

## Troubleshooting

### Issue: Plots look different on different machines
**Solution**: Ensure same matplotlib version and use explicit style:
```python
plot_gen = PlotGenerator(style='default')
```

### Issue: Confidence bands are zero
**Solution**: Ensure you're providing data from multiple seeds or runs

### Issue: Labels overlap
**Solution**: Adjust figure size or rotate labels:
```python
plot_gen = PlotGenerator(figsize=(14, 6))
```

### Issue: Out of memory with large datasets
**Solution**: Use smoothing or downsample data before plotting:
```python
# Downsample every 10th point
metrics_downsampled = metrics[::10]
```

## Examples

See the following files for complete examples:
- `test_visualization.py` - Comprehensive test suite
- `example_visualization.py` - Real-world usage examples

## API Reference

### PlotGenerator Class

#### Constructor
```python
PlotGenerator(
    style='seaborn-v0_8-darkgrid',
    figsize=(10, 6),
    dpi=100,
    random_seed=42,
)
```

#### Methods

- `plot_learning_curves()` - Learning curves with confidence bands
- `plot_multiple_metrics()` - Multiple metrics in subplots
- `plot_ablation_table()` - Ablation study comparison table
- `plot_algorithm_comparison()` - Bar plot comparing algorithms
- `plot_exploration_comparison()` - Exploration strategy comparison
- `plot_memory_comparison()` - Recurrent vs non-recurrent comparison
- `plot_generalization()` - In-dist vs out-of-dist comparison
- `plot_performance_degradation()` - Performance degradation bar plot
- `plot_coverage_heatmap()` - State visitation heatmap
- `plot_coverage_comparison()` - Coverage comparison across agents

All methods return a `matplotlib.figure.Figure` object and accept:
- `save_path` (optional): Path to save the figure
- `show` (optional): Whether to display the plot (default: True)

## Requirements Satisfied

This implementation satisfies the following requirements:

- Learning curves with confidence bands over multiple seeds
- Deterministic plotting with fixed random seeds
- Ablation comparisons isolating exploration and memory contributions
- Both curves and tables for temporal dynamics and final comparisons
- Generalization performance visualization
- Coverage metrics and visualizations
