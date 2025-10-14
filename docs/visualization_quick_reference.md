# PlotGenerator Quick Reference

## Import
```python
from src.visualization import PlotGenerator
plot_gen = PlotGenerator(random_seed=42)
```

## Learning Curves
```python
# Single metric
plot_gen.plot_learning_curves(
    {'Agent': metrics_list},
    metric='reward',
    save_path='curve.png'
)

# Multiple metrics
plot_gen.plot_multiple_metrics(
    {'Agent': metrics_list},
    metrics=['reward', 'steps'],
    save_path='multi.png'
)
```

## Comparisons
```python
# Algorithm comparison
plot_gen.plot_algorithm_comparison(
    {'Q-Learning': agg_metrics, 'DQN': agg_metrics},
    metric='success_rate',
    save_path='algo.png'
)

# Ablation table
plot_gen.plot_ablation_table(
    {'Config1': agg_metrics, 'Config2': agg_metrics},
    save_path='ablation.png'
)

# Exploration strategies
plot_gen.plot_exploration_comparison(
    {'Epsilon': agg_metrics, 'Curiosity': agg_metrics},
    save_path='exploration.png'
)

# Memory contribution
plot_gen.plot_memory_comparison(
    recurrent_metrics,
    non_recurrent_metrics,
    save_path='memory.png'
)
```

## Generalization
```python
# In-dist vs out-of-dist
plot_gen.plot_generalization(
    in_dist_dict,
    out_dist_dict,
    save_path='gen.png'
)

# Performance degradation
plot_gen.plot_performance_degradation(
    in_dist_dict,
    out_dist_dict,
    metric='success_rate',
    save_path='degrade.png'
)
```

## Coverage
```python
# Heatmap
plot_gen.plot_coverage_heatmap(
    state_visitation_array,
    save_path='heatmap.png'
)

# Comparison
plot_gen.plot_coverage_comparison(
    {'Agent1': coverage_array, 'Agent2': coverage_array},
    save_path='coverage.png'
)
```

## Common Parameters
- `save_path`: Where to save (PNG, PDF, etc.)
- `show`: Display plot (default: True)
- `smoothing_window`: Moving average window
- `title`, `xlabel`, `ylabel`: Custom labels

## Input Formats
- Training metrics: List of dicts or JSON file path
- Aggregate metrics: AggregateMetrics objects
- Coverage: NumPy arrays

## Tips
- Use `random_seed=42` for reproducibility
- Set `dpi=300` for publications
- Use `smoothing_window=10` for noisy curves
- Set `show=False` for batch processing
