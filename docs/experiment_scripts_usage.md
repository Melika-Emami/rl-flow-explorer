# Experiment Scripts Usage Guide

This guide explains how to use the experiment scripts for training, evaluating, and visualizing RL agents.

## Overview

The experiment infrastructure consists of three main scripts:

1. **`train.py`** - Train a single agent
2. **`evaluate.py`** - Evaluate a trained agent
3. **`plot_results.py`** - Generate visualizations
4. **`run_experiments.py`** - Run experiments across multiple seeds

## Training a Single Agent

### Basic Usage

```bash
python train.py --agent dqn --seed 42
```

### Using a Configuration File

```bash
python train.py --config configs/dqn_baseline.json
```

### Custom Hyperparameters

```bash
python train.py \
  --agent ppo \
  --num-episodes 2000 \
  --learning-rate 0.0003 \
  --num-nodes 15 \
  --max-depth 7 \
  --seed 42 \
  --experiment-name ppo_large_env
```

### Available Arguments

- `--agent`: Agent type (`qlearning`, `dqn`, `ppo`, `recurrent_ppo`)
- `--num-episodes`: Number of training episodes (default: 1000)
- `--learning-rate`: Learning rate (default: 0.001)
- `--seed`: Random seed (default: 42)
- `--log-dir`: Directory for logs (default: `logs`)
- `--experiment-name`: Name for the experiment

Environment parameters:
- `--num-nodes`: Number of nodes in graph (default: 10)
- `--branching-factor`: Graph branching factor (default: 2.0)
- `--max-depth`: Maximum graph depth (default: 5)
- `--popup-probability`: Popup failure probability (default: 0.1)

See `python train.py --help` for all options.

## Evaluating a Trained Agent

### Basic Evaluation

```bash
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_seed42/checkpoints/best_model.pt \
  --num-episodes 100
```

### Out-of-Distribution Evaluation

```bash
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_seed42/checkpoints/best_model.pt \
  --eval-ood \
  --ood-shift deeper \
  --num-episodes 100
```

Available OOD shifts:
- `deeper`: Larger graphs with more nodes
- `higher_popup`: Higher popup failure probability
- `different_topology`: Different graph structure

### Available Arguments

- `--agent`: Agent type
- `--checkpoint`: Path to checkpoint file
- `--num-episodes`: Number of evaluation episodes (default: 100)
- `--num-test-envs`: Number of test environments (default: 5)
- `--eval-ood`: Enable OOD evaluation
- `--ood-shift`: Type of distribution shift
- `--output-dir`: Directory for results (default: `evaluation_results`)
- `--seed`: Random seed (default: 42)

## Generating Plots

### Learning Curves

```bash
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_seed42 logs/ppo_seed42 \
  --agent-names DQN PPO \
  --metric reward \
  --output-dir plots
```

### Multiple Metrics

```bash
python plot_results.py \
  --plot-type multiple_metrics \
  --results-dirs logs/dqn_seed42 \
  --agent-names DQN \
  --metrics reward steps success \
  --output-dir plots
```

### Algorithm Comparison

```bash
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results evaluation_results/dqn/in_distribution_results.json \
                 evaluation_results/ppo/in_distribution_results.json \
  --eval-names DQN PPO \
  --metric success_rate \
  --output-dir plots
```

### Generalization Analysis

```bash
python plot_results.py \
  --plot-type generalization \
  --in-dist-results evaluation_results/dqn/in_distribution_results.json \
                     evaluation_results/ppo/in_distribution_results.json \
  --ood-results evaluation_results/dqn/ood_deeper_results.json \
                evaluation_results/ppo/ood_deeper_results.json \
  --eval-names DQN PPO \
  --output-dir plots
```

### Available Plot Types

- `learning_curves`: Learning curves with confidence bands
- `multiple_metrics`: Multiple metrics in subplots
- `ablation`: Ablation study table
- `algorithm_comparison`: Bar chart comparing algorithms
- `exploration_comparison`: Compare exploration strategies
- `memory_comparison`: Compare recurrent vs non-recurrent
- `generalization`: In-dist vs OOD performance
- `degradation`: Performance degradation analysis
- `all`: Generate all available plots

## Running Multiple Seeds

### Basic Multi-Seed Experiment

```bash
python run_experiments.py \
  --agent dqn \
  --experiment-name dqn_baseline \
  --num-seeds 5 \
  --base-seed 42
```

### Using Configuration File

```bash
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_baseline \
  --seeds 42 43 44 45 46
```

### Parallel Execution

```bash
python run_experiments.py \
  --agent ppo \
  --experiment-name ppo_baseline \
  --num-seeds 10 \
  --parallel \
  --max-workers 4
```

### Available Arguments

- `--agent`: Agent type
- `--config`: Configuration file path
- `--experiment-name`: Name for the experiment
- `--seeds`: Specific seeds to use (e.g., `--seeds 42 43 44`)
- `--num-seeds`: Number of seeds to generate (default: 5)
- `--base-seed`: Base seed for generation (default: 42)
- `--parallel`: Run experiments in parallel
- `--max-workers`: Number of parallel workers (default: 4)
- `--log-dir`: Base directory for logs (default: `logs`)
- `--output-dir`: Directory for aggregated results (default: `experiment_results`)

## Configuration Files

Configuration files allow you to define standard experiment setups. They are JSON files with the following structure:

```json
{
  "experiment_name": "dqn_baseline",
  "agent": "dqn",
  "seed": 42,
  
  "environment": {
    "num_nodes": 10,
    "branching_factor": 2.0,
    "max_depth": 5,
    "popup_probability": 0.1,
    "num_dependencies": 2,
    "max_steps": 100
  },
  
  "training": {
    "num_episodes": 1000,
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "batch_size": 32
  },
  
  "agent_config": {
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "buffer_size": 10000,
    "hidden_dims": [128, 128]
  }
}
```

### Available Configurations

Pre-defined configurations are available in the `configs/` directory:

- `dqn_baseline.json` - Standard DQN configuration
- `ppo_baseline.json` - Standard PPO configuration
- `recurrent_ppo_baseline.json` - Recurrent PPO configuration
- `qlearning_small.json` - Q-learning on small environments
- `exploration_comparison.json` - Configuration for exploration ablation

## Reproducibility

All scripts ensure reproducibility through:

1. **Fixed Random Seeds**: Set seeds for Python, NumPy, and PyTorch
2. **Deterministic Operations**: Use deterministic algorithms
3. **Configuration Saving**: All hyperparameters saved with results
4. **System Information**: Hardware and library versions logged

### Reproducibility Features

- Random seeds set globally for all libraries
- Deterministic CUDA operations (when available)
- All configurations saved to JSON
- System information logged with results

## Example Workflows

### Complete Training and Evaluation Pipeline

```bash
# 1. Train agent across multiple seeds
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_baseline \
  --num-seeds 5 \
  --parallel

# 2. Evaluate best model from each seed
for seed in 42 142 242 342 442; do
  python evaluate.py \
    --agent dqn \
    --checkpoint logs/dqn_baseline_seed${seed}/checkpoints/best_model.pt \
    --num-episodes 100 \
    --eval-ood \
    --ood-shift deeper \
    --output-dir evaluation_results/dqn_seed${seed}
done

# 3. Generate plots
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_baseline_seed42 logs/dqn_baseline_seed142 \
  --agent-names "Seed 42" "Seed 142" \
  --metric reward \
  --output-dir plots
```

### Algorithm Comparison

```bash
# Train multiple algorithms
for agent in dqn ppo recurrent_ppo; do
  python run_experiments.py \
    --agent $agent \
    --experiment-name ${agent}_comparison \
    --num-seeds 5 \
    --parallel
done

# Evaluate all
for agent in dqn ppo recurrent_ppo; do
  python evaluate.py \
    --agent $agent \
    --checkpoint logs/${agent}_comparison_seed42/checkpoints/best_model.pt \
    --num-episodes 100 \
    --output-dir evaluation_results/${agent}
done

# Generate comparison plots
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results evaluation_results/dqn/in_distribution_results.json \
                 evaluation_results/ppo/in_distribution_results.json \
                 evaluation_results/recurrent_ppo/in_distribution_results.json \
  --eval-names DQN PPO "Recurrent PPO" \
  --metric success_rate \
  --output-dir plots
```

## Tips and Best Practices

1. **Use Configuration Files**: Define standard experiments in config files for reproducibility
2. **Multiple Seeds**: Always run experiments with at least 5 seeds for statistical significance
3. **Parallel Execution**: Use `--parallel` for faster multi-seed experiments
4. **Save Checkpoints**: Use `--checkpoint-frequency` to save intermediate models
5. **Monitor Training**: Use TensorBoard to monitor training in real-time:
   ```bash
   tensorboard --logdir logs/
   ```
6. **Evaluate OOD**: Always evaluate on out-of-distribution test sets to assess generalization
7. **Document Experiments**: Use descriptive `--experiment-name` values

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Reduce `--buffer-size` (for DQN)
- Use smaller networks with `--hidden-dims`

### Slow Training

- Enable `--parallel` for multi-seed experiments
- Reduce `--num-episodes`
- Use smaller environments with fewer `--num-nodes`

### Reproducibility Issues

- Ensure same PyTorch and CUDA versions
- Use `--seed` consistently
- Check that deterministic mode is enabled

### Plot Generation Errors

- Verify result files exist at specified paths
- Check that JSON files are not corrupted
- Ensure metric names match those in results files
