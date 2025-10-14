# RL Flow Explorer - Experiment Infrastructure

Complete experiment infrastructure for training, evaluating, and visualizing RL agents on flow environments.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [Configuration Files](#configuration-files)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Examples](#examples)

## 🎯 Overview

This infrastructure provides:

✅ **Training**: Train RL agents with configurable hyperparameters  
✅ **Evaluation**: Evaluate agents on in-distribution and OOD test sets  
✅ **Visualization**: Generate publication-quality plots  
✅ **Multi-Seed Experiments**: Run experiments across multiple seeds with parallelization  
✅ **Reproducibility**: Full reproducibility guarantees with deterministic operations  
✅ **Configuration Management**: JSON-based configuration files  

## 🚀 Quick Start

### 1. Train an Agent

```bash
python train.py --agent dqn --config configs/dqn_baseline.json
```

### 2. Evaluate the Agent

```bash
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100
```

### 3. Generate Plots

```bash
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_baseline \
  --agent-names DQN \
  --metric reward
```

### 4. Run Multi-Seed Experiment

```bash
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_multiseed \
  --num-seeds 5 \
  --parallel
```

## 📜 Scripts

### `train.py` - Training Script

Train RL agents on flow environments.

**Key Features**:
- Support for Q-learning, DQN, PPO, Recurrent PPO
- Configurable via CLI or JSON config files
- Automatic validation environment creation
- TensorBoard integration
- Checkpoint saving (best, periodic, final)

**Example**:
```bash
python train.py \
  --agent ppo \
  --num-episodes 1000 \
  --learning-rate 0.0003 \
  --seed 42
```

### `evaluate.py` - Evaluation Script

Evaluate trained agents on test environments.

**Key Features**:
- In-distribution and OOD evaluation
- Multiple distribution shift types
- Comprehensive metrics collection
- Results saved to JSON

**Example**:
```bash
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn/checkpoints/best_model.pt \
  --eval-ood \
  --ood-shift deeper
```

### `plot_results.py` - Visualization Script

Generate publication-quality plots.

**Key Features**:
- Multiple plot types
- High-resolution output (PNG, PDF, SVG)
- Deterministic plotting
- Confidence bands

**Example**:
```bash
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results results/dqn.json results/ppo.json \
  --eval-names DQN PPO \
  --metric success_rate
```

### `run_experiments.py` - Multi-Seed Runner

Run experiments across multiple seeds with optional parallelization.

**Key Features**:
- Sequential or parallel execution
- Automatic result aggregation
- Summary statistics
- Failed experiment tracking

**Example**:
```bash
python run_experiments.py \
  --agent dqn \
  --experiment-name my_experiment \
  --num-seeds 5 \
  --parallel \
  --max-workers 4
```

## ⚙️ Configuration Files

Pre-defined configurations in `configs/`:

| File | Description |
|------|-------------|
| `dqn_baseline.json` | Standard DQN configuration |
| `ppo_baseline.json` | Standard PPO configuration |
| `recurrent_ppo_baseline.json` | Recurrent PPO with LSTM |
| `qlearning_small.json` | Q-learning on small environments |
| `exploration_comparison.json` | Exploration strategy ablation |

### Configuration Format

```json
{
  "experiment_name": "my_experiment",
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

## 🔒 Reproducibility

Full reproducibility guaranteed through:

### 1. Random Seed Management
- Seeds set for Python, NumPy, PyTorch
- Deterministic CUDA operations
- Environment variables for determinism

### 2. Configuration Tracking
- All hyperparameters saved to JSON
- System information logged
- Timestamps on all outputs

### 3. Reproducibility Module

```python
from src.utils.reproducibility import set_global_seeds, ReproducibilityContext

# Set seeds globally
set_global_seeds(42, deterministic=True)

# Or use context manager
with ReproducibilityContext(seed=42):
    # Your code here
    result = train_model()
```

## 📚 Documentation

- **[Quick Start Guide](EXPERIMENT_QUICK_START.md)** - Quick reference for common commands
- **[Complete Usage Guide](docs/experiment_scripts_usage.md)** - Detailed documentation

### Getting Help

```bash
python train.py --help
python evaluate.py --help
python plot_results.py --help
python run_experiments.py --help
```

## 💡 Examples

### Example 1: Train and Evaluate

```bash
# Train
python train.py --agent dqn --config configs/dqn_baseline.json

# Evaluate
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100

# Plot
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_baseline \
  --agent-names DQN \
  --metric reward
```

### Example 2: Compare Algorithms

```bash
# Train multiple algorithms
for agent in dqn ppo recurrent_ppo; do
  python train.py --agent $agent --config configs/${agent}_baseline.json
done

# Evaluate all
for agent in dqn ppo recurrent_ppo; do
  python evaluate.py \
    --agent $agent \
    --checkpoint logs/${agent}_baseline/checkpoints/best_model.pt \
    --num-episodes 100 \
    --output-dir evaluation_results/$agent
done

# Compare
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results evaluation_results/*/in_distribution_results.json \
  --eval-names DQN PPO "Recurrent PPO" \
  --metric success_rate
```

### Example 3: Multi-Seed with Statistics

```bash
# Run across 5 seeds in parallel
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_multiseed \
  --num-seeds 5 \
  --parallel

# Results automatically aggregated with statistics
cat experiment_results/dqn_multiseed_aggregated.json
```

### Example 4: Generalization Analysis

```bash
# Train
python train.py --agent dqn --config configs/dqn_baseline.json

# Evaluate in-distribution
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100 \
  --output-dir evaluation_results/dqn

# Evaluate OOD
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100 \
  --eval-ood \
  --ood-shift deeper \
  --output-dir evaluation_results/dqn

# Plot generalization
python plot_results.py \
  --plot-type generalization \
  --in-dist-results evaluation_results/dqn/in_distribution_results.json \
  --ood-results evaluation_results/dqn/ood_deeper_results.json \
  --eval-names DQN
```

### Example Workflow Script

Run the complete example workflow:
```bash
bash example_experiment_workflow.sh
```

## 📊 Monitoring

Monitor training in real-time with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## 📁 Output Structure

```
.
├── logs/                           # Training logs
│   └── <experiment_name>/
│       ├── checkpoints/            # Model checkpoints
│       ├── tensorboard/            # TensorBoard logs
│       ├── training_metrics.json   # Training metrics
│       ├── experiment_config.json  # Experiment configuration
│       └── hyperparameters.json    # Hyperparameters + system info
│
├── evaluation_results/             # Evaluation results
│   └── <experiment_name>/
│       ├── in_distribution_results.json
│       ├── ood_deeper_results.json
│       └── evaluation_config.json
│
├── experiment_results/             # Multi-seed aggregated results
│   └── <experiment_name>_aggregated.json
│
└── plots/                          # Generated plots
    ├── learning_curves.png
    ├── algorithm_comparison.png
    └── generalization.png
```

## 🎓 Best Practices

1. **Use Configuration Files**: Define experiments in JSON for reproducibility
2. **Run Multiple Seeds**: Always use at least 5 seeds for statistical significance
3. **Enable Parallelization**: Use `--parallel` for faster multi-seed experiments
4. **Monitor Training**: Use TensorBoard to track progress
5. **Evaluate OOD**: Always test generalization capabilities
6. **Document Experiments**: Use descriptive experiment names

## 🐛 Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--buffer-size` (for DQN)
- Use smaller networks

### Slow Training
- Enable `--parallel` for multi-seed experiments
- Reduce `--num-episodes`
- Use smaller environments

### Reproducibility Issues
- Ensure same PyTorch/CUDA versions
- Use `--seed` consistently
- Check deterministic mode is enabled

## 📝 Requirements

See `requirements.txt` for dependencies:
- PyTorch
- Gymnasium
- NetworkX
- NumPy
- Matplotlib
- Seaborn
- TensorBoard

## 🤝 Contributing

When adding new features:
1. Update configuration files if needed
2. Add command-line arguments with defaults
3. Update documentation
4. Ensure reproducibility is maintained

## 📄 License

See LICENSE file for details.

---

**Need Help?** Check the [Complete Usage Guide](docs/experiment_scripts_usage.md) or run any script with `--help`.
