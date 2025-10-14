# Usage Quick Start Guide

This guide provides quick examples for common tasks in the RL Flow Explorer system.

## Table of Contents

- [Installation](#installation)
- [Basic Training](#basic-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Running Experiments](#running-experiments)
- [Next Steps](#next-steps)

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd rl-flow-explorer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print('Gymnasium: OK')"
```

---

## Basic Training

### Train a Q-Learning Agent

```bash
# Quick training run
python example_training.py
```

Or programmatically:

```python
from src.environment.graph_generator import GraphGenerator
from src.environment.flow_environment import FlowEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import GraphConfig, EnvConfig, TrainingConfig

# Create environment
generator = GraphGenerator()
graph = generator.generate(GraphConfig(num_nodes=10, branching_factor=2.0))
env = FlowEnvironment(graph, EnvConfig(max_steps_per_episode=50))

# Create agent
agent = QLearningAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.1
)

# Train
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=TrainingConfig(num_episodes=500),
    log_dir="logs/quickstart"
)
results = orchestrator.train()

print(f"Training complete! Best reward: {results['best_val_reward']:.2f}")
```

### Train a DQN Agent

```bash
# Using the main training script
python train.py \
    --algorithm dqn \
    --num-episodes 2000 \
    --output-dir logs/dqn_test
```

### Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir logs/quickstart/tensorboard

# Open browser to http://localhost:6006
```

---

## Evaluation

### Evaluate a Trained Agent

```python
from src.evaluation.evaluator import Evaluator

# Load trained agent
agent.load("logs/quickstart/checkpoints/best_model.pt")

# Create evaluator
evaluator = Evaluator(verbose=True)

# Evaluate
results = evaluator.evaluate_single_env(
    agent=agent,
    env=test_env,
    num_episodes=100,
    agent_name="Q-Learning"
)

print(f"Success Rate: {results.metrics.success_rate:.2%}")
print(f"Mean Steps: {results.metrics.mean_steps_to_success:.2f}")
print(f"Coverage: {results.metrics.mean_coverage:.2%}")
```

### Using the Evaluation Script

```bash
# Evaluate saved checkpoint
python evaluate.py \
    --checkpoint logs/dqn_test/checkpoints/best_model.pt \
    --algorithm dqn \
    --num-episodes 100
```

---

## Visualization

### Generate Learning Curves

```bash
# From baseline experiments
python generate_learning_curves.py \
    --results-dir baseline_results \
    --output-dir learning_curves
```

### Create Comparison Plots

```python
from src.visualization.plot_generator import PlotGenerator

plotter = PlotGenerator(output_dir="plots", dpi=300)

# Compare algorithms
results = {
    "Q-Learning": "logs/qlearning/training_metrics.json",
    "DQN": "logs/dqn/training_metrics.json",
    "PPO": "logs/ppo/training_metrics.json"
}

plotter.plot_learning_curves(
    results=results,
    metric="reward",
    title="Algorithm Comparison",
    save_path="plots/comparison.png",
    show=True
)
```

### View Example Visualizations

```bash
# Run visualization examples
python example_visualization.py

# Run ablation plot examples
python example_ablation_plots.py

# Run learning curve examples
python example_learning_curves.py
```

---

## Running Experiments

### Baseline Experiments

```bash
# Run all baseline algorithms with 5 seeds
python run_baseline_experiments.py --num-seeds 5 --parallel

# Results saved to baseline_results/
# Logs saved to baseline_logs/
```

### Exploration Ablations

```bash
# Compare exploration strategies
python run_exploration_ablations.py --config configs/exploration_comparison.json


# Results saved to exploration_results/
```

### Memory Ablations

```bash
# Compare recurrent vs non-recurrent agents
python run_memory_ablations.py --num-seeds 5

# Results saved to memory_results/
```

### Generalization Evaluation

```bash
# Test generalization to unseen environments
python run_generalization_evaluation.py \
    --checkpoint-dir baseline_logs \
    --num-episodes 100

# Results saved to test_generalization_results/
```

### Generate All Plots

```bash
# Learning curves
python generate_learning_curves.py

# Ablation plots
python generate_ablation_plots.py

# Generalization plots
python generate_generalization_plots.py
```

---

## Common Workflows

### Complete Experiment Pipeline

```bash
# 1. Run baseline experiments
python run_baseline_experiments.py --num-seeds 5 --parallel

# 2. Generate learning curves
python generate_learning_curves.py

# 3. Generate ablation plots
python generate_ablation_plots.py

# 4. Evaluate generalization
python run_generalization_evaluation.py --checkpoint-dir baseline_logs

# 5. Generate generalization plots
python generate_generalization_plots.py

# All results now in respective directories!
```

### Quick Test Run

```bash
# Fast test with minimal configuration
python train.py \
    --algorithm qlearning \
    --num-episodes 100 \
    --graph-size small \
    --output-dir test_run

# Evaluate
python evaluate.py \
    --checkpoint test_run/checkpoints/best_model.pt \
    --algorithm qlearning \
    --num-episodes 20
```

### Custom Experiment

```python
# custom_experiment.py
from src.environment.graph_generator import GraphGenerator
from src.environment.flow_environment import FlowEnvironment
from src.agents.dqn import DQNAgent
from src.agents.exploration import CountBasedBonus
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import *

# Custom configuration
graph_config = GraphConfig(
    num_nodes=20,
    branching_factor=2.5,
    max_depth=8,
    popup_probability=0.2
)

env_config = EnvConfig(
    step_penalty=-0.02,
    success_reward=2.0,
    max_steps_per_episode=60
)

# Create environment
generator = GraphGenerator()
graph = generator.generate(graph_config)
env = FlowEnvironment(graph, env_config)

# Create agent with exploration
agent = DQNAgent(
    observation_dim=env.observation_space.shape[0],
    action_space_size=env.action_space.n,
    learning_rate=0.0005
)

exploration = CountBasedBonus(beta=0.5)

# Train
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=TrainingConfig(num_episodes=3000),
    log_dir="logs/custom_experiment"
)

results = orchestrator.train()
print(f"Experiment complete! Success rate: {results['final_success_rate']:.2%}")
```

---

## Example Scripts

The repository includes several example scripts:

| Script | Description |
|--------|-------------|
| `example_training.py` | Basic training with logging |
| `example_visualization.py` | Visualization examples |
| `example_exploration_integration.py` | Exploration strategy usage |
| `example_generalization.py` | Generalization evaluation |
| `example_learning_curves.py` | Learning curve generation |
| `example_ablation_plots.py` | Ablation plot generation |

Run any example:

```bash
python example_training.py
python example_visualization.py
python example_exploration_integration.py
```

---

## Configuration Files

Pre-configured experiments are available in `configs/`:

```bash
# Use a configuration file
python run_experiments.py --config configs/dqn_baseline.json

# Available configs:
# - dqn_baseline.json
# - ppo_baseline.json
# - qlearning_small.json
# - recurrent_ppo_baseline.json
# - exploration_comparison.json
# - memory_ablation.json
```

---

## Quick Reference

### Training Commands

```bash
# Q-Learning (small graphs)
python train.py --algorithm qlearning --num-episodes 1000

# DQN (medium graphs)
python train.py --algorithm dqn --num-episodes 2000

# PPO (medium graphs)
python train.py --algorithm ppo --num-episodes 2000

# Recurrent PPO (partial observability)
python train.py --algorithm recurrent_ppo --num-episodes 2000
```

### Evaluation Commands

```bash
# Single checkpoint
python evaluate.py --checkpoint path/to/model.pt --algorithm dqn

# Multiple checkpoints
python evaluate.py --checkpoint-dir baseline_logs --num-episodes 100
```

### Plotting Commands

```bash
# Learning curves
python generate_learning_curves.py --results-dir baseline_results

# Ablation plots
python generate_ablation_plots.py --results-dir baseline_results

# Generalization plots
python generate_generalization_plots.py --results-dir test_generalization_results
```

---

## Troubleshooting

### Common Issues

**Training not improving:**
- Reduce learning rate: `--learning-rate 0.0001`
- Increase exploration: `--epsilon-decay 0.999`
- Simplify environment: `--graph-size small`

**Out of memory:**
- Reduce batch size: `--batch-size 32`
- Use CPU: `--device cpu`
- Reduce buffer size: `--buffer-capacity 5000`

**Plots not showing:**
- Save instead: `--no-show` (saves to file)
- Check backend: `export MPLBACKEND=TkAgg`

See [Troubleshooting Guide](docs/troubleshooting.md) for more help.

---

## Next Steps

### Learn More

- **Detailed Usage:** [docs/usage_examples.md](docs/usage_examples.md)
- **Configuration:** [docs/configuration_reference.md](docs/configuration_reference.md)
- **Troubleshooting:** [docs/troubleshooting.md](docs/troubleshooting.md)

### Explore Components

- **Training:** [docs/training_orchestration.md](docs/training_orchestration.md)
- **Evaluation:** [docs/evaluation_usage.md](docs/evaluation_usage.md)
- **Visualization:** [docs/visualization_usage.md](docs/visualization_usage.md)
- **Exploration:** [docs/exploration_strategies.md](docs/exploration_strategies.md)
- **Generalization:** [docs/generalization_evaluation.md](docs/generalization_evaluation.md)

### Run Full Experiments

- **Baselines:** [BASELINE_EXPERIMENTS.md](BASELINE_EXPERIMENTS.md)
- **Exploration:** [EXPLORATION_ABLATIONS.md](EXPLORATION_ABLATIONS.md)
- **Memory:** [MEMORY_ABLATIONS.md](MEMORY_ABLATIONS.md)
- **Generalization:** [GENERALIZATION_EVALUATION_QUICK_START.md](GENERALIZATION_EVALUATION_QUICK_START.md)

### View Results

- **Learning Curves:** [LEARNING_CURVES_QUICK_START.md](LEARNING_CURVES_QUICK_START.md)
- **Ablation Plots:** [ABLATION_PLOTS_QUICK_START.md](ABLATION_PLOTS_QUICK_START.md)
- **Generalization Plots:** [GENERALIZATION_PLOTS_QUICK_START.md](GENERALIZATION_PLOTS_QUICK_START.md)

---

## Quick Help

```bash
# Get help for any script
python train.py --help
python evaluate.py --help
python run_baseline_experiments.py --help

# View available algorithms
python train.py --list-algorithms

# View available metrics
python generate_learning_curves.py --list-metrics
```

---

## Summary

**To get started quickly:**

1. Install dependencies: `pip install -r requirements.txt`
2. Run example: `python example_training.py`
3. View logs: `tensorboard --logdir logs`
4. Generate plots: `python generate_learning_curves.py`

**For full experiments:**

1. Run baselines: `python run_baseline_experiments.py --num-seeds 5`
2. Generate all plots: `python generate_*.py`
3. View results in respective directories

**For custom experiments:**

1. Copy and modify example scripts
2. Or use configuration files in `configs/`
3. Or write custom Python code using the API

Happy experimenting! 🚀
