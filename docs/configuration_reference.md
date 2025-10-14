# Configuration Reference

This document provides a comprehensive reference for all configuration options in the RL Flow Explorer system.

## Table of Contents

1. [Graph Configuration](#graph-configuration)
2. [Environment Configuration](#environment-configuration)
3. [Training Configuration](#training-configuration)
4. [Agent Configuration](#agent-configuration)
5. [Exploration Configuration](#exploration-configuration)
6. [Evaluation Configuration](#evaluation-configuration)
7. [Visualization Configuration](#visualization-configuration)
8. [Experiment Configuration Files](#experiment-configuration-files)

---

## Graph Configuration

Configuration for graph generation (GraphConfig dataclass).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | int | Required | Number of nodes in the graph |
| `branching_factor` | float | 0.5 | Average number of outgoing edges per node |
| `max_depth` | int | None | Maximum depth from start to goal |
| `depth` | int | None | Alias for max_depth |
| `popup_probability` | float | 0.0 | Probability of stochastic popup failure |
| `dead_end_probability` | float | 0.0 | Probability of dead-end nodes |
| `dependency_probability` | float | 0.0 | Probability of hidden dependencies |
| `num_dependencies` | int | 0 | Number of hidden dependencies to inject |
| `random_seed` | int | None | Random seed for reproducibility |

### Example

```python
from src.utils.data_models import GraphConfig

config = GraphConfig(
    num_nodes=15,
    branching_factor=2.0,
    max_depth=7,
    popup_probability=0.15,
    dead_end_probability=0.1,
    dependency_probability=0.1,
    num_dependencies=2,
    random_seed=42
)
```

### Recommended Values

**Small Graphs (Quick Testing)**
```python
GraphConfig(
    num_nodes=8,
    branching_factor=1.5,
    max_depth=4,
    popup_probability=0.1
)
```

**Medium Graphs (Standard Training)**
```python
GraphConfig(
    num_nodes=15,
    branching_factor=2.0,
    max_depth=7,
    popup_probability=0.15,
    num_dependencies=1
)
```

**Large Graphs (Challenging)**
```python
GraphConfig(
    num_nodes=25,
    branching_factor=2.5,
    max_depth=10,
    popup_probability=0.2,
    num_dependencies=3
)
```

---

## Environment Configuration

Configuration for FlowEnvironment (EnvConfig dataclass).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | int | None | Number of nodes (usually from graph) |
| `step_penalty` | float | -0.01 | Penalty for each step taken |
| `success_reward` | float | 1.0 | Reward for reaching goal |
| `failure_penalty` | float | -0.5 | Penalty for triggering failure |
| `max_steps_per_episode` | int | 100 | Maximum steps before timeout |
| `max_steps` | int | None | Alias for max_steps_per_episode |
| `random_seed` | int | None | Random seed for reproducibility |
| `partial_observability` | float | 1.0 | Fraction of state visible (1.0 = full) |

### Example

```python
from src.utils.data_models import EnvConfig

config = EnvConfig(
    step_penalty=-0.01,
    success_reward=1.0,
    failure_penalty=-0.5,
    max_steps_per_episode=50,
    random_seed=42,
    partial_observability=0.7  # 70% observability
)
```

### Reward Shaping Guidelines

**Sparse Rewards (Challenging)**
```python
EnvConfig(
    step_penalty=0.0,
    success_reward=1.0,
    failure_penalty=-1.0
)
```

**Dense Rewards (Easier Learning)**
```python
EnvConfig(
    step_penalty=-0.01,
    success_reward=1.0,
    failure_penalty=-0.5
)
```

**Efficiency-Focused**
```python
EnvConfig(
    step_penalty=-0.05,  # Stronger penalty
    success_reward=1.0,
    failure_penalty=-1.0
)
```

---

## Training Configuration

Configuration for training orchestration (TrainingConfig dataclass).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_episodes` | int | Required | Total number of training episodes |
| `max_steps_per_episode` | int | 100 | Maximum steps per episode |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `discount_factor` | float | 0.99 | Discount factor (gamma) |
| `batch_size` | int | 32 | Batch size for updates |
| `eval_frequency` | int | 100 | Episodes between evaluations |
| `num_eval_episodes` | int | 10 | Episodes per evaluation |
| `random_seed` | int | None | Random seed for reproducibility |
| `checkpoint_frequency` | int | 500 | Episodes between checkpoints |
| `log_frequency` | int | 10 | Episodes between logging |
| `early_stopping_patience` | int | None | Episodes without improvement before stopping |
| `save_dir` | str | "checkpoints" | Directory for saving checkpoints |

### Example

```python
from src.utils.data_models import TrainingConfig

config = TrainingConfig(
    num_episodes=2000,
    max_steps_per_episode=50,
    learning_rate=0.001,
    discount_factor=0.99,
    batch_size=64,
    eval_frequency=100,
    num_eval_episodes=20,
    random_seed=42,
    checkpoint_frequency=200,
    log_frequency=10,
    early_stopping_patience=500,
    save_dir="checkpoints/dqn"
)
```

### Recommended Values by Algorithm

**Q-Learning**
```python
TrainingConfig(
    num_episodes=1000,
    learning_rate=0.1,
    discount_factor=0.99,
    batch_size=1  # Online learning
)
```

**DQN**
```python
TrainingConfig(
    num_episodes=2000,
    learning_rate=0.001,
    discount_factor=0.99,
    batch_size=64
)
```

**PPO**
```python
TrainingConfig(
    num_episodes=2000,
    learning_rate=0.0003,
    discount_factor=0.99,
    batch_size=64
)
```

---

## Agent Configuration

### Q-Learning Agent

```python
from src.agents.q_learning import QLearningAgent

agent = QLearningAgent(
    state_dim=10,              # State space dimension
    action_dim=5,              # Action space size
    learning_rate=0.1,         # Alpha
    discount_factor=0.99,      # Gamma
    epsilon=1.0,               # Initial exploration rate
    epsilon_decay=0.995,       # Decay per episode
    epsilon_min=0.01           # Minimum epsilon
)
```

### DQN Agent

```python
from src.agents.dqn import DQNAgent

agent = DQNAgent(
    observation_dim=10,        # Observation space dimension
    action_space_size=5,       # Action space size
    learning_rate=0.001,       # Optimizer learning rate
    discount_factor=0.99,      # Gamma
    epsilon=1.0,               # Initial exploration rate
    epsilon_decay=0.995,       # Decay per episode
    epsilon_min=0.01,          # Minimum epsilon
    buffer_capacity=10000,     # Replay buffer size
    batch_size=64,             # Batch size for updates
    target_update_frequency=100,  # Steps between target updates
    hidden_sizes=[128, 128],   # Network architecture
    device="cuda"              # Device (cuda/cpu)
)
```

### PPO Agent

```python
from src.agents.ppo import PPOAgent

agent = PPOAgent(
    observation_dim=10,        # Observation space dimension
    action_space_size=5,       # Action space size
    learning_rate=0.0003,      # Optimizer learning rate
    discount_factor=0.99,      # Gamma
    gae_lambda=0.95,           # GAE lambda
    clip_epsilon=0.2,          # PPO clip parameter
    entropy_coef=0.01,         # Entropy regularization
    value_coef=0.5,            # Value loss coefficient
    max_grad_norm=0.5,         # Gradient clipping
    hidden_sizes=[128, 128],   # Network architecture
    device="cuda"              # Device (cuda/cpu)
)
```

### Recurrent PPO Agent

```python
from src.agents.recurrent_ppo import RecurrentPPOAgent

agent = RecurrentPPOAgent(
    observation_dim=10,        # Observation space dimension
    action_space_size=5,       # Action space size
    hidden_size=128,           # LSTM hidden size
    learning_rate=0.0003,      # Optimizer learning rate
    discount_factor=0.99,      # Gamma
    gae_lambda=0.95,           # GAE lambda
    clip_epsilon=0.2,          # PPO clip parameter
    entropy_coef=0.01,         # Entropy regularization
    value_coef=0.5,            # Value loss coefficient
    max_grad_norm=0.5,         # Gradient clipping
    num_layers=1,              # Number of LSTM layers
    device="cuda"              # Device (cuda/cpu)
)
```

---

## Exploration Configuration

### Epsilon-Greedy

```python
from src.agents.exploration import EpsilonGreedy

exploration = EpsilonGreedy(
    epsilon=1.0,               # Initial epsilon
    epsilon_min=0.01,          # Minimum epsilon
    epsilon_decay=0.995,       # Decay factor
    decay_mode="episode"       # "episode" or "step"
)
```

### Count-Based Exploration

```python
from src.agents.exploration import CountBasedBonus

exploration = CountBasedBonus(
    beta=0.5,                  # Bonus scaling factor
    count_mode="state",        # "state" or "state_action"
    bonus_type="inverse_sqrt"  # Bonus computation method
)
```

**Bonus Types:**
- `"inverse_sqrt"`: bonus = beta / sqrt(count)
- `"inverse"`: bonus = beta / count
- `"log"`: bonus = beta * log(1 + 1/count)

### Curiosity-Based Exploration

```python
from src.agents.exploration import CuriosityBonus

exploration = CuriosityBonus(
    state_dim=10,              # State dimension
    action_dim=5,              # Action dimension
    beta=0.1,                  # Bonus scaling factor
    learning_rate=0.001,       # Forward model learning rate
    hidden_size=128,           # Forward model hidden size
    batch_size=32,             # Batch size for training
    train_frequency=1,         # Steps between training
    buffer_capacity=1000,      # Transition buffer size
    device="cuda"              # Device (cuda/cpu)
)
```

---

## Evaluation Configuration

### Evaluator

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(
    verbose=True,              # Print progress
    random_seed=42             # Random seed
)

# Evaluate single environment
results = evaluator.evaluate_single_env(
    agent=agent,
    env=env,
    num_episodes=100,          # Episodes to run
    agent_name="DQN",          # Name for logging
    render=False               # Render episodes
)

# Evaluate multiple environments
results = evaluator.evaluate_multiple_envs(
    agent=agent,
    envs=envs,
    num_episodes_per_env=50,   # Episodes per environment
    agent_name="DQN",
    parallel=False             # Parallel evaluation
)
```

### Generalization Evaluator

```python
from src.evaluation.generalization import GeneralizationEvaluator

gen_evaluator = GeneralizationEvaluator(
    verbose=True,              # Print progress
    random_seed=42             # Random seed
)

results = gen_evaluator.evaluate_zero_shot(
    agent=agent,
    graph_splits=splits,       # Dict of graph splits
    env_config=env_config,
    num_episodes_per_env=100,  # Episodes per environment
    agent_name="DQN"
)
```

---

## Visualization Configuration

### PlotGenerator

```python
from src.visualization.plot_generator import PlotGenerator

plotter = PlotGenerator(
    output_dir="plots",        # Output directory
    figsize=(10, 6),           # Figure size (width, height)
    dpi=300,                   # Resolution
    style="seaborn-v0_8",      # Matplotlib style
    random_seed=42             # Random seed
)
```

### Plot Methods

**Learning Curves**
```python
plotter.plot_learning_curves(
    results=results_dict,      # Dict of result paths
    metric="reward",           # Metric to plot
    title="Learning Curves",   # Plot title
    xlabel="Episode",          # X-axis label
    ylabel="Reward",           # Y-axis label
    save_path="curves.png",    # Save path
    show=True,                 # Display plot
    smoothing_window=10,       # Smoothing window size
    confidence_level=0.95      # Confidence interval
)
```

**Algorithm Comparison**
```python
plotter.plot_algorithm_comparison(
    results=metrics_dict,      # Dict of AggregateMetrics
    metric="success_rate",     # Metric to compare
    title="Comparison",        # Plot title
    save_path="comparison.png",
    show=True
)
```

**Ablation Table**
```python
plotter.plot_ablation_table(
    ablation_results=results,  # Dict of AggregateMetrics
    save_path="ablation.png",
    show=True,
    metrics=["success_rate", "mean_steps", "coverage"]
)
```

---

## Experiment Configuration Files

### JSON Configuration Format

Example: `configs/dqn_baseline.json`

```json
{
  "experiment_name": "dqn_baseline",
  "algorithm": "dqn",
  "graph_config": {
    "num_nodes": 15,
    "branching_factor": 2.0,
    "max_depth": 7,
    "popup_probability": 0.15,
    "num_dependencies": 1
  },
  "env_config": {
    "step_penalty": -0.01,
    "success_reward": 1.0,
    "failure_penalty": -0.5,
    "max_steps_per_episode": 50
  },
  "training_config": {
    "num_episodes": 2000,
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "batch_size": 64,
    "eval_frequency": 100,
    "num_eval_episodes": 20
  },
  "agent_config": {
    "learning_rate": 0.001,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "buffer_capacity": 10000,
    "target_update_frequency": 100
  },
  "num_train_graphs": 50,
  "num_val_graphs": 10,
  "random_seeds": [42, 142, 242, 342, 442]
}
```

### Loading Configuration

```python
import json
from pathlib import Path

# Load configuration
config_path = Path("configs/dqn_baseline.json")
with open(config_path) as f:
    config = json.load(f)

# Use configuration
graph_config = GraphConfig(**config["graph_config"])
env_config = EnvConfig(**config["env_config"])
training_config = TrainingConfig(**config["training_config"])
```

---

## Environment Variables

The system supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RL_FLOW_LOG_DIR` | Base directory for logs | `logs` |
| `RL_FLOW_CHECKPOINT_DIR` | Base directory for checkpoints | `checkpoints` |
| `RL_FLOW_RESULTS_DIR` | Base directory for results | `results` |
| `RL_FLOW_PLOTS_DIR` | Base directory for plots | `plots` |
| `RL_FLOW_DEVICE` | Device for PyTorch (cuda/cpu) | `cuda` if available |
| `RL_FLOW_NUM_WORKERS` | Number of parallel workers | CPU count |

### Setting Environment Variables

**Linux/Mac:**
```bash
export RL_FLOW_LOG_DIR=/path/to/logs
export RL_FLOW_DEVICE=cuda
```

**Windows:**
```cmd
set RL_FLOW_LOG_DIR=C:\path\to\logs
set RL_FLOW_DEVICE=cuda
```

**Python:**
```python
import os
os.environ["RL_FLOW_LOG_DIR"] = "/path/to/logs"
os.environ["RL_FLOW_DEVICE"] = "cuda"
```

---

## See Also

- [Usage Examples](usage_examples.md)
- [Training Orchestration](training_orchestration.md)
- [Evaluation Usage](evaluation_usage.md)
- [Visualization Usage](visualization_usage.md)
- [Exploration Strategies](exploration_strategies.md)
