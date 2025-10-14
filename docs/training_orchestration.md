# Training Orchestration

This document describes the training orchestration system for RL agents, including logging, monitoring, and checkpointing capabilities.

## Overview

The `TrainingOrchestrator` class provides a complete training pipeline with:

- **Episode collection and agent updates**
- **TensorBoard integration for real-time monitoring**
- **JSON logging for post-hoc analysis**
- **Checkpoint saving (best model + periodic snapshots)**
- **Validation evaluation**
- **Early stopping**
- **Deterministic logging with fixed random seeds**

## Components

### TrainingOrchestrator

Main class that coordinates the training loop.

**Key Features:**
- Manages training episodes and agent updates
- Logs metrics to TensorBoard and JSON files
- Saves checkpoints periodically and when validation improves
- Implements early stopping based on validation performance
- Ensures reproducibility through deterministic seeding

**Usage:**

```python
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import TrainingConfig

# Create orchestrator
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=train_env,
    config=training_config,
    val_envs=val_envs,  # Optional validation environments
    log_dir="logs/experiment_1",
    experiment_name="my_experiment"
)

# Run training
results = orchestrator.train()
```

### ExperimentLogger

Utility class for structured logging with deterministic output.

**Features:**
- Logs configuration and hyperparameters to JSON
- Logs training metrics with timestamps
- Saves final results in JSON format
- Converts numpy types for JSON serialization

**Usage:**

```python
from src.utils.logger import ExperimentLogger

logger = ExperimentLogger(log_dir="logs", experiment_name="exp_1")
logger.log_config(config_dict)
logger.log_hyperparameters(hyperparameters)
logger.log_metrics(metrics, step=100)
logger.save_results(results)
```

## Training Configuration

The `TrainingConfig` dataclass defines all training parameters:

```python
@dataclass
class TrainingConfig:
    num_episodes: int = 1000              # Total training episodes
    max_steps_per_episode: int = 100      # Max steps per episode
    learning_rate: float = 0.001          # Agent learning rate
    discount_factor: float = 0.99         # Reward discount factor
    batch_size: int = 32                  # Batch size for updates
    eval_frequency: int = 100             # Episodes between evaluations
    num_eval_episodes: int = 10           # Episodes per evaluation
    random_seed: int = 42                 # Random seed for reproducibility
    checkpoint_frequency: int = 500       # Episodes between checkpoints
    log_frequency: int = 10               # Episodes between log outputs
    early_stopping_patience: int = 50     # Episodes without improvement before stopping
    save_dir: str = "checkpoints"         # Directory for checkpoints
```

## Logged Metrics

### Training Metrics (per episode)

- **reward**: Total episode reward
- **steps**: Number of steps in episode
- **success**: Whether episode reached goal
- **failure**: Whether episode triggered failure
- **Agent-specific metrics**: Loss values, Q-values, etc.

### Validation Metrics (periodic)

- **mean_reward**: Average reward across validation episodes
- **std_reward**: Standard deviation of rewards
- **success_rate**: Percentage of successful episodes
- **mean_steps**: Average steps to completion

## Output Structure

```
logs/
├── experiment_name/
│   ├── tensorboard/              # TensorBoard logs
│   │   └── events.out.tfevents.*
│   ├── checkpoints/              # Model checkpoints
│   │   ├── best_model.pt
│   │   ├── best_model.json
│   │   ├── final_model.pt
│   │   ├── final_model.json
│   │   └── checkpoint_ep*.pt
│   ├── experiment_name.log       # Text log file
│   ├── experiment_name_config.json
│   ├── experiment_name_hyperparameters.json
│   ├── experiment_name_results.json
│   └── training_metrics.json     # All episode metrics
```

## Monitoring with TensorBoard

To view training progress in real-time:

```bash
tensorboard --logdir logs/experiment_name/tensorboard
```

Then open http://localhost:6006 in your browser.

**Available plots:**
- `train/reward`: Episode rewards over time
- `train/steps`: Episode lengths over time
- `train/success`: Success indicators over time
- `train/loss`: Agent loss values (if applicable)
- `val/mean_reward`: Validation performance
- `val/success_rate`: Validation success rate

## Reproducibility

The training system ensures reproducibility through:

1. **Fixed random seeds** for Python, NumPy, PyTorch
2. **Deterministic CUDA operations** (when using GPU)
3. **Configuration logging** (all hyperparameters saved)
4. **Deterministic plotting** (consistent random state)

To reproduce results:
1. Use the same `random_seed` in `TrainingConfig`
2. Use the same environment and agent configurations
3. Run on the same hardware (CPU vs GPU can affect results)

## Checkpointing

### Automatic Checkpointing

- **Best model**: Saved when validation performance improves
- **Periodic checkpoints**: Saved every `checkpoint_frequency` episodes
- **Final model**: Saved at end of training

### Loading Checkpoints

```python
# Load best model
agent.load("logs/experiment_name/checkpoints/best_model.pt")

# Load specific checkpoint
agent.load("logs/experiment_name/checkpoints/checkpoint_ep500.pt")
```

## Early Stopping

Training stops early if validation performance doesn't improve for `early_stopping_patience` episodes.

This prevents overfitting and saves computation time.

## Example: Complete Training Pipeline

```python
import numpy as np
from src.environment.graph_generator import GraphGenerator
from src.environment.flow_environment import FlowEnvironment
from src.agents.dqn import DQNAgent
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import GraphConfig, EnvConfig, TrainingConfig

# Generate environments
generator = GraphGenerator()
graph_config = GraphConfig(num_nodes=10, random_seed=42)
train_graph = generator.generate(graph_config)
val_graphs = [generator.generate(graph_config) for _ in range(5)]

env_config = EnvConfig(random_seed=42)
train_env = FlowEnvironment(train_graph, env_config)
val_envs = [FlowEnvironment(g, env_config) for g in val_graphs]

# Create agent
agent = DQNAgent(
    state_dim=train_env.observation_space.shape[0],
    action_dim=train_env.action_space.n,
    learning_rate=0.001,
)

# Configure training
training_config = TrainingConfig(
    num_episodes=1000,
    eval_frequency=100,
    random_seed=42,
)

# Train
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=train_env,
    config=training_config,
    val_envs=val_envs,
    log_dir="logs/dqn_experiment",
)

results = orchestrator.train()
print(f"Best validation reward: {results['best_val_reward']:.2f}")
```

## Best Practices

1. **Use validation environments** to monitor generalization
2. **Set appropriate eval_frequency** (too frequent slows training, too rare misses improvements)
3. **Monitor TensorBoard** during training to catch issues early
4. **Save configuration files** with your experiments for reproducibility
5. **Use meaningful experiment names** for easy identification
6. **Set random seeds** for reproducible results
7. **Adjust early_stopping_patience** based on problem difficulty

## Troubleshooting

### Training is slow
- Reduce `eval_frequency` or `num_eval_episodes`
- Reduce `log_frequency`
- Use fewer validation environments

### Metrics not improving
- Check TensorBoard for learning curves
- Verify agent is updating correctly
- Try different hyperparameters
- Check environment is solvable

### Out of memory
- Reduce batch size
- Reduce replay buffer size (for DQN)
- Use gradient accumulation

### Results not reproducible
- Ensure same random seed
- Check for non-deterministic operations
- Verify same hardware (CPU vs GPU)
