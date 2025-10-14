# Usage Examples

This document provides comprehensive examples for using each component of the RL Flow Explorer system.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Training Agents](#training-agents)
3. [Evaluation](#evaluation)
4. [Visualization](#visualization)
5. [Exploration Strategies](#exploration-strategies)
6. [Generalization Testing](#generalization-testing)
7. [Running Experiments](#running-experiments)

---

## Environment Setup

### Creating a Simple Environment

```python
from src.environment.graph_generator import GraphGenerator
from src.environment.flow_environment import FlowEnvironment
from src.utils.data_models import GraphConfig, EnvConfig

# Configure graph generation
graph_config = GraphConfig(
    num_nodes=10,
    branching_factor=2.0,
    max_depth=5,
    popup_probability=0.1,
    dead_end_probability=0.1,
    dependency_probability=0.1,
    num_dependencies=1,
    random_seed=42
)

# Generate graph
generator = GraphGenerator()
graph = generator.generate(graph_config)

# Configure environment
env_config = EnvConfig(
    step_penalty=-0.01,
    success_reward=1.0,
    failure_penalty=-0.5,
    max_steps_per_episode=50,
    random_seed=42
)

# Create environment
env = FlowEnvironment(graph, env_config)

# Use the environment
obs, info = env.reset()
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
```

### Creating Multiple Environments for Training

```python
# Generate multiple training graphs
num_train_envs = 10
train_graphs = [generator.generate(graph_config) for _ in range(num_train_envs)]
train_envs = [FlowEnvironment(g, env_config) for g in train_graphs]

# Generate validation graphs
num_val_envs = 3
val_graphs = [generator.generate(graph_config) for _ in range(num_val_envs)]
val_envs = [FlowEnvironment(g, env_config) for g in val_graphs]
```

---

## Training Agents

### Training Q-Learning Agent

```python
from src.agents.q_learning import QLearningAgent
from src.training.orchestrator import TrainingOrchestrator
from src.utils.data_models import TrainingConfig

# Create agent
agent = QLearningAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Configure training
training_config = TrainingConfig(
    num_episodes=1000,
    max_steps_per_episode=50,
    learning_rate=0.1,
    discount_factor=0.99,
    eval_frequency=100,
    num_eval_episodes=10,
    random_seed=42,
    checkpoint_frequency=200,
    log_frequency=10
)

# Create orchestrator
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=training_config,
    val_envs=val_envs,
    log_dir="logs/q_learning",
    experiment_name="q_learning_baseline"
)

# Train
results = orchestrator.train()
```

### Training DQN Agent

```python
from src.agents.dqn import DQNAgent

# Create DQN agent
agent = DQNAgent(
    observation_dim=env.observation_space.shape[0],
    action_space_size=env.action_space.n,
    learning_rate=0.001,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    buffer_capacity=10000,
    batch_size=64,
    target_update_frequency=100
)

# Training configuration
training_config = TrainingConfig(
    num_episodes=2000,
    max_steps_per_episode=50,
    learning_rate=0.001,
    discount_factor=0.99,
    batch_size=64,
    eval_frequency=100,
    num_eval_episodes=10,
    random_seed=42
)

# Train using orchestrator
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=training_config,
    log_dir="logs/dqn"
)

results = orchestrator.train()
```

### Training PPO Agent

```python
from src.agents.ppo import PPOAgent

# Create PPO agent
agent = PPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_space_size=env.action_space.n,
    learning_rate=0.0003,
    discount_factor=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5
)

# Training configuration
training_config = TrainingConfig(
    num_episodes=2000,
    max_steps_per_episode=50,
    learning_rate=0.0003,
    discount_factor=0.99,
    batch_size=64,
    eval_frequency=100,
    num_eval_episodes=10,
    random_seed=42
)

# Train
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=training_config,
    log_dir="logs/ppo"
)

results = orchestrator.train()
```

### Training Recurrent PPO Agent

```python
from src.agents.recurrent_ppo import RecurrentPPOAgent

# Create recurrent PPO agent
agent = RecurrentPPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_space_size=env.action_space.n,
    hidden_size=128,
    learning_rate=0.0003,
    discount_factor=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01
)

# Train (same as regular PPO)
orchestrator = TrainingOrchestrator(
    agent=agent,
    env=env,
    config=training_config,
    log_dir="logs/recurrent_ppo"
)

results = orchestrator.train()
```

---

## Evaluation

### Basic Evaluation

```python
from src.evaluation.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(verbose=True)

# Evaluate on single environment
results = evaluator.evaluate_single_env(
    agent=agent,
    env=env,
    num_episodes=100,
    agent_name="Q-Learning"
)

# Print results
print(f"Success Rate: {results.metrics.success_rate:.2%}")
print(f"Mean Steps: {results.metrics.mean_steps_to_success:.2f}")
print(f"Mean Reward: {results.metrics.mean_reward:.2f}")
print(f"Coverage: {results.metrics.mean_coverage:.2%}")
```

### Evaluating Across Multiple Environments

```python
# Evaluate on multiple test environments
test_results = evaluator.evaluate_multiple_envs(
    agent=agent,
    envs=test_envs,
    num_episodes_per_env=50,
    agent_name="Q-Learning"
)

# Aggregate results
print(f"Average Success Rate: {test_results.metrics.success_rate:.2%}")
print(f"Std Dev: {test_results.metrics.success_rate_std:.3f}")
```

### Comparing Multiple Agents

```python
# Create multiple agents
agents = {
    "Q-Learning": q_agent,
    "DQN": dqn_agent,
    "PPO": ppo_agent
}

# Evaluate all agents
comparison_results = {}
for name, agent in agents.items():
    results = evaluator.evaluate_single_env(
        agent=agent,
        env=env,
        num_episodes=100,
        agent_name=name
    )
    comparison_results[name] = results.metrics

# Print comparison
for name, metrics in comparison_results.items():
    print(f"{name}: {metrics.success_rate:.2%} success rate")
```

---

## Visualization

### Generating Learning Curves

```python
from src.visualization.plot_generator import PlotGenerator

# Create plot generator
plotter = PlotGenerator(
    output_dir="plots",
    dpi=300,
    random_seed=42
)

# Plot learning curves from training logs
results = {
    "Q-Learning": "logs/q_learning/training_metrics.json",
    "DQN": "logs/dqn/training_metrics.json",
    "PPO": "logs/ppo/training_metrics.json"
}

plotter.plot_learning_curves(
    results=results,
    metric="reward",
    title="Training Performance Comparison",
    save_path="plots/learning_curves.png",
    show=True,
    smoothing_window=10
)
```

### Creating Ablation Tables

```python
# Create ablation comparison
ablation_results = {
    "Baseline": baseline_metrics,
    "+ Count Exploration": count_metrics,
    "+ Curiosity": curiosity_metrics,
    "+ Memory": memory_metrics
}

plotter.plot_ablation_table(
    ablation_results=ablation_results,
    save_path="plots/ablation_table.png",
    show=True
)
```

### Algorithm Comparison Plots

```python
# Compare algorithms on specific metric
plotter.plot_algorithm_comparison(
    results=comparison_results,
    metric="success_rate",
    title="Algorithm Performance Comparison",
    save_path="plots/algorithm_comparison.png",
    show=True
)
```

### Generalization Plots

```python
# Plot generalization performance
plotter.plot_generalization(
    in_dist_results=in_dist_metrics,
    out_dist_results=out_dist_metrics,
    save_path="plots/generalization.png",
    show=True
)

# Plot performance degradation
plotter.plot_performance_degradation(
    in_dist_results=in_dist_metrics,
    out_dist_results=out_dist_metrics,
    metric="success_rate",
    save_path="plots/degradation.png",
    show=True
)
```

### Coverage Heatmaps

```python
# Plot state coverage
plotter.plot_coverage_heatmap(
    state_visitation=visitation_matrix,
    title="State Visitation Across Episodes",
    save_path="plots/coverage_heatmap.png",
    show=True
)
```

---

## Exploration Strategies

### Epsilon-Greedy Exploration

```python
from src.agents.exploration import EpsilonGreedy

# Create epsilon-greedy strategy
exploration = EpsilonGreedy(
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    decay_mode="episode"  # or "step"
)

# Use during training
if exploration.should_explore():
    action = env.action_space.sample()
else:
    action = agent.select_action(obs, training=False)

# Decay after episode
exploration.decay()
```

### Count-Based Exploration

```python
from src.agents.exploration import CountBasedBonus

# Create count-based exploration
exploration = CountBasedBonus(
    beta=0.5,
    count_mode="state"  # or "state_action"
)

# Compute intrinsic reward
intrinsic_reward = exploration.compute_bonus(state, action, next_state)

# Total reward for learning
total_reward = extrinsic_reward + intrinsic_reward

# Update counts
exploration.update(state, action, next_state)

# Get statistics
stats = exploration.get_statistics()
print(f"Unique states visited: {stats['unique_states']}")
print(f"Mean intrinsic reward: {stats['mean_bonus']:.4f}")
```

### Curiosity-Based Exploration

```python
from src.agents.exploration import CuriosityBonus

# Create curiosity-based exploration
exploration = CuriosityBonus(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    beta=0.1,
    learning_rate=0.001,
    batch_size=32,
    train_frequency=1,
    device="cuda"  # or "cpu"
)

# Compute intrinsic reward (prediction error)
intrinsic_reward = exploration.compute_bonus(state, action, next_state)

# Total reward
total_reward = extrinsic_reward + intrinsic_reward

# Train forward model
exploration.update(state, action, next_state)

# Get statistics
stats = exploration.get_statistics()
print(f"Mean prediction error: {stats['mean_prediction_error']:.6f}")
```

---

## Generalization Testing

### Creating Train/Test Splits

```python
from src.evaluation.generalization import GeneralizationEvaluator

# Create evaluator
gen_evaluator = GeneralizationEvaluator(verbose=True)

# Create graph splits
generator = GraphGenerator(random_seed=42)
base_config = GraphConfig(
    num_nodes=12,
    branching_factor=0.5,
    depth=6
)

splits = generator.create_standard_splits(
    base_config=base_config,
    num_train=50,
    num_test_in_dist=20,
    num_test_per_shift=10
)

print(f"Training graphs: {len(splits['train'])}")
print(f"In-dist test: {len(splits['test_in_dist'])}")
print(f"Deeper graphs: {len(splits['test_deeper'])}")
print(f"Wider graphs: {len(splits['test_wider'])}")
```

### Zero-Shot Evaluation

```python
# Evaluate generalization
results = gen_evaluator.evaluate_zero_shot(
    agent=agent,
    graph_splits=splits,
    env_config=env_config,
    num_episodes_per_env=100,
    agent_name="DQN"
)

# Print results
print(f"In-Dist Success: {results.in_dist_results.metrics.success_rate:.2%}")

for shift_name, out_results in results.out_dist_results.items():
    degradation = results.degradation_metrics[shift_name]
    print(f"{shift_name}: {out_results.metrics.success_rate:.2%} "
          f"({degradation.success_rate_drop_pct:+.1f}%)")
```

### Custom Distribution Shifts

```python
# Create custom distribution shift
custom_config = GraphConfig(
    num_nodes=20,  # Larger graphs
    branching_factor=1.0,  # More branching
    max_depth=10,  # Deeper
    popup_probability=0.3,  # More popups
    random_seed=42
)

custom_graphs = [generator.generate(custom_config) for _ in range(10)]
custom_envs = [FlowEnvironment(g, env_config) for g in custom_graphs]

# Evaluate on custom shift
custom_results = evaluator.evaluate_multiple_envs(
    agent=agent,
    envs=custom_envs,
    num_episodes_per_env=50,
    agent_name="DQN"
)

print(f"Custom Shift Success: {custom_results.metrics.success_rate:.2%}")
```

---

## Running Experiments

### Running Baseline Experiments

```bash
# Run all baseline experiments
python run_baseline_experiments.py --num-seeds 5 --parallel

# Run specific algorithm
python run_baseline_experiments.py --algorithms dqn --num-seeds 3

# Custom configuration
python run_baseline_experiments.py \
    --num-seeds 5 \
    --num-episodes 2000 \
    --output-dir my_results
```

### Running Exploration Ablations

```bash
# Run exploration ablation study
python run_exploration_ablations.py --num-seeds 5

# Specific strategies
python run_exploration_ablations.py \
    --strategies epsilon count curiosity \
    --num-seeds 3
```

### Running Memory Ablations

```bash
# Run memory ablation study
python run_memory_ablations.py --num-seeds 5

# Custom observability levels
python run_memory_ablations.py \
    --observability-levels 0.3 0.5 0.7 1.0 \
    --num-seeds 3
```

### Running Generalization Evaluation

```bash
# Run generalization evaluation
python run_generalization_evaluation.py \
    --checkpoint-dir baseline_logs \
    --num-episodes 100

# Specific agents
python run_generalization_evaluation.py \
    --agents dqn ppo \
    --checkpoint-dir baseline_logs
```

### Generating Plots

```bash
# Generate learning curves
python generate_learning_curves.py \
    --results-dir baseline_results \
    --output-dir learning_curves

# Generate ablation plots
python generate_ablation_plots.py \
    --results-dir baseline_results \
    --output-dir ablation_plots

# Generate generalization plots
python generate_generalization_plots.py \
    --results-dir test_generalization_results \
    --output-dir generalization_plots
```

---

## Advanced Usage

### Custom Training Loop

```python
from src.utils.data_models import Transition

# Manual training loop for custom logic
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Select action
        action = agent.select_action(obs, training=True)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Compute intrinsic reward (if using exploration)
        intrinsic_reward = exploration.compute_bonus(obs, action, next_obs)
        total_reward = reward + intrinsic_reward
        
        # Create transition
        transition = Transition(
            state=obs,
            action=action,
            reward=total_reward,
            next_state=next_obs,
            done=done,
            info=info
        )
        
        # Update agent
        agent.update(transition)
        
        # Update exploration
        exploration.update(obs, action, next_obs)
        
        obs = next_obs
        episode_reward += reward
    
    # Decay exploration
    agent.decay_epsilon()
    exploration.decay()
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
```

### Saving and Loading Models

```python
# Save agent
agent.save("checkpoints/my_agent.pt")

# Load agent
agent.load("checkpoints/my_agent.pt")

# Save with metadata
import json
metadata = {
    "algorithm": "DQN",
    "episodes_trained": 2000,
    "success_rate": 0.85,
    "config": training_config.__dict__
}

with open("checkpoints/my_agent_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### Custom Metrics Collection

```python
from src.evaluation.metrics import MetricsCollector

# Create metrics collector
collector = MetricsCollector()

# Collect episode metrics
episode_metrics = collector.evaluate_episode(trajectory)

# Aggregate across episodes
all_metrics = [collector.evaluate_episode(traj) for traj in trajectories]
aggregate_metrics = collector.aggregate(all_metrics)

# Access specific metrics
print(f"Success Rate: {aggregate_metrics.success_rate:.2%}")
print(f"Mean Steps: {aggregate_metrics.mean_steps_to_success:.2f}")
print(f"Coverage: {aggregate_metrics.mean_coverage:.2%}")
```

---

## See Also

- [Training Orchestration](training_orchestration.md)
- [Evaluation Usage](evaluation_usage.md)
- [Visualization Usage](visualization_usage.md)
- [Exploration Strategies](exploration_strategies.md)
- [Generalization Evaluation](generalization_evaluation.md)
- [Experiment Scripts Usage](experiment_scripts_usage.md)
