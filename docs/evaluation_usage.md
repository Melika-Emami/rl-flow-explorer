# Evaluation Module Usage Guide

## Overview

The evaluation module provides comprehensive tools for evaluating trained RL agents on test environments. It computes detailed metrics, aggregates statistics across multiple episodes, and provides confidence intervals for rigorous analysis.

## Components

### 1. MetricsCollector

Computes and aggregates evaluation metrics.

```python
from src.evaluation import MetricsCollector

collector = MetricsCollector(confidence_level=0.95)
```

#### Computing Episode Metrics

```python
# Trajectory is a list of step dictionaries
trajectory = [
    {"reward": -0.01, "done": False, "info": {"visited_nodes": 1}},
    {"reward": -0.01, "done": False, "info": {"visited_nodes": 2}},
    {"reward": 1.0, "done": True, "info": {"success": True, "visited_nodes": 3}},
]

# Compute metrics for single episode
episode_metrics = collector.evaluate_episode(trajectory, total_states=5)

print(f"Success: {episode_metrics.success}")
print(f"Steps: {episode_metrics.steps_to_completion}")
print(f"Coverage: {episode_metrics.visited_states / episode_metrics.total_states}")
```

#### Aggregating Metrics

```python
# Collect metrics from multiple episodes
episode_metrics_list = [ep1_metrics, ep2_metrics, ep3_metrics, ...]

# Aggregate with confidence intervals
aggregate = collector.aggregate(episode_metrics_list)

print(f"Success Rate: {aggregate.success_rate:.2%} ± {aggregate.success_rate_std:.2%}")
print(f"95% CI: [{aggregate.success_rate_ci[0]:.2%}, {aggregate.success_rate_ci[1]:.2%}]")
print(f"Mean Steps to Success: {aggregate.mean_steps_to_success:.2f}")
print(f"Failure Types: {aggregate.failure_types}")
```

### 2. Evaluator

Orchestrates evaluation across multiple episodes and environments.

```python
from src.evaluation import Evaluator

evaluator = Evaluator(verbose=True)
```

#### Evaluating on Single Environment

```python
from src.agents.dqn import DQNAgent
from src.environment.flow_environment import FlowEnvironment

# Load trained agent
agent = DQNAgent(...)
agent.load("checkpoints/best_model.pt")

# Create test environment
test_env = FlowEnvironment(test_graph, env_config)

# Run evaluation
results = evaluator.evaluate_single_env(
    agent=agent,
    env=test_env,
    num_episodes=100,
    random_seeds=[42, 43, 44, ...],  # Optional, auto-generated if None
    agent_name="DQN",
    environment_name="TestFlow",
    collect_trajectories=False  # Set True to store full trajectories
)

# Access results
print(f"Success Rate: {results.metrics.success_rate:.2%}")
print(f"Mean Coverage: {results.metrics.mean_coverage:.2%}")
print(f"Mean Reward: {results.metrics.mean_reward:.3f}")
```

#### Evaluating on Multiple Environments

```python
# Create multiple test environments
test_envs = [env1, env2, env3]

# Evaluate across all environments
results = evaluator.evaluate(
    agent=agent,
    test_envs=test_envs,
    num_episodes=300,  # Will be distributed across environments
    agent_name="PPO",
    environment_name="MultiEnv",
)

# Episodes are automatically distributed:
# - 100 episodes on env1
# - 100 episodes on env2
# - 100 episodes on env3
```

## Metrics Computed

### Episode-Level Metrics (EpisodeMetrics)

- `success`: Whether episode reached goal
- `steps_to_completion`: Number of steps taken
- `total_reward`: Cumulative reward
- `failure_type`: Type of failure (if any)
- `visited_states`: Number of unique states visited
- `total_states`: Total states in environment
- `trajectory_length`: Length of trajectory

### Aggregated Metrics (AggregateMetrics)

#### Success Metrics
- `success_rate`: Percentage of successful episodes
- `success_rate_std`: Standard deviation
- `success_rate_ci`: 95% confidence interval

#### Steps Metrics (for successful episodes only)
- `mean_steps_to_success`: Average steps to reach goal
- `std_steps_to_success`: Standard deviation
- `steps_ci`: 95% confidence interval

#### Failure Metrics
- `failure_rate`: Percentage of failed episodes
- `failure_types`: Dictionary mapping failure type to count

#### Coverage Metrics
- `mean_coverage`: Average percentage of states visited
- `std_coverage`: Standard deviation
- `coverage_ci`: 95% confidence interval

#### Reward Metrics
- `mean_reward`: Average cumulative reward
- `std_reward`: Standard deviation
- `reward_ci`: 95% confidence interval

#### Episode Length
- `mean_episode_length`: Average episode length
- `std_episode_length`: Standard deviation

## Complete Example

```python
from src.evaluation import Evaluator
from src.agents.ppo import PPOAgent
from src.environment.flow_environment import FlowEnvironment
from src.environment.graph_generator import GraphGenerator, GraphConfig

# Create test environments
generator = GraphGenerator(random_seed=42)
test_graphs = [
    generator.generate(GraphConfig(num_nodes=10, branching_factor=2.0, depth=5))
    for _ in range(5)
]

test_envs = [
    FlowEnvironment(graph, env_config)
    for graph in test_graphs
]

# Load trained agent
agent = PPOAgent(...)
agent.load("checkpoints/ppo_best.pt")

# Create evaluator
evaluator = Evaluator(verbose=True)

# Run comprehensive evaluation
results = evaluator.evaluate(
    agent=agent,
    test_envs=test_envs,
    num_episodes=500,  # 100 episodes per environment
    random_seeds=list(range(500)),  # Reproducible seeds
    agent_name="PPO",
    environment_name="TestSet",
    collect_trajectories=False
)

# Print summary (automatically printed if verbose=True)
# Access detailed results
print(f"\n=== Detailed Results ===")
print(f"Total Episodes: {results.num_episodes}")
print(f"Success Rate: {results.metrics.success_rate:.2%}")
print(f"  95% CI: [{results.metrics.success_rate_ci[0]:.2%}, "
      f"{results.metrics.success_rate_ci[1]:.2%}]")
print(f"Mean Steps to Success: {results.metrics.mean_steps_to_success:.2f}")
print(f"Mean Coverage: {results.metrics.mean_coverage:.2%}")
print(f"Failure Breakdown:")
for failure_type, count in results.metrics.failure_types.items():
    print(f"  {failure_type}: {count} ({count/results.num_episodes*100:.1f}%)")

# Access per-episode data
for i, ep_metrics in enumerate(results.episode_metrics[:5]):
    print(f"\nEpisode {i+1}:")
    print(f"  Success: {ep_metrics.success}")
    print(f"  Steps: {ep_metrics.steps_to_completion}")
    print(f"  Reward: {ep_metrics.total_reward:.3f}")
```

## Integration with Training

The evaluator can be used during training for periodic validation:

```python
from src.training.orchestrator import TrainingOrchestrator
from src.evaluation import Evaluator

# Create training orchestrator
orchestrator = TrainingOrchestrator(...)

# Create evaluator for validation
evaluator = Evaluator(verbose=False)

# During training loop
for episode in range(num_episodes):
    # Train agent
    orchestrator.train_episode(agent, env)
    
    # Periodic evaluation
    if episode % eval_frequency == 0:
        val_results = evaluator.evaluate_single_env(
            agent=agent,
            env=validation_env,
            num_episodes=10,
            collect_trajectories=False
        )
        
        print(f"Episode {episode}: Val Success Rate = "
              f"{val_results.metrics.success_rate:.2%}")
```

## Tips and Best Practices

1. **Random Seeds**: Always provide explicit random seeds for reproducibility
2. **Memory Management**: Set `collect_trajectories=False` for large evaluations
3. **Confidence Intervals**: Use at least 30 episodes for reliable confidence intervals
4. **Multiple Environments**: Distribute episodes across diverse test environments
5. **Verbose Mode**: Use `verbose=True` during development, `False` in automated scripts
6. **Failure Analysis**: Check `failure_types` to understand agent weaknesses

## Statistical Notes

- Confidence intervals use the t-distribution (appropriate for small samples)
- Standard error of the mean (SEM) is computed using scipy.stats.sem
- Confidence level is configurable (default: 95%)
- Minimum 2 samples required for confidence intervals (returns mean otherwise)
