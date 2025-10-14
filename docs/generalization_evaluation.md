# Generalization and Transfer Evaluation

This document describes how to use the generalization evaluation functionality to assess how well trained agents transfer to unseen environments.

## Overview

The generalization evaluation module provides tools for:
1. Creating train/test splits with distinct graphs
2. Generating distribution shift variants (deeper, wider, different topologies)
3. Running zero-shot evaluation protocols
4. Computing performance degradation metrics

## Creating Train/Test Splits

### Basic Train/Test Split

```python
from src.environment.graph_generator import GraphGenerator, GraphConfig

# Initialize generator
generator = GraphGenerator(random_seed=42)

# Define base configuration
config = GraphConfig(
    num_nodes=15,
    branching_factor=0.6,
    depth=7
)

# Generate train and test sets
train_graphs, test_graphs = generator.generate_train_test_split(
    config=config,
    num_train=50,
    num_test=20,
    train_seed_offset=0,
    test_seed_offset=10000  # Large offset ensures distinct graphs
)
```

### Distribution Shift Variants

Create test sets with specific distribution shifts:

```python
from src.environment.graph_generator import DistributionShiftConfig

# Deeper graphs (1.5x depth)
deeper_shift = DistributionShiftConfig(depth_multiplier=1.5)
deeper_graphs = generator.generate_distribution_shift(
    base_config=config,
    shift_config=deeper_shift,
    num_graphs=10,
    seed_offset=20000
)

# Wider graphs (1.5x branching)
wider_shift = DistributionShiftConfig(branching_multiplier=1.5)
wider_graphs = generator.generate_distribution_shift(
    base_config=config,
    shift_config=wider_shift,
    num_graphs=10,
    seed_offset=21000
)

# Larger graphs (1.5x size)
larger_shift = DistributionShiftConfig(size_multiplier=1.5)
larger_graphs = generator.generate_distribution_shift(
    base_config=config,
    shift_config=larger_shift,
    num_graphs=10,
    seed_offset=22000
)

# Linear topology (minimal branching)
linear_shift = DistributionShiftConfig(topology_type="linear")
linear_graphs = generator.generate_distribution_shift(
    base_config=config,
    shift_config=linear_shift,
    num_graphs=10,
    seed_offset=23000
)

# Wide topology (high branching, shallow depth)
wide_shift = DistributionShiftConfig(topology_type="wide")
wide_graphs = generator.generate_distribution_shift(
    base_config=config,
    shift_config=wide_shift,
    num_graphs=10,
    seed_offset=24000
)
```

### Standard Splits (Convenience Method)

Create a complete evaluation setup with one call:

```python
splits = generator.create_standard_splits(
    base_config=config,
    num_train=50,
    num_test_in_dist=20,
    num_test_per_shift=10
)

# Returns dictionary with keys:
# - 'train': Training graphs
# - 'test_in_dist': In-distribution test graphs
# - 'test_deeper': Deeper graphs
# - 'test_wider': Wider graphs
# - 'test_larger': Larger graphs
# - 'test_linear': Linear topology
# - 'test_wide_topology': Wide topology
```

## Zero-Shot Evaluation

### Basic Zero-Shot Evaluation

```python
from src.evaluation.generalization import GeneralizationEvaluator
from src.environment.flow_environment import EnvConfig

# Create evaluator
gen_evaluator = GeneralizationEvaluator(verbose=True)

# Define environment configuration
env_config = EnvConfig(
    step_penalty=-0.01,
    success_reward=1.0,
    failure_penalty=-1.0,
    max_steps=100
)

# Run zero-shot evaluation
results = gen_evaluator.evaluate_zero_shot(
    agent=trained_agent,
    graph_splits=splits,  # From create_standard_splits()
    env_config=env_config,
    num_episodes_per_env=10,
    agent_name="PPOAgent"
)

# Access results
print(f"In-Dist Success Rate: {results.in_dist_results.metrics.success_rate:.2%}")

for shift_name, out_results in results.out_dist_results.items():
    print(f"{shift_name} Success Rate: {out_results.metrics.success_rate:.2%}")
```

### Evaluating Specific Distribution Shift

```python
# Evaluate a specific shift
degradation = gen_evaluator.evaluate_distribution_shift(
    agent=trained_agent,
    in_dist_graphs=splits['test_in_dist'],
    out_dist_graphs=splits['test_deeper'],
    env_config=env_config,
    shift_name="deeper_graphs",
    num_episodes_per_env=10,
    agent_name="PPOAgent"
)

# Access degradation metrics
print(f"Success Rate Drop: {degradation.success_rate_drop:.2%}")
print(f"Steps Increase: {degradation.steps_increase:.2f}")
print(f"Coverage Drop: {degradation.coverage_drop:.2%}")
```

## Performance Degradation Metrics

The `PerformanceDegradation` dataclass contains:

- **Success Rate Degradation**:
  - `success_rate_in_dist`: In-distribution success rate
  - `success_rate_out_dist`: Out-of-distribution success rate
  - `success_rate_drop`: Absolute drop (in-dist - out-dist)
  - `success_rate_drop_pct`: Percentage drop

- **Steps to Success Degradation**:
  - `steps_in_dist`: Mean steps in-distribution
  - `steps_out_dist`: Mean steps out-of-distribution
  - `steps_increase`: Absolute increase
  - `steps_increase_pct`: Percentage increase

- **Coverage Degradation**:
  - `coverage_in_dist`: Mean coverage in-distribution
  - `coverage_out_dist`: Mean coverage out-of-distribution
  - `coverage_drop`: Absolute drop
  - `coverage_drop_pct`: Percentage drop

- **Reward Degradation**:
  - `reward_in_dist`: Mean reward in-distribution
  - `reward_out_dist`: Mean reward out-of-distribution
  - `reward_drop`: Absolute drop
  - `reward_drop_pct`: Percentage drop

## Complete Example

```python
from src.environment.graph_generator import GraphGenerator, GraphConfig
from src.environment.flow_environment import FlowEnvironment, EnvConfig
from src.agents.ppo import PPOAgent
from src.training.orchestrator import TrainingOrchestrator
from src.evaluation.generalization import GeneralizationEvaluator

# 1. Create graph splits
generator = GraphGenerator(random_seed=42)
base_config = GraphConfig(num_nodes=15, branching_factor=0.6, depth=7)

splits = generator.create_standard_splits(
    base_config=base_config,
    num_train=50,
    num_test_in_dist=20,
    num_test_per_shift=10
)

# 2. Train agent on training set
env_config = EnvConfig(
    step_penalty=-0.01,
    success_reward=1.0,
    failure_penalty=-1.0,
    max_steps=100
)

# Create training environments
train_envs = [FlowEnvironment(graph, env_config) for graph in splits['train']]

# Initialize agent
agent = PPOAgent(
    observation_dim=train_envs[0].observation_space.shape[0],
    action_dim=train_envs[0].action_space.n,
    hidden_dim=128,
    learning_rate=3e-4
)

# Train agent (simplified - use TrainingOrchestrator for full training)
# ... training code ...

# 3. Evaluate generalization
gen_evaluator = GeneralizationEvaluator(verbose=True)

results = gen_evaluator.evaluate_zero_shot(
    agent=agent,
    graph_splits=splits,
    env_config=env_config,
    num_episodes_per_env=10,
    agent_name="PPOAgent"
)

# 4. Analyze results
print("\n=== Generalization Results ===")
print(f"In-Distribution: {results.in_dist_results.metrics.success_rate:.2%}")

for shift_name, degradation in results.degradation_metrics.items():
    print(f"\n{shift_name}:")
    print(f"  Success Rate: {degradation.success_rate_out_dist:.2%}")
    print(f"  Degradation: {degradation.success_rate_drop_pct:+.1f}%")
```

## Interpreting Results

### Good Generalization
- Small success rate drop (< 10%)
- Minimal increase in steps to success
- Maintained coverage across shifts

### Poor Generalization
- Large success rate drop (> 30%)
- Significant increase in steps or failures
- Reduced coverage on shifted distributions

### Distribution Shift Insights

- **Deeper graphs**: Tests ability to handle longer sequences
- **Wider graphs**: Tests ability to explore more branches
- **Larger graphs**: Tests scalability to bigger state spaces
- **Linear topology**: Tests performance on simpler structures
- **Wide topology**: Tests handling of high branching factor

## Best Practices

1. **Use distinct seeds**: Ensure train and test sets use different seed ranges
2. **Multiple shifts**: Evaluate on multiple types of distribution shifts
3. **Sufficient episodes**: Run at least 10 episodes per environment for stable metrics
4. **Baseline comparison**: Compare against random agent to validate improvements
5. **Statistical significance**: Use confidence intervals to assess reliability

## Requirements Mapping

This implementation satisfies the following requirements:

- Train/test split functionality with distinct graphs
- Zero-shot evaluation without fine-tuning
- Multiple distribution shift scenarios
- Distinct test environments with structural similarity
- Performance degradation metrics
