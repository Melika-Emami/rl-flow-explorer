# Exploration Strategies

This document describes the exploration strategy framework and available implementations.

## Overview

Exploration strategies determine how agents balance exploration (trying new actions) and exploitation (using known good actions). The framework provides a pluggable interface for different exploration methods.

## Architecture

### Base Class: `ExplorationStrategy`

All exploration strategies inherit from the abstract base class `ExplorationStrategy`:

```python
from src.agents.exploration import ExplorationStrategy

class ExplorationStrategy(ABC):
    def compute_bonus(self, state, action, next_state, info=None) -> float:
        """Compute intrinsic reward bonus for exploration."""
        pass
    
    def reset(self) -> None:
        """Reset strategy state at episode start."""
        pass
    
    def update(self, state, action, next_state) -> None:
        """Update strategy based on observed transition."""
        pass
```

### Key Methods

- **`compute_bonus()`**: Returns an intrinsic reward bonus to encourage exploration. This bonus is added to the extrinsic reward from the environment.

- **`reset()`**: Called at the start of each episode to reset any episode-specific state.

- **`update()`**: Called after each transition to update internal statistics or models.

## Available Strategies

### 1. Epsilon-Greedy

The classic epsilon-greedy exploration strategy.

**Usage:**

```python
from src.agents.exploration import EpsilonGreedy

# Create strategy
strategy = EpsilonGreedy(
    epsilon=1.0,           # Initial exploration rate
    epsilon_min=0.01,      # Minimum exploration rate
    epsilon_decay=0.995,   # Decay factor
    decay_mode="episode"   # Decay per episode or per step
)

# Check if should explore
if strategy.should_explore():
    action = random_action()
else:
    action = greedy_action()

# Decay epsilon (call after each episode if decay_mode="episode")
strategy.decay()
```

**Parameters:**

- `epsilon` (float): Initial exploration probability [0, 1]
- `epsilon_min` (float): Minimum exploration probability [0, 1]
- `epsilon_decay` (float): Multiplicative decay factor (0, 1]
- `decay_mode` (str): When to decay - "episode" or "step"

**Decay Modes:**

- **"episode"**: Epsilon decays only when `decay()` is called manually (typically after each episode)
- **"step"**: Epsilon decays automatically on each `update()` call (after each step)

**Methods:**

- `should_explore()`: Returns True with probability epsilon
- `decay()`: Manually decay epsilon
- `get_epsilon()`: Get current epsilon value
- `set_epsilon(value)`: Set epsilon to specific value
- `reset_epsilon()`: Reset epsilon to initial value

**Note:** Epsilon-greedy returns 0.0 for `compute_bonus()` since exploration is handled through action selection rather than intrinsic rewards.

## Integration with Agents

### Pattern 1: Using Existing Agent Epsilon

Most agents (Q-learning, DQN) already have built-in epsilon-greedy exploration:

```python
from src.agents import DQNAgent

agent = DQNAgent(
    observation_dim=10,
    action_space_size=4,
    epsilon=1.0,
    epsilon_decay=0.995
)

# Agent handles exploration internally
action = agent.select_action(observation, training=True)

# Decay after episode
agent.decay_epsilon()
```

### Pattern 2: Using Exploration Strategy for Intrinsic Rewards

For future intrinsic reward methods (count-based, curiosity):

```python
from src.agents.exploration import EpsilonGreedy
from src.utils.data_models import Transition

# Create exploration strategy
exploration = EpsilonGreedy(epsilon=0.1)

# During training
state = env.reset()
action = agent.select_action(state)
next_state, reward, done, info = env.step(action)

# Compute intrinsic reward bonus
intrinsic_reward = exploration.compute_bonus(state, action, next_state)

# Combine rewards
total_reward = reward + intrinsic_reward

# Update agent with total reward
transition = Transition(state, action, total_reward, next_state, done, info)
agent.update(transition)

# Update exploration strategy
exploration.update(state, action, next_state)
```

### Pattern 3: Separate Exploration Control

Use exploration strategy independently of agent:

```python
from src.agents.exploration import EpsilonGreedy
import numpy as np

exploration = EpsilonGreedy(epsilon=0.2)

# Action selection with exploration
if exploration.should_explore():
    action = np.random.randint(0, num_actions)
else:
    action = agent.select_action(state, training=False)  # Greedy
```

### 2. Count-Based Bonus

Provides intrinsic rewards inversely proportional to state visit counts, encouraging exploration of novel or rarely-visited states.

**Usage:**

```python
from src.agents.exploration import CountBasedBonus

# Create strategy
strategy = CountBasedBonus(
    beta=1.0,              # Scaling factor for bonus
    count_mode="state"     # Count states or state-action pairs
)

# Compute exploration bonus
bonus = strategy.compute_bonus(state, action, next_state)

# Update visit counts
strategy.update(state, action, next_state)

# Get statistics
stats = strategy.get_statistics()
print(f"Mean bonus: {stats['mean_bonus']}")
print(f"States visited: {stats['num_states_visited']}")
```

**Parameters:**

- `beta` (float): Scaling factor for intrinsic reward bonus (≥ 0)
- `state_hash_fn` (callable, optional): Function to hash states for counting. Default uses `tuple(state.flatten())`
- `count_mode` (str): What to count - "state" or "state_action"

**Count Modes:**

- **"state"**: Counts visits to states. Encourages visiting new states regardless of action taken.
- **"state_action"**: Counts visits to state-action pairs. Encourages trying different actions in the same state.

**Bonus Computation:**

The bonus is computed as:
```
bonus = beta / sqrt(count + 1)
```

Where `count` is the number of times the state (or state-action pair) has been visited.

**Methods:**

- `compute_bonus(state, action, next_state, info=None)`: Returns intrinsic reward bonus
- `update(state, action, next_state)`: Updates visit counts
- `reset_counts()`: Clears all visit counts and statistics
- `get_statistics()`: Returns dictionary with exploration statistics
- `get_state_count(state)`: Returns visit count for a specific state
- `get_state_action_count(state, action)`: Returns visit count for a state-action pair

**Statistics Tracked:**

- `mean_bonus`: Average intrinsic reward across all transitions
- `max_bonus`: Maximum intrinsic reward observed
- `min_bonus`: Minimum intrinsic reward observed
- `num_states_visited`: Number of unique states visited
- `total_bonuses`: Sum of all bonuses provided

**Example with Agent:**

```python
from src.agents.exploration import CountBasedBonus
from src.agents import QLearningAgent
from src.utils.data_models import Transition

# Create agent and exploration strategy
agent = QLearningAgent(action_space_size=4, epsilon=0.1)
exploration = CountBasedBonus(beta=0.5, count_mode="state")

# Training loop
state = env.reset()
action = agent.select_action(state, training=True)
next_state, extrinsic_reward, done, info = env.step(action)

# Compute intrinsic reward
intrinsic_reward = exploration.compute_bonus(state, action, next_state)
total_reward = extrinsic_reward + intrinsic_reward

# Update agent with augmented reward
transition = Transition(state, action, total_reward, next_state, done, info)
agent.update(transition)

# Update exploration counts
exploration.update(state, action, next_state)

# Log statistics periodically
if episode % 100 == 0:
    stats = exploration.get_statistics()
    print(f"Exploration stats: {stats}")
```

### 3. Curiosity-Based Bonus

Provides intrinsic rewards based on prediction error from a forward dynamics model. The model learns to predict next states, and transitions that are difficult to predict receive higher bonuses, encouraging exploration of surprising or novel states.

**Usage:**

```python
from src.agents.exploration import CuriosityBonus

# Create strategy
strategy = CuriosityBonus(
    state_dim=10,              # Dimension of state observations
    action_dim=4,              # Number of possible actions
    beta=0.1,                  # Scaling factor for bonus
    learning_rate=1e-3,        # Learning rate for forward model
    hidden_dims=(128, 128),    # Hidden layer sizes
    batch_size=32,             # Batch size for training
    train_frequency=1,         # Train every N transitions
    device="cpu"               # Device for PyTorch model
)

# Compute exploration bonus (prediction error)
bonus = strategy.compute_bonus(state, action, next_state)

# Update forward model
strategy.update(state, action, next_state)

# Get statistics
stats = strategy.get_statistics()
print(f"Mean bonus: {stats['mean_bonus']}")
print(f"Forward model loss: {stats['mean_forward_loss']}")
```

**Parameters:**

- `state_dim` (int): Dimension of state observation vectors
- `action_dim` (int): Number of possible discrete actions
- `beta` (float): Scaling factor for intrinsic reward bonus (≥ 0)
- `learning_rate` (float): Learning rate for training forward model (> 0)
- `hidden_dims` (tuple): Tuple of hidden layer dimensions for forward model
- `batch_size` (int): Batch size for training forward model (> 0)
- `train_frequency` (int): Train forward model every N transitions (> 0)
- `device` (str): Device to run model on ("cpu" or "cuda")

**How It Works:**

1. **Forward Dynamics Model**: A neural network that predicts the next state given the current state and action:
   ```
   predicted_next_state = forward_model(state, action)
   ```

2. **Prediction Error**: The mean squared error between predicted and actual next state:
   ```
   prediction_error = MSE(predicted_next_state, actual_next_state)
   ```

3. **Intrinsic Reward**: The prediction error scaled by beta:
   ```
   bonus = beta * prediction_error
   ```

4. **Model Training**: The forward model is trained continuously on observed transitions to improve predictions.

**Bonus Computation:**

The bonus is computed as:
```
bonus = beta * MSE(forward_model(state, action), next_state)
```

Higher prediction errors indicate surprising or novel transitions, which receive higher bonuses.

**Methods:**

- `compute_bonus(state, action, next_state, info=None)`: Returns intrinsic reward based on prediction error
- `update(state, action, next_state)`: Stores transition and trains forward model
- `reset_model()`: Resets forward model and experience buffer
- `get_statistics()`: Returns dictionary with exploration and training statistics
- `save(path)`: Saves forward model checkpoint
- `load(path)`: Loads forward model checkpoint

**Statistics Tracked:**

- `mean_bonus`: Average intrinsic reward across all transitions
- `max_bonus`: Maximum intrinsic reward observed
- `min_bonus`: Minimum intrinsic reward observed
- `total_bonuses`: Sum of all bonuses provided
- `mean_forward_loss`: Average training loss of forward model
- `num_forward_updates`: Number of forward model training updates
- `buffer_size`: Current size of experience buffer

**Example with Agent:**

```python
from src.agents.exploration import CuriosityBonus
from src.agents import DQNAgent
from src.utils.data_models import Transition

# Create agent and exploration strategy
agent = DQNAgent(observation_dim=10, action_space_size=4, epsilon=0.1)
exploration = CuriosityBonus(
    state_dim=10,
    action_dim=4,
    beta=0.1,
    batch_size=32,
    device="cpu"
)

# Training loop
state = env.reset()
action = agent.select_action(state, training=True)
next_state, extrinsic_reward, done, info = env.step(action)

# Compute intrinsic reward (prediction error)
intrinsic_reward = exploration.compute_bonus(state, action, next_state)
total_reward = extrinsic_reward + intrinsic_reward

# Update agent with augmented reward
transition = Transition(state, action, total_reward, next_state, done, info)
agent.update(transition)

# Update forward model
exploration.update(state, action, next_state)

# Log statistics periodically
if episode % 100 == 0:
    stats = exploration.get_statistics()
    print(f"Curiosity stats: {stats}")
```

**Advantages:**

- Works well in high-dimensional continuous state spaces
- Automatically adapts to environment dynamics
- Encourages exploration of unpredictable or surprising states
- No need for manual state discretization

**Considerations:**

- Requires more computation than count-based methods (neural network training)
- May struggle in stochastic environments (high prediction error even for visited states)
- Forward model quality affects exploration effectiveness
- Requires tuning of beta and learning rate parameters

**Comparison with Count-Based:**

| Aspect | Count-Based | Curiosity-Based |
|--------|-------------|-----------------|
| State Space | Best for discrete/tabular | Best for continuous/high-dim |
| Computation | Very fast (hash table lookup) | Slower (neural network) |
| Memory | Grows with unique states | Fixed (model parameters) |
| Stochasticity | Handles well | May struggle |
| Adaptability | Fixed formula | Learns dynamics |

## Best Practices

1. **Decay Schedule**: Start with high epsilon (0.9-1.0) and decay to low values (0.01-0.1)

2. **Decay Mode Selection**:
   - Use "episode" mode for episodic tasks with clear episode boundaries
   - Use "step" mode for continuous tasks or very long episodes

3. **Epsilon Range**: Keep epsilon_min > 0 to maintain some exploration throughout training

4. **Combining Strategies**: Different strategies can be combined by summing their bonuses:
   ```python
   total_bonus = epsilon_greedy.compute_bonus(...) + count_based.compute_bonus(...)
   ```

5. **Evaluation**: Set epsilon=0 or use `training=False` during evaluation to disable exploration

## Testing

Run the test suite to verify exploration strategies:

```bash
python test_exploration.py
```

Run integration examples:

```bash
python example_exploration_integration.py
```

## References

- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Mnih et al. (2015): Human-level control through deep reinforcement learning (DQN)
- Pathak et al. (2017): Curiosity-driven Exploration (ICM)
- Bellemare et al. (2016): Unifying Count-Based Exploration
