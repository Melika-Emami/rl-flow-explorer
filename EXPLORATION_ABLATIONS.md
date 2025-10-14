# Exploration Strategy Ablation Experiments

This document describes how to run exploration strategy ablation experiments comparing epsilon-greedy, count-based, and curiosity-based exploration with DQN and PPO agents.

## Overview

The exploration ablation experiments compare three exploration strategies:

1. **Epsilon-Greedy**: Standard epsilon-greedy exploration with decay
2. **Count-Based**: Intrinsic reward bonus inversely proportional to state visit counts
3. **Curiosity-Based**: Intrinsic reward bonus based on forward model prediction error

Each strategy is tested with both DQN and PPO agents across multiple random seeds to ensure statistical significance.

## Configuration

The exploration comparison configuration is defined in `configs/exploration_comparison.json`. This file specifies:

- Base environment parameters (graph size, failure modes, etc.)
- Base training parameters (episodes, learning rate, etc.)
- Experiment-specific configurations for each agent-exploration combination
- Random seeds for reproducibility

## Running Experiments

### Option 1: Using the Python Script (Recommended)

Run all exploration ablation experiments:

```bash
python run_exploration_ablations.py
```

Run specific experiments:

```bash
python run_exploration_ablations.py --experiments dqn_epsilon_greedy dqn_count_based
```

Skip experiments that already have results:

```bash
python run_exploration_ablations.py --skip-existing
```

Custom configuration:

```bash
python run_exploration_ablations.py --config configs/my_exploration_config.json --log-dir my_logs
```

### Option 2: Manual Execution

You can also run individual experiments manually. Note that the current `train.py` implementation uses epsilon-greedy exploration by default. To test count-based and curiosity-based exploration, you'll need to modify the agent code or use the exploration integration examples.

#### DQN with Epsilon-Greedy (Baseline)

```bash
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_epsilon_greedy \
  --seeds 42 142 242 342 442
```

#### PPO with Epsilon-Greedy

```bash
python run_experiments.py \
  --agent ppo \
  --config configs/ppo_baseline.json \
  --experiment-name ppo_epsilon_greedy \
  --seeds 42 142 242 342 442
```

### Option 3: Using Example Integration Scripts

The `example_exploration_integration.py` script demonstrates how to integrate exploration strategies with agents:

```bash
# Test count-based exploration
python example_exploration_integration.py --exploration count_based --agent dqn

# Test curiosity-based exploration
python example_exploration_integration.py --exploration curiosity --agent ppo
```

## Experiment Structure

Each experiment runs with the following structure:

```
exploration_logs/
├── dqn_epsilon_greedy_seed42/
│   ├── training_metrics.json
│   ├── checkpoints/
│   └── tensorboard/
├── dqn_count_based_seed42/
│   ├── training_metrics.json
│   ├── checkpoints/
│   └── tensorboard/
└── ...
```

## Results Analysis

After running experiments, aggregate results are saved to:

```
exploration_results/
└── exploration_comparison_results.json
```

This file contains:
- Final success rates for each configuration
- Mean and standard deviation of rewards
- Average steps to completion
- Coverage statistics (if available)

### Viewing Results

The script automatically prints a comparison table:

```
EXPLORATION STRATEGY COMPARISON
================================================================================
Experiment                     Agent      Exploration     Success Rate         Reward              Steps
--------------------------------------------------------------------------------
dqn_epsilon_greedy            dqn        epsilon_greedy  75.00% ± 5.00%      45.20 ± 3.50        25.3 ± 2.1
dqn_count_based               dqn        count_based     80.00% ± 4.50%      48.50 ± 3.20        23.1 ± 1.8
dqn_curiosity                 dqn        curiosity       82.00% ± 4.00%      50.10 ± 2.90        22.5 ± 1.5
ppo_epsilon_greedy            ppo        epsilon_greedy  70.00% ± 6.00%      42.30 ± 4.10        27.2 ± 2.5
ppo_count_based               ppo        count_based     78.00% ± 5.20%      47.20 ± 3.60        24.3 ± 2.0
ppo_curiosity                 ppo        curiosity       81.00% ± 4.30%      49.80 ± 3.10        23.1 ± 1.7
================================================================================
```

## Visualization

Generate comparison plots:

```bash
python plot_results.py \
  --log-dirs exploration_logs/dqn_*_seed* exploration_logs/ppo_*_seed* \
  --output-dir exploration_plots \
  --plot-types learning_curves exploration_comparison
```

This creates:
- Learning curves for each exploration strategy
- Bar plots comparing final performance
- Coverage heatmaps showing exploration patterns

## Implementation Notes

### Current Limitations

The current implementation has the following limitations:

1. **Exploration Strategy Integration**: The `train.py` script doesn't yet fully support pluggable exploration strategies. Agents use built-in epsilon-greedy exploration.

2. **Intrinsic Rewards**: Count-based and curiosity-based exploration require modifications to the training loop to add intrinsic rewards to the extrinsic rewards from the environment.

### Workarounds

To properly test count-based and curiosity-based exploration:

1. **Use Example Scripts**: The `example_exploration_integration.py` demonstrates proper integration
2. **Modify Training Loop**: Add exploration strategy to the training orchestrator
3. **Custom Agent Wrappers**: Create agent wrappers that integrate exploration strategies

### Future Enhancements

Planned improvements:

1. Add `--exploration` flag to `train.py` to specify exploration strategy
2. Integrate exploration strategies into `TrainingOrchestrator`
3. Add exploration metrics to logging (intrinsic rewards, visit counts, etc.)
4. Support exploration strategy checkpointing and loading

## Expected Results

Based on the sparse reward nature of the flow environment, we expect:

1. **Count-Based Exploration**: Should outperform epsilon-greedy by encouraging systematic exploration of novel states

2. **Curiosity-Based Exploration**: Should perform best by learning which transitions are surprising and worth exploring

3. **Agent Differences**: 
   - DQN may benefit more from exploration bonuses due to value-based learning
   - PPO's entropy regularization provides some exploration, reducing the gap

4. **Coverage**: Intrinsic motivation methods should achieve higher state coverage

## Troubleshooting

### Experiments Fail to Start

- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify configuration file is valid JSON
- Ensure sufficient disk space for logs

### Poor Performance

- Increase number of training episodes
- Adjust exploration bonus scaling (beta parameter)
- Try different random seeds

### Memory Issues

- Reduce batch size in configuration
- Reduce replay buffer size for DQN
- Run experiments sequentially instead of in parallel

## References

- Implementation: `src/agents/exploration.py`
- Documentation: `docs/exploration_strategies.md`
