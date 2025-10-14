# Memory Ablation Experiments

This document describes how to run memory ablation experiments comparing PPO vs Recurrent PPO agents on environments with varying levels of partial observability to isolate the contribution of memory (LSTM) to performance.

## Overview

The memory ablation experiments compare two agent architectures:

1. **PPO (Standard)**: Policy-gradient agent with feedforward neural network (no memory)
2. **Recurrent PPO**: Policy-gradient agent with LSTM layer for maintaining memory across timesteps

Each architecture is tested on three levels of partial observability:
- **Low** (10% popup rate, 2 dependencies): Minimal partial observability
- **Medium** (30% popup rate, 3 dependencies): Moderate partial observability
- **High** (50% popup rate, 4 dependencies): Severe partial observability

This design allows us to measure how much memory (LSTM) helps as partial observability increases.

## Hypothesis

We expect that:
1. **Low observability**: PPO and Recurrent PPO should perform similarly, as memory is less critical
2. **Medium observability**: Recurrent PPO should start showing advantages
3. **High observability**: Recurrent PPO should significantly outperform PPO, as memory becomes crucial for tracking hidden state

## Configuration

The memory ablation configuration is defined in `configs/memory_ablation.json`. This file specifies:

- Base training parameters (episodes, learning rate, etc.)
- Three observability levels with different popup probabilities and dependencies
- Agent-specific configurations for PPO and Recurrent PPO
- Random seeds for reproducibility (5 seeds per configuration)

### Experiment Matrix

| Experiment | Agent | Popup Rate | Dependencies | Expected Memory Benefit |
|------------|-------|------------|--------------|------------------------|
| ppo_low_observability | PPO | 10% | 2 | Minimal |
| recurrent_ppo_low_observability | Recurrent PPO | 10% | 2 | Minimal |
| ppo_medium_observability | PPO | 30% | 3 | Moderate |
| recurrent_ppo_medium_observability | Recurrent PPO | 30% | 3 | Moderate |
| ppo_high_observability | PPO | 50% | 4 | Significant |
| recurrent_ppo_high_observability | Recurrent PPO | 50% | 4 | Significant |

## Running Experiments

### Quick Start

Run all memory ablation experiments:

```bash
python run_memory_ablations.py
```

This will:
1. Run 6 experiments (3 observability levels × 2 agent types)
2. Execute 5 seeds per experiment (30 total runs)
3. Aggregate results and compute memory contribution
4. Generate comparison tables and analysis
5. Save results to `memory_results/memory_ablation_results.json`

### Advanced Options

Run specific experiments:

```bash
python run_memory_ablations.py --experiments ppo_high_observability recurrent_ppo_high_observability
```

Skip experiments that already have results:

```bash
python run_memory_ablations.py --skip-existing
```

Custom configuration:

```bash
python run_memory_ablations.py \
  --config configs/my_memory_config.json \
  --log-dir my_memory_logs \
  --output-dir my_memory_results
```

### Manual Execution

You can also run individual experiments manually:

```bash
# PPO on high observability environment
python train.py --config configs/ppo_baseline.json --seed 42

# Recurrent PPO on high observability environment
python train.py --config configs/recurrent_ppo_baseline.json --seed 42
```

Note: You'll need to modify the config files to adjust popup probability and dependencies for different observability levels.

## Experiment Structure

Each experiment creates the following directory structure:

```
memory_logs/
├── ppo_low_observability_seed42/
│   ├── training_metrics.json
│   ├── ppo_low_observability_seed42_config.json
│   ├── ppo_low_observability_seed42_hyperparameters.json
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_episode_500.pt
│   └── tensorboard/
├── recurrent_ppo_low_observability_seed42/
│   └── ...
└── ...
```

## Results Analysis

After running experiments, results are saved to:

```
memory_results/
└── memory_ablation_results.json
```

This file contains:
- Aggregated metrics for each configuration (mean, std, min, max)
- Memory contribution analysis comparing PPO vs Recurrent PPO
- Improvement percentages for success rate, reward, and steps

### Output Format

The script prints two tables:

#### 1. Comparison Table

Shows performance of all configurations:

```
MEMORY ABLATION: PPO vs RECURRENT PPO
================================================================================
Experiment                          Agent           Popup%     Success Rate         Reward              Steps
--------------------------------------------------------------------------------
ppo_low_observability              ppo             10%        75.00% ± 5.00%      45.20 ± 3.50        25.3 ± 2.1
recurrent_ppo_low_observability    recurrent_ppo   10%        76.00% ± 4.80%      45.80 ± 3.40        25.0 ± 2.0
ppo_medium_observability           ppo             30%        65.00% ± 6.00%      38.50 ± 4.20        28.5 ± 2.8
recurrent_ppo_medium_observability recurrent_ppo   30%        72.00% ± 5.20%      43.20 ± 3.60        26.2 ± 2.3
ppo_high_observability             ppo             50%        50.00% ± 7.00%      28.30 ± 5.10        32.8 ± 3.5
recurrent_ppo_high_observability   recurrent_ppo   50%        68.00% ± 5.50%      40.50 ± 4.20        27.5 ± 2.7
================================================================================
```

#### 2. Memory Contribution Analysis

Shows improvement from adding LSTM memory:

```
MEMORY CONTRIBUTION ANALYSIS
================================================================================
Improvement from adding LSTM memory (Recurrent PPO vs PPO):

Observability        Popup%     Success Δ            Reward Δ             Steps Δ
--------------------------------------------------------------------------------
Low                  10%        +1.00% (+1.3%)       +0.60 (+1.3%)        -0.3 (-1.2%)
Medium               30%        +7.00% (+10.8%)      +4.70 (+12.2%)       -2.3 (-8.1%)
High                 50%        +18.00% (+36.0%)     +12.20 (+43.1%)      -5.3 (-16.2%)
================================================================================

INTERPRETATION:
--------------------------------------------------------------------------------

Low Partial Observability (10% popup rate):
  ~ Memory provides MODEST benefit (+1.3% success rate)

Medium Partial Observability (30% popup rate):
  ✓ Memory provides SIGNIFICANT benefit (+10.8% success rate)
  ✓ Reward improvement: +12.2%
  ✓ Efficiency gain: -8.1% fewer steps

High Partial Observability (50% popup rate):
  ✓ Memory provides SIGNIFICANT benefit (+36.0% success rate)
  ✓ Reward improvement: +43.1%
  ✓ Efficiency gain: -16.2% fewer steps
================================================================================
```

## Visualization

Generate comparison plots:

```bash
python plot_results.py \
  --log-dirs memory_logs/ppo_*_seed* memory_logs/recurrent_ppo_*_seed* \
  --output-dir memory_plots \
  --plot-types learning_curves memory_comparison
```

This creates:
- Learning curves comparing PPO vs Recurrent PPO at each observability level
- Bar plots showing memory contribution across observability levels
- Success rate degradation as observability decreases

## Expected Results

Based on the design, we expect:

### Low Observability (10% popup rate)
- **PPO**: ~70-80% success rate
- **Recurrent PPO**: ~72-82% success rate
- **Memory benefit**: Minimal (~1-5% improvement)
- **Interpretation**: Environment is mostly observable, memory not critical

### Medium Observability (30% popup rate)
- **PPO**: ~60-70% success rate
- **Recurrent PPO**: ~70-80% success rate
- **Memory benefit**: Moderate (~10-15% improvement)
- **Interpretation**: Partial observability starts to matter, memory helps track hidden state

### High Observability (50% popup rate)
- **PPO**: ~40-60% success rate
- **Recurrent PPO**: ~65-80% success rate
- **Memory benefit**: Significant (~20-40% improvement)
- **Interpretation**: Severe partial observability, memory crucial for success

## Runtime Estimates

- **Per experiment**: ~20-40 minutes (1000 episodes)
- **Per seed**: ~20-40 minutes
- **Total (6 experiments × 5 seeds)**: ~3-4 hours sequential
- **Parallel execution**: Can run multiple seeds in parallel to reduce wall-clock time

## Monitoring Progress

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir memory_logs
```

View metrics:
- Episode reward over time
- Success rate over time
- Episode length over time
- Loss values (policy loss, value loss)

### Log Files

Check training progress:

```bash
# View metrics for a specific experiment
cat memory_logs/ppo_high_observability_seed42/training_metrics.json

# Monitor latest experiment
tail -f memory_logs/*/training_metrics.json
```

## Troubleshooting

### Experiments Fail to Start

- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify configuration file is valid JSON: `python -m json.tool configs/memory_ablation.json`
- Ensure sufficient disk space for logs

### Poor Performance

- Increase number of training episodes (default: 1000)
- Adjust learning rate (default: 0.0003)
- Try different random seeds
- Check TensorBoard for training instability

### Memory Issues

- Reduce batch size in configuration (default: 64)
- Reduce LSTM hidden dimension (default: 128)
- Run experiments sequentially instead of in parallel

### Recurrent PPO Not Learning

- Check that hidden states are properly reset at episode boundaries
- Verify LSTM layer is receiving gradients
- Increase LSTM hidden dimension
- Adjust learning rate specifically for recurrent agent

## Implementation Notes

### Key Differences: PPO vs Recurrent PPO

**PPO (Feedforward)**:
- Network: MLP with 2 hidden layers (128 units each)
- Input: Current observation only
- No memory of past observations
- Faster training (fewer parameters)

**Recurrent PPO (LSTM)**:
- Network: MLP (1 hidden layer) + LSTM (128 units) + policy/value heads
- Input: Current observation + hidden state from previous timestep
- Maintains memory across episode
- Slower training (more parameters, sequential processing)

### Partial Observability Mechanisms

The environment creates partial observability through:

1. **Stochastic Popups**: Random pop-ups mask the true state with probability p
2. **Hidden Dependencies**: Earlier actions modify later transitions invisibly
3. **Limited Observation**: Agent only sees local features, not full graph structure

As popup probability and dependencies increase, the agent needs memory to:
- Track which states it has visited
- Remember which actions triggered dependencies
- Infer hidden state from observation history