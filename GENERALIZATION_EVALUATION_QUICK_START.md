# Generalization Evaluation Quick Start

This guide explains how to evaluate trained agents on unseen graphs with distribution shifts.

## Overview

Evaluates how well trained agents generalize to:
- **In-distribution test graphs**: Similar to training but unseen
- **Out-of-distribution test graphs**: Distribution shifts including:
  - Deeper graphs (1.5x depth)
  - Wider graphs (1.5x branching factor)
  - Larger graphs (1.5x size)
  - Linear topology (minimal branching)
  - Wide topology (high branching, shallow depth)

## Prerequisites

You must have trained models from:
- Baseline experiments (Q-learning, DQN, PPO, Recurrent PPO)
- Exploration ablations (optional)
- Memory ablations (optional)

## Quick Start

### 1. Run Generalization Evaluation on All Trained Agents

```bash
python run_generalization_evaluation.py
```

This will:
- Search for trained models in `baseline_logs/`, `memory_logs/`, and `test_logs/`
- Create test splits with 10 in-distribution and 5 per shift graphs
- Evaluate each agent on all test splits (10 episodes per environment)
- Save results to `generalization_results/`
- Generate plots in `generalization_plots/`

**Expected Runtime:** ~10-30 minutes depending on number of agents

### 2. Evaluate Specific Agents

```bash
# Evaluate only DQN and PPO
python run_generalization_evaluation.py --agents dqn ppo

# Evaluate only Recurrent PPO
python run_generalization_evaluation.py --agents recurrent_ppo
```

### 3. Customize Test Configuration

```bash
# More test graphs for better statistics
python run_generalization_evaluation.py \
    --num-test-in-dist 20 \
    --num-test-per-shift 10 \
    --episodes-per-env 20

# Faster evaluation with fewer graphs
python run_generalization_evaluation.py \
    --num-test-in-dist 5 \
    --num-test-per-shift 3 \
    --episodes-per-env 5
```

### 4. Specify Model Directories

```bash
# Search in custom directories
python run_generalization_evaluation.py \
    --model-dirs baseline_logs memory_logs my_custom_logs
```

## Output Structure

After running, you'll have:

```
generalization_results/
├── dqn_generalization.json
├── ppo_generalization.json
├── recurrent_ppo_generalization.json
└── qlearning_generalization.json

generalization_plots/
├── generalization_comparison.png
└── degradation_heatmap.png
```

## Understanding Results

### Results JSON Structure

Each agent's results file contains:

```json
{
  "agent_name": "dqn",
  "in_distribution": {
    "success_rate": 0.75,
    "mean_steps": 28.5,
    "mean_reward": 0.65,
    "mean_coverage": 0.82,
    "failure_rate": 0.15
  },
  "out_of_distribution": {
    "test_deeper": {
      "success_rate": 0.60,
      "mean_steps": 35.2,
      ...
    },
    "test_wider": { ... },
    ...
  },
  "degradation_metrics": {
    "test_deeper": {
      "success_rate_drop": 0.15,
      "success_rate_drop_pct": 20.0,
      "steps_increase": 6.7,
      "steps_increase_pct": 23.5,
      ...
    },
    ...
  }
}
```

### Interpreting Degradation Metrics

**Excellent Generalization:**
- Success rate drop < 10%
- Minimal increase in steps
- Maintained coverage

**Good Generalization:**
- Success rate drop 10-25%
- Moderate increase in steps
- Some coverage loss

**Poor Generalization:**
- Success rate drop > 25%
- Significant increase in steps or failures
- Large coverage loss

### Distribution Shift Insights

- **test_deeper**: Tests ability to handle longer sequences
- **test_wider**: Tests ability to explore more branches
- **test_larger**: Tests scalability to bigger state spaces
- **test_linear**: Tests performance on simpler structures
- **test_wide_topology**: Tests handling of high branching factor

## Visualizations

### 1. Generalization Comparison Plot

Shows success rates for each agent across all distribution shifts.

- Grouped bars for easy comparison
- In-distribution baseline vs. each shift
- Identifies which agents generalize best

### 2. Degradation Heatmap

Shows percentage drop in success rate as a heatmap.

- Rows: Agents
- Columns: Distribution shifts
- Color: Red (high degradation) to Green (low degradation)
- Values: Percentage drop in success rate

## Example Output

```
================================================================================
GENERALIZATION EVALUATION SUMMARY
================================================================================

DQN
--------------------------------------------------------------------------------

In-Distribution Performance:
  Success Rate: 72.00%
  Mean Steps:   28.50
  Coverage:     81.50%
  Mean Reward:  0.645

Out-of-Distribution Performance:

  test_deeper:
    Success Rate: 58.00%
    Degradation:  -14.00% (-19.4%)
    Steps:        35.20 (+23.5%)

  test_wider:
    Success Rate: 62.00%
    Degradation:  -10.00% (-13.9%)
    Steps:        32.10 (+12.6%)

  test_larger:
    Success Rate: 55.00%
    Degradation:  -17.00% (-23.6%)
    Steps:        38.50 (+35.1%)

  test_linear:
    Success Rate: 78.00%
    Degradation:  +6.00% (+8.3%)
    Steps:        22.30 (-21.8%)

  test_wide_topology:
    Success Rate: 60.00%
    Degradation:  -12.00% (-16.7%)
    Steps:        33.80 (+18.6%)

  Average Degradation: -13.5%
  Assessment: ○ Good generalization
```

## Advanced Usage

### Custom Graph Configuration

```bash
# Test on larger graphs
python run_generalization_evaluation.py \
    --num-nodes 15 \
    --branching-factor 2.5 \
    --depth 7

# Test on smaller graphs
python run_generalization_evaluation.py \
    --num-nodes 8 \
    --branching-factor 1.5 \
    --depth 4
```

### Verbose Output

```bash
# Show detailed progress during evaluation
python run_generalization_evaluation.py --verbose
```

### Custom Output Directories

```bash
python run_generalization_evaluation.py \
    --output-dir my_results \
    --plot-dir my_plots
```

## Analyzing Results

### Compare Agent Generalization

```python
import json

# Load results
with open('generalization_results/dqn_generalization.json') as f:
    dqn_results = json.load(f)

with open('generalization_results/recurrent_ppo_generalization.json') as f:
    rppo_results = json.load(f)

# Compare average degradation
dqn_avg = sum(
    d['success_rate_drop_pct'] 
    for d in dqn_results['degradation_metrics'].values()
) / len(dqn_results['degradation_metrics'])

rppo_avg = sum(
    d['success_rate_drop_pct'] 
    for d in rppo_results['degradation_metrics'].values()
) / len(rppo_results['degradation_metrics'])

print(f"DQN Average Degradation: {dqn_avg:.1f}%")
print(f"Recurrent PPO Average Degradation: {rppo_avg:.1f}%")
```

### Identify Most Challenging Shift

```python
# Find worst shift for each agent
for agent_file in Path('generalization_results').glob('*.json'):
    with open(agent_file) as f:
        results = json.load(f)
    
    worst_shift = max(
        results['degradation_metrics'].items(),
        key=lambda x: x[1]['success_rate_drop_pct']
    )
    
    print(f"{results['agent_name']}: {worst_shift[0]} "
          f"({worst_shift[1]['success_rate_drop_pct']:.1f}%)")
```

## Troubleshooting

### No Trained Models Found

If you see "No trained models found":

1. Run baseline experiments first:
   ```bash
   python run_baseline_experiments.py
   ```

2. Check that models were saved:
   ```bash
   ls baseline_logs/*/checkpoints/final_model.pt
   ```

3. Specify correct model directories:
   ```bash
   python run_generalization_evaluation.py --model-dirs baseline_logs
   ```

### Agent Loading Errors

If agents fail to load:

1. Check checkpoint files exist
2. Verify agent type matches checkpoint
3. Check for version compatibility issues

### Slow Evaluation

If evaluation is too slow:

```bash
# Reduce test graphs and episodes
python run_generalization_evaluation.py \
    --num-test-in-dist 5 \
    --num-test-per-shift 3 \
    --episodes-per-env 5
```

### Memory Issues

If you encounter OOM errors:

1. Reduce number of test graphs
2. Evaluate agents one at a time:
   ```bash
   python run_generalization_evaluation.py --agents dqn
   python run_generalization_evaluation.py --agents ppo
   ```
