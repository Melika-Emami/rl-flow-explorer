# Experiment Scripts Quick Start Guide

## Quick Commands

### Train an Agent
```bash
# Using config file (recommended)
python train.py --agent dqn --config configs/dqn_baseline.json

# Using command-line arguments
python train.py --agent ppo --num-episodes 1000 --seed 42
```

### Run Multi-Seed Experiment
```bash
# Sequential execution
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name my_experiment \
  --num-seeds 5

# Parallel execution (faster)
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name my_experiment \
  --num-seeds 5 \
  --parallel \
  --max-workers 4
```

### Evaluate Agent
```bash
# In-distribution evaluation
python evaluate.py \
  --agent dqn \
  --checkpoint logs/my_experiment_seed42/checkpoints/best_model.pt \
  --num-episodes 100

# Out-of-distribution evaluation
python evaluate.py \
  --agent dqn \
  --checkpoint logs/my_experiment_seed42/checkpoints/best_model.pt \
  --num-episodes 100 \
  --eval-ood \
  --ood-shift deeper
```

### Generate Plots
```bash
# Learning curves
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_seed42 logs/ppo_seed42 \
  --agent-names DQN PPO \
  --metric reward

# Algorithm comparison
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results eval_results/dqn.json eval_results/ppo.json \
  --eval-names DQN PPO \
  --metric success_rate
```

## Available Agents

- `qlearning` - Tabular Q-learning (for small environments)
- `dqn` - Deep Q-Network
- `ppo` - Proximal Policy Optimization
- `recurrent_ppo` - PPO with LSTM (for partial observability)

## Available Configurations

Pre-defined configs in `configs/`:
- `dqn_baseline.json` - Standard DQN
- `ppo_baseline.json` - Standard PPO
- `recurrent_ppo_baseline.json` - Recurrent PPO
- `qlearning_small.json` - Q-learning on small graphs
- `exploration_comparison.json` - Exploration strategy ablation

## Common Workflows

### 1. Train and Evaluate Single Agent
```bash
# Train
python train.py --agent dqn --config configs/dqn_baseline.json

# Evaluate
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100

# Plot
python plot_results.py \
  --plot-type learning_curves \
  --results-dirs logs/dqn_baseline \
  --agent-names DQN \
  --metric reward
```

### 2. Compare Multiple Algorithms
```bash
# Train each algorithm
for agent in dqn ppo recurrent_ppo; do
  python train.py --agent $agent --config configs/${agent}_baseline.json
done

# Evaluate each
for agent in dqn ppo recurrent_ppo; do
  python evaluate.py \
    --agent $agent \
    --checkpoint logs/${agent}_baseline/checkpoints/best_model.pt \
    --num-episodes 100 \
    --output-dir evaluation_results/$agent
done

# Compare
python plot_results.py \
  --plot-type algorithm_comparison \
  --eval-results evaluation_results/dqn/in_distribution_results.json \
                 evaluation_results/ppo/in_distribution_results.json \
                 evaluation_results/recurrent_ppo/in_distribution_results.json \
  --eval-names DQN PPO "Recurrent PPO" \
  --metric success_rate
```

### 3. Multi-Seed Experiment with Statistics
```bash
# Run experiment across 5 seeds in parallel
python run_experiments.py \
  --agent dqn \
  --config configs/dqn_baseline.json \
  --experiment-name dqn_multiseed \
  --num-seeds 5 \
  --parallel

# Results automatically aggregated in:
# experiment_results/dqn_multiseed_aggregated.json
```

### 4. Generalization Analysis
```bash
# Train agent
python train.py --agent dqn --config configs/dqn_baseline.json

# Evaluate in-distribution
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100 \
  --output-dir evaluation_results/dqn

# Evaluate out-of-distribution
python evaluate.py \
  --agent dqn \
  --checkpoint logs/dqn_baseline/checkpoints/best_model.pt \
  --num-episodes 100 \
  --eval-ood \
  --ood-shift deeper \
  --output-dir evaluation_results/dqn

# Plot generalization
python plot_results.py \
  --plot-type generalization \
  --in-dist-results evaluation_results/dqn/in_distribution_results.json \
  --ood-results evaluation_results/dqn/ood_deeper_results.json \
  --eval-names DQN \
  --output-dir plots
```

## Monitoring Training

Use TensorBoard to monitor training in real-time:
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Output Locations

- **Training logs**: `logs/<experiment_name>/`
- **Checkpoints**: `logs/<experiment_name>/checkpoints/`
- **TensorBoard logs**: `logs/<experiment_name>/tensorboard/`
- **Training metrics**: `logs/<experiment_name>/training_metrics.json`
- **Evaluation results**: `evaluation_results/<experiment_name>/`
- **Plots**: `plots/`
- **Aggregated results**: `experiment_results/`

## Tips

1. **Always use seeds**: Specify `--seed` for reproducibility
2. **Use config files**: Easier to track and share experiments
3. **Run multiple seeds**: Use `run_experiments.py` for statistical significance
4. **Monitor with TensorBoard**: Real-time training visualization
5. **Save checkpoints**: Use `--checkpoint-frequency` to save intermediate models
6. **Evaluate OOD**: Always test generalization with `--eval-ood`

## Getting Help

For detailed documentation, see:
- `docs/experiment_scripts_usage.md` - Complete usage guide
- `python train.py --help` - Training script help
- `python evaluate.py --help` - Evaluation script help
- `python plot_results.py --help` - Plotting script help
- `python run_experiments.py --help` - Multi-seed runner help

## Example Script

Run the complete example workflow:
```bash
bash example_experiment_workflow.sh
```

This demonstrates training, evaluation, and visualization in one script.
