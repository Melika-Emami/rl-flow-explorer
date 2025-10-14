# Quick Start: Running Baseline Experiments

## TL;DR

```bash
# Verify everything is set up correctly
python test_baseline_setup.py

# Run all baseline experiments (recommended: use parallel execution)
python run_baseline_experiments.py --parallel --max-workers 4
```

That's it! The script will train all 4 agents with 5 seeds each (~2.5-3 hours with parallel execution).

## What Gets Trained

1. **Q-learning** on small graphs (5 nodes) - 5 seeds
2. **DQN** on medium graphs (10 nodes) - 5 seeds  
3. **PPO** on medium graphs (10 nodes) - 5 seeds
4. **Recurrent PPO** on medium graphs (10 nodes) - 5 seeds

**Total**: 20 training runs

## Expected Time

- **Parallel (4 workers)**: 2.5-3 hours
- **Sequential**: 9-10 hours

## Monitoring Progress

### Real-time with TensorBoard

While training is running:
```bash
tensorboard --logdir baseline_logs
```
Then open http://localhost:6006

### Check Completed Experiments

```bash
# List completed experiments
ls baseline_results/

# View aggregated results for DQN
cat baseline_results/baseline_dqn_medium_aggregated.json
```

## Output

Results will be saved to:
- `baseline_results/` - Aggregated statistics across seeds
- `baseline_logs/` - Individual training logs, checkpoints, and TensorBoard data

## Troubleshooting

### Out of Memory
```bash
# Reduce parallel workers
python run_baseline_experiments.py --parallel --max-workers 2

# Or run sequentially
python run_baseline_experiments.py
```

### Run Specific Agents Only
```bash
# Train only DQN and PPO
python run_baseline_experiments.py --agents dqn ppo --parallel
```

### Run with More Seeds
```bash
# For better statistical confidence
python run_baseline_experiments.py --num-seeds 10 --parallel
```

## After Training

Once training completes, you can:

1. **Analyze results**:
   ```bash
   python plot_results.py --log-dirs baseline_logs/baseline_*_seed* --output-dir baseline_plots
   ```

2. **Compare agents**:
   ```bash
   python plot_results.py --log-dirs baseline_logs/baseline_*_seed* --plot-types algorithm_comparison
   ```

## Need Help?

See `BASELINE_EXPERIMENTS.md` for detailed documentation.
