# Memory Ablations Quick Start Guide

This guide provides quick commands to run memory ablation experiments comparing PPO vs Recurrent PPO.

## What This Tests

Compares **PPO** (no memory) vs **Recurrent PPO** (LSTM memory) on environments with varying partial observability to isolate the contribution of memory.

## Quick Commands

### Run All Memory Ablations

```bash
python run_memory_ablations.py
```

This runs:
- 6 experiments (3 observability levels × 2 agent types)
- 5 seeds per experiment (30 total runs)
- ~3-4 hours total runtime

### Run Specific Observability Level

```bash
# High observability only (where memory matters most)
python run_memory_ablations.py --experiments ppo_high_observability recurrent_ppo_high_observability
```

### Skip Already Completed Experiments

```bash
python run_memory_ablations.py --skip-existing
```

## Experiment Matrix

| Observability | Popup Rate | PPO Experiment | Recurrent PPO Experiment |
|---------------|------------|----------------|--------------------------|
| Low | 10% | ppo_low_observability | recurrent_ppo_low_observability |
| Medium | 30% | ppo_medium_observability | recurrent_ppo_medium_observability |
| High | 50% | ppo_high_observability | recurrent_ppo_high_observability |

## Expected Results

- **Low observability**: Memory provides minimal benefit (~1-5% improvement)
- **Medium observability**: Memory provides moderate benefit (~10-15% improvement)
- **High observability**: Memory provides significant benefit (~20-40% improvement)

## Output

Results are saved to:
- **Logs**: `memory_logs/` (training metrics, checkpoints, TensorBoard)
- **Aggregated**: `memory_results/memory_ablation_results.json`

## Monitoring

```bash
# View progress with TensorBoard
tensorboard --logdir memory_logs

# Check latest results
cat memory_results/memory_ablation_results.json
```

## Visualization

```bash
python plot_results.py \
  --log-dirs memory_logs/ppo_*_seed* memory_logs/recurrent_ppo_*_seed* \
  --output-dir memory_plots \
  --plot-types learning_curves memory_comparison
```

## Troubleshooting

- **Out of memory**: Reduce batch size or LSTM hidden dimension in config
- **Slow training**: Run fewer seeds or reduce episodes
- **Poor results**: Check TensorBoard for training instability

## Full Documentation

See `MEMORY_ABLATIONS.md` for complete documentation.
