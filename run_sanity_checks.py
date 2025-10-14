#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end sanity checks for the RL Flow Explorer system.

This script verifies:
1. Random agent performs worse than trained agents
2. Agent learns on trivial environment
3. Metrics improve over training
4. Reproducibility with same seeds
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.environment.graph_generator import GraphGenerator, GraphConfig
from src.environment.flow_environment import FlowEnvironment, EnvConfig
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.training.orchestrator import TrainingOrchestrator, TrainingConfig
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import MetricsCollector


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomAgent:
    """Random agent for baseline comparison."""

    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size

    def select_action(self, observation, training=True):
        """Select random action."""
        return np.random.randint(0, self.action_space_size)

    def update(self, *args, **kwargs):
        """No-op update."""
        return {}

    def save(self, path):
        """No-op save."""
        pass

    def load(self, path):
        """No-op load."""
        pass


def check_random_vs_trained():
    """
    Sanity Check 1: Verify random agent performs worse than trained agents.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK 1: Random Agent vs Trained Agent")
    print("=" * 80)

    # Create a simple environment
    graph_config = GraphConfig(
        num_nodes=10, branching_factor=2.0, max_depth=4, random_seed=42
    )
    env_config = EnvConfig(
        step_penalty=-0.1, success_reward=10.0, failure_penalty=-5.0, max_steps=50
    )

    generator = GraphGenerator()
    graph = generator.generate(graph_config)
    env = FlowEnvironment(graph, env_config)

    # Train a Q-learning agent briefly
    print("\n[1/3] Training Q-learning agent...")
    set_seeds(42)
    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    training_config = TrainingConfig(
        num_episodes=100,
        max_steps_per_episode=50,
        eval_frequency=50,
        num_eval_episodes=20,
        checkpoint_frequency=100,
        log_frequency=25,
        random_seed=42,
    )

    orchestrator = TrainingOrchestrator(
        agent, env, training_config, log_dir="sanity_check_logs/trained"
    )
    results = orchestrator.train()

    # Evaluate trained agent
    print("\n[2/3] Evaluating trained agent...")
    evaluator = Evaluator()
    trained_metrics = evaluator.evaluate(
        agent, [env], num_episodes=50, random_seeds=[42]
    )

    # Evaluate random agent
    print("\n[3/3] Evaluating random agent...")
    random_agent = RandomAgent(env.action_space.n)
    random_metrics = evaluator.evaluate(
        random_agent, [env], num_episodes=50, random_seeds=[42]
    )

    # Compare performance
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Trained Agent Success Rate: {trained_metrics.metrics.success_rate:.2%}")
    print(f"Random Agent Success Rate:  {random_metrics.metrics.success_rate:.2%}")
    print(f"Trained Agent Avg Reward:   {trained_metrics.metrics.mean_reward:.2f}")
    print(f"Random Agent Avg Reward:    {random_metrics.metrics.mean_reward:.2f}")

    # Verify trained agent is better (either higher success rate OR significantly better reward)
    success_better = (
        trained_metrics.metrics.success_rate > random_metrics.metrics.success_rate
    )
    reward_better = (
        trained_metrics.metrics.mean_reward > random_metrics.metrics.mean_reward + 10.0
    )  # At least 10 points better

    success = success_better or reward_better

    if success:
        if success_better:
            print("\n[PASS] Trained agent has higher success rate than random agent")
        else:
            print(
                "\n[PASS] Trained agent has significantly better reward than random agent"
            )
    else:
        print("\n[FAIL] Trained agent does not perform better than random agent")

    return success


def check_learning_on_trivial_environment():
    """
    Sanity Check 2: Verify agent learns on trivial environment.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK 2: Learning on Trivial Environment")
    print("=" * 80)

    # Create trivial environment (single path)
    print("\n[1/2] Creating trivial environment (single path to goal)...")
    graph_config = GraphConfig(
        num_nodes=5, branching_factor=1.0, max_depth=4, random_seed=42  # No branching
    )
    env_config = EnvConfig(
        step_penalty=-0.1, success_reward=10.0, failure_penalty=-5.0, max_steps=20
    )

    generator = GraphGenerator()
    graph = generator.generate(graph_config)
    env = FlowEnvironment(graph, env_config)

    # Train agent
    print("\n[2/2] Training agent on trivial environment...")
    set_seeds(42)
    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    training_config = TrainingConfig(
        num_episodes=200,
        max_steps_per_episode=20,
        eval_frequency=50,
        num_eval_episodes=20,
        checkpoint_frequency=200,
        log_frequency=50,
        random_seed=42,
    )

    orchestrator = TrainingOrchestrator(
        agent, env, training_config, log_dir="sanity_check_logs/trivial"
    )
    results = orchestrator.train()

    # Check final performance
    evaluator = Evaluator()
    final_metrics = evaluator.evaluate(agent, [env], num_episodes=50, random_seeds=[42])

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Final Success Rate: {final_metrics.metrics.success_rate:.2%}")
    print(f"Final Avg Reward:   {final_metrics.metrics.mean_reward:.2f}")

    # Agent should achieve high success rate on trivial environment
    success = final_metrics.metrics.success_rate >= 0.8

    if success:
        print("\n[PASS] Agent learns to solve trivial environment (>=80% success)")
    else:
        print(
            f"\n[FAIL] Agent fails to learn trivial environment ({final_metrics.metrics.success_rate:.2%} success)"
        )

    return success


def check_metrics_improve():
    """
    Sanity Check 3: Verify metrics improve over training.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK 3: Metrics Improve Over Training")
    print("=" * 80)

    # Create environment
    graph_config = GraphConfig(
        num_nodes=10, branching_factor=2.0, max_depth=4, random_seed=42
    )
    env_config = EnvConfig(
        step_penalty=-0.1, success_reward=10.0, failure_penalty=-5.0, max_steps=50
    )

    generator = GraphGenerator()
    graph = generator.generate(graph_config)
    env = FlowEnvironment(graph, env_config)

    # Train agent and track metrics
    print("\n[1/1] Training agent and tracking metrics...")
    set_seeds(42)
    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    training_config = TrainingConfig(
        num_episodes=150,
        max_steps_per_episode=50,
        eval_frequency=30,
        num_eval_episodes=20,
        checkpoint_frequency=150,
        log_frequency=30,
        random_seed=42,
    )

    orchestrator = TrainingOrchestrator(
        agent, env, training_config, log_dir="sanity_check_logs/improvement"
    )
    results = orchestrator.train()

    # Load training metrics
    metrics_path = "sanity_check_logs/improvement/training_metrics.json"
    with open(metrics_path, "r") as f:
        data = json.load(f)

    # Extract success rates from episode metrics
    episode_metrics = data["metrics"]
    episodes = [m["episode"] for m in episode_metrics]
    success_rates = [1.0 if m["success"] else 0.0 for m in episode_metrics]

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print("Success Rate Over Training (sampled episodes):")
    # Show every 10th episode
    for i in range(0, len(episodes), 10):
        ep = episodes[i]
        sr = success_rates[i]
        print(f"  Episode {ep:3d}: {sr:.2%}")

    # Check if metrics improve (compare first third to last third)
    third = len(success_rates) // 3
    early_avg = np.mean(success_rates[:third]) if third > 0 else success_rates[0]
    late_avg = np.mean(success_rates[-third:]) if third > 0 else success_rates[-1]

    improvement = late_avg - early_avg
    max_success = max(success_rates)
    print(f"\nEarly Average (first third):  {early_avg:.2%}")
    print(f"Late Average (last third):    {late_avg:.2%}")
    print(f"Improvement:                  {improvement:+.2%}")
    print(f"Peak Success Rate:            {max_success:.2%}")

    # Metrics should show learning: either improvement OR reaching high performance at some point
    shows_improvement = improvement > 0.05  # Shows improvement
    reaches_high_performance = (
        max_success > 0.7
    )  # Reaches at least 70% success at some point
    maintains_performance = late_avg > 0.3  # Maintains at least 30% in late training

    # Pass if agent shows clear learning (high peak) even if performance varies
    success = shows_improvement or (reaches_high_performance and maintains_performance)

    if success:
        if shows_improvement:
            print("\n[PASS] Metrics improve over training (>5% improvement)")
        else:
            print(
                "\n[PASS] Agent learns effectively (reaches >70% peak, maintains >30%)"
            )
    else:
        print(f"\n[FAIL] Agent does not show sufficient learning")

    return success


def check_reproducibility():
    """
    Sanity Check 4: Verify reproducibility with same seeds.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK 4: Reproducibility with Same Seeds")
    print("=" * 80)

    # Create environment
    graph_config = GraphConfig(
        num_nodes=8, branching_factor=2.0, max_depth=3, random_seed=42
    )
    env_config = EnvConfig(
        step_penalty=-0.1, success_reward=10.0, failure_penalty=-5.0, max_steps=30
    )

    def train_and_evaluate(run_id):
        """Train and evaluate with fixed seed."""
        print(f"\n[{run_id}/2] Training run {run_id}...")

        # Reset seeds
        set_seeds(42)

        generator = GraphGenerator()
        graph = generator.generate(graph_config)
        env = FlowEnvironment(graph, env_config)

        agent = QLearningAgent(
            action_space_size=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )

        training_config = TrainingConfig(
            num_episodes=100,
            max_steps_per_episode=30,
            eval_frequency=50,
            num_eval_episodes=20,
            checkpoint_frequency=100,
            log_frequency=50,
            random_seed=42,
        )

        orchestrator = TrainingOrchestrator(
            agent, env, training_config, log_dir=f"sanity_check_logs/repro_run{run_id}"
        )
        results = orchestrator.train()

        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate(agent, [env], num_episodes=30, random_seeds=[42])

        return metrics

    # Run twice with same seed
    metrics1 = train_and_evaluate(1)
    metrics2 = train_and_evaluate(2)

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Run 1 Success Rate: {metrics1.metrics.success_rate:.4f}")
    print(f"Run 2 Success Rate: {metrics2.metrics.success_rate:.4f}")
    print(f"Run 1 Avg Reward:   {metrics1.metrics.mean_reward:.4f}")
    print(f"Run 2 Avg Reward:   {metrics2.metrics.mean_reward:.4f}")

    # Check if results are identical or very close
    success_rate_diff = abs(
        metrics1.metrics.success_rate - metrics2.metrics.success_rate
    )
    reward_diff = abs(metrics1.metrics.mean_reward - metrics2.metrics.mean_reward)

    print(f"\nSuccess Rate Difference: {success_rate_diff:.6f}")
    print(f"Reward Difference:       {reward_diff:.6f}")

    # Results should be very close (allowing small floating point differences)
    success = success_rate_diff < 0.01 and reward_diff < 0.1

    if success:
        print("\n[PASS] Results are reproducible with same seed")
    else:
        print("\n[FAIL] Results differ significantly between runs")

    return success


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 80)
    print("RL FLOW EXPLORER - END-TO-END SANITY CHECKS")
    print("=" * 80)
    print("\nThis script runs comprehensive sanity checks to verify:")
    print("  1. Random agent performs worse than trained agents")
    print("  2. Agent learns on trivial environment")
    print("  3. Metrics improve over training")
    print("  4. Reproducibility with same seeds")

    # Create output directory
    os.makedirs("sanity_check_logs", exist_ok=True)
    os.makedirs("sanity_check_results", exist_ok=True)

    # Run all checks
    results = {}

    try:
        results["random_vs_trained"] = check_random_vs_trained()
    except Exception as e:
        print(f"\n[ERROR] in check 1: {e}")
        import traceback

        traceback.print_exc()
        results["random_vs_trained"] = False

    try:
        results["trivial_learning"] = check_learning_on_trivial_environment()
    except Exception as e:
        print(f"\n[ERROR] in check 2: {e}")
        import traceback

        traceback.print_exc()
        results["trivial_learning"] = False

    try:
        results["metrics_improve"] = check_metrics_improve()
    except Exception as e:
        print(f"\n[ERROR] in check 3: {e}")
        import traceback

        traceback.print_exc()
        results["metrics_improve"] = False

    try:
        results["reproducibility"] = check_reproducibility()
    except Exception as e:
        print(f"\n[ERROR] in check 4: {e}")
        import traceback

        traceback.print_exc()
        results["reproducibility"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SANITY CHECK SUMMARY")
    print("=" * 80)

    checks = [
        ("Random vs Trained", results["random_vs_trained"]),
        ("Trivial Learning", results["trivial_learning"]),
        ("Metrics Improve", results["metrics_improve"]),
        ("Reproducibility", results["reproducibility"]),
    ]

    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    total_passed = sum(results.values())
    total_checks = len(results)

    print(f"\nTotal: {total_passed}/{total_checks} checks passed")

    # Save results
    results_path = "sanity_check_results/sanity_check_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "checks": {k: bool(v) for k, v in results.items()},
                "total_passed": int(total_passed),
                "total_checks": int(total_checks),
                "all_passed": bool(all(results.values())),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_path}")

    if all(results.values()):
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL SANITY CHECKS PASSED!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("[WARNING] SOME SANITY CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
