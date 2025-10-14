"""
Evaluator: Runs evaluation protocols on trained agents.

This module provides functionality to evaluate trained agents on test
environments, collecting trajectories and computing metrics with support
for multiple random seeds and confidence intervals.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from ..agents.base import Agent
from ..environment.flow_environment import FlowEnvironment
from .metrics import MetricsCollector, EpisodeMetrics, AggregateMetrics


@dataclass
class EvaluationResults:
    """Results from evaluating an agent."""

    # Aggregated metrics
    metrics: AggregateMetrics

    # Per-episode metrics
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)

    # Additional information
    agent_name: str = ""
    environment_name: str = ""
    num_episodes: int = 0
    random_seeds: List[int] = field(default_factory=list)

    # Raw trajectories (optional, can be large)
    trajectories: Optional[List[List[Dict[str, Any]]]] = None


class Evaluator:
    """
    Evaluates trained agents on test environments.

    This class runs evaluation protocols, collecting trajectories and
    computing metrics across multiple episodes and random seeds.
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        verbose: bool = True,
    ):
        """
        Initialize the Evaluator.

        Args:
            metrics_collector: MetricsCollector instance (creates default if None)
            verbose: Whether to show progress bars and print information
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.verbose = verbose

    def evaluate(
        self,
        agent: Agent,
        test_envs: List[FlowEnvironment],
        num_episodes: int = 100,
        random_seeds: Optional[List[int]] = None,
        agent_name: str = "",
        environment_name: str = "",
        collect_trajectories: bool = False,
    ) -> EvaluationResults:
        """
        Evaluate agent on test environments.

        Runs the agent on multiple test environments for a specified number
        of episodes, collecting metrics and optionally full trajectories.

        Args:
            agent: Trained agent to evaluate
            test_envs: List of test environments
            num_episodes: Number of episodes to run per environment (default: 100)
            random_seeds: List of random seeds for reproducibility (optional)
            agent_name: Name of the agent for results tracking
            environment_name: Name of the environment for results tracking
            collect_trajectories: Whether to store full trajectories (can be large)

        Returns:
            EvaluationResults with aggregated metrics and episode-level data
        """
        if not test_envs:
            raise ValueError("Must provide at least one test environment")

        # Generate random seeds if not provided
        if random_seeds is None:
            random_seeds = [np.random.randint(0, 1000000) for _ in range(num_episodes)]
        elif len(random_seeds) < num_episodes:
            # Extend seeds if not enough provided
            base_seeds = random_seeds.copy()
            while len(random_seeds) < num_episodes:
                random_seeds.extend(base_seeds)
            random_seeds = random_seeds[:num_episodes]

        all_episode_metrics: List[EpisodeMetrics] = []
        all_trajectories: List[List[Dict[str, Any]]] = []

        # Distribute episodes across environments
        episodes_per_env = num_episodes // len(test_envs)
        remaining_episodes = num_episodes % len(test_envs)

        episode_idx = 0

        # Iterate over test environments
        for env_idx, env in enumerate(test_envs):
            # Determine number of episodes for this environment
            env_episodes = episodes_per_env
            if env_idx < remaining_episodes:
                env_episodes += 1

            if self.verbose:
                print(
                    f"\nEvaluating on environment {env_idx + 1}/{len(test_envs)} "
                    f"({env_episodes} episodes)..."
                )

            # Get total states for coverage calculation
            total_states = len(env.graph.nodes())

            # Run episodes
            iterator = range(env_episodes)
            if self.verbose:
                iterator = tqdm(iterator, desc=f"Env {env_idx + 1}", leave=False)

            for ep_num in iterator:
                seed = random_seeds[episode_idx]
                episode_idx += 1

                # Run single episode
                trajectory = self._run_episode(env, agent, seed)

                # Compute metrics for this episode
                episode_metrics = self.metrics_collector.evaluate_episode(
                    trajectory, total_states
                )
                all_episode_metrics.append(episode_metrics)

                # Store trajectory if requested
                if collect_trajectories:
                    all_trajectories.append(trajectory)

                # Print progress for slow episodes
                if self.verbose and len(trajectory) > 30:
                    print(
                        f"  Episode {ep_num + 1}/{env_episodes}: {len(trajectory)} steps, "
                        f"success={episode_metrics.success}, "
                        f"reward={episode_metrics.total_reward:.2f}"
                    )

        # Aggregate metrics across all episodes
        aggregate_metrics = self.metrics_collector.aggregate(all_episode_metrics)

        if self.verbose:
            self._print_summary(aggregate_metrics)

        # Create results object
        results = EvaluationResults(
            metrics=aggregate_metrics,
            episode_metrics=all_episode_metrics,
            agent_name=agent_name,
            environment_name=environment_name,
            num_episodes=num_episodes,
            random_seeds=random_seeds[:num_episodes],
            trajectories=all_trajectories if collect_trajectories else None,
        )

        return results

    def evaluate_single_env(
        self,
        agent: Agent,
        env: FlowEnvironment,
        num_episodes: int = 100,
        random_seeds: Optional[List[int]] = None,
        agent_name: str = "",
        environment_name: str = "",
        collect_trajectories: bool = False,
    ) -> EvaluationResults:
        """
        Evaluate agent on a single test environment.

        Convenience method for evaluating on a single environment.

        Args:
            agent: Trained agent to evaluate
            env: Test environment
            num_episodes: Number of episodes to run (default: 100)
            random_seeds: List of random seeds for reproducibility (optional)
            agent_name: Name of the agent for results tracking
            environment_name: Name of the environment for results tracking
            collect_trajectories: Whether to store full trajectories

        Returns:
            EvaluationResults with aggregated metrics and episode-level data
        """
        return self.evaluate(
            agent=agent,
            test_envs=[env],
            num_episodes=num_episodes,
            random_seeds=random_seeds,
            agent_name=agent_name,
            environment_name=environment_name,
            collect_trajectories=collect_trajectories,
        )

    def _run_episode(
        self,
        env: FlowEnvironment,
        agent: Agent,
        seed: int,
        max_steps_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a single episode with the agent in the environment.

        Args:
            env: Environment to run in
            agent: Agent to evaluate
            seed: Random seed for this episode
            max_steps_override: Override max steps to prevent infinite loops (optional)

        Returns:
            List of step dictionaries containing trajectory information
        """
        trajectory = []

        # Reset environment
        observation, info = env.reset(seed=seed)

        done = False
        step_count = 0

        # Use override if provided, otherwise use environment's max_steps
        max_steps = (
            max_steps_override
            if max_steps_override is not None
            else env.config.max_steps
        )

        while not done and step_count < max_steps:
            # Select action (not in training mode)
            action = agent.select_action(observation, training=False)

            # Take step in environment
            next_observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Store step information
            step_info = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
                "step": step_count,
            }
            trajectory.append(step_info)

            observation = next_observation
            step_count += 1

        # If we hit the override limit, mark as truncated
        if step_count >= max_steps and not done:
            if trajectory:
                trajectory[-1]["truncated"] = True
                trajectory[-1]["done"] = True
                trajectory[-1]["info"]["timeout"] = True

        return trajectory

    def _print_summary(self, metrics: AggregateMetrics):
        """
        Print a summary of evaluation metrics.

        Args:
            metrics: Aggregated metrics to summarize
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal Episodes: {metrics.num_episodes}")
        print(f"Successes: {metrics.num_successes}")
        print(f"Failures: {metrics.num_failures}")

        print(f"\n--- Success Metrics ---")
        print(
            f"Success Rate: {metrics.success_rate:.2%} ± {metrics.success_rate_std:.2%}"
        )
        print(
            f"  95% CI: [{metrics.success_rate_ci[0]:.2%}, {metrics.success_rate_ci[1]:.2%}]"
        )

        if metrics.num_successes > 0:
            print(
                f"Mean Steps to Success: {metrics.mean_steps_to_success:.2f} ± {metrics.std_steps_to_success:.2f}"
            )
            print(f"  95% CI: [{metrics.steps_ci[0]:.2f}, {metrics.steps_ci[1]:.2f}]")

        print(f"\n--- Failure Metrics ---")
        print(f"Failure Rate: {metrics.failure_rate:.2%}")
        if metrics.failure_types:
            print("Failure Types:")
            for failure_type, count in sorted(
                metrics.failure_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / metrics.num_episodes) * 100
                print(f"  {failure_type}: {count} ({percentage:.1f}%)")

        print(f"\n--- Coverage Metrics ---")
        print(
            f"Mean Coverage: {metrics.mean_coverage:.2%} ± {metrics.std_coverage:.2%}"
        )
        print(f"  95% CI: [{metrics.coverage_ci[0]:.2%}, {metrics.coverage_ci[1]:.2%}]")

        print(f"\n--- Reward Metrics ---")
        print(f"Mean Reward: {metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}")
        print(f"  95% CI: [{metrics.reward_ci[0]:.3f}, {metrics.reward_ci[1]:.3f}]")

        print(f"\n--- Episode Length ---")
        print(
            f"Mean Episode Length: {metrics.mean_episode_length:.2f} ± {metrics.std_episode_length:.2f}"
        )

        print("=" * 60 + "\n")
