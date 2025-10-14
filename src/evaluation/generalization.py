"""
Generalization evaluation: Zero-shot evaluation and transfer learning protocols.

This module provides functionality to evaluate how well trained agents
generalize to unseen environments, including in-distribution and
out-of-distribution test sets with various distribution shifts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..agents.base import Agent
from ..environment.flow_environment import FlowEnvironment, EnvConfig
from ..environment.graph_generator import GraphGenerator, GraphConfig
from .evaluator import Evaluator, EvaluationResults
from .metrics import AggregateMetrics


@dataclass
class GeneralizationResults:
    """Results from generalization evaluation."""

    # In-distribution results
    in_dist_results: EvaluationResults

    # Out-of-distribution results by shift type
    out_dist_results: Dict[str, EvaluationResults] = field(default_factory=dict)

    # Performance degradation metrics
    degradation_metrics: Dict[str, float] = field(default_factory=dict)

    # Agent and experiment info
    agent_name: str = ""
    base_config: Optional[GraphConfig] = None


@dataclass
class PerformanceDegradation:
    """Metrics for performance degradation from in-dist to out-of-dist."""

    # Success rate degradation
    success_rate_in_dist: float = 0.0
    success_rate_out_dist: float = 0.0
    success_rate_drop: float = 0.0  # Absolute drop
    success_rate_drop_pct: float = 0.0  # Percentage drop

    # Steps to success degradation
    steps_in_dist: float = 0.0
    steps_out_dist: float = 0.0
    steps_increase: float = 0.0  # Absolute increase
    steps_increase_pct: float = 0.0  # Percentage increase

    # Coverage degradation
    coverage_in_dist: float = 0.0
    coverage_out_dist: float = 0.0
    coverage_drop: float = 0.0  # Absolute drop
    coverage_drop_pct: float = 0.0  # Percentage drop

    # Reward degradation
    reward_in_dist: float = 0.0
    reward_out_dist: float = 0.0
    reward_drop: float = 0.0  # Absolute drop
    reward_drop_pct: float = 0.0  # Percentage drop


class GeneralizationEvaluator:
    """
    Evaluates agent generalization and transfer capabilities.

    This class implements zero-shot evaluation protocols, measuring
    performance on in-distribution and out-of-distribution test sets
    and computing performance degradation metrics.
    """

    def __init__(
        self,
        evaluator: Optional[Evaluator] = None,
        verbose: bool = True,
    ):
        """
        Initialize the GeneralizationEvaluator.

        Args:
            evaluator: Evaluator instance (creates default if None)
            verbose: Whether to show progress and print information
        """
        self.evaluator = evaluator or Evaluator(verbose=verbose)
        self.verbose = verbose

    def evaluate_zero_shot(
        self,
        agent: Agent,
        graph_splits: Dict[str, List],
        env_config: EnvConfig,
        num_episodes_per_env: int = 10,
        agent_name: str = "",
    ) -> GeneralizationResults:
        """
        Evaluate agent zero-shot on multiple test sets.

        Runs the trained agent on in-distribution and out-of-distribution
        test sets without any fine-tuning, measuring generalization performance.

        Args:
            agent: Trained agent to evaluate
            graph_splits: Dictionary mapping split names to lists of graphs
                         Expected keys: 'test_in_dist', 'test_deeper', 'test_wider', etc.
            env_config: Environment configuration for creating test environments
            num_episodes_per_env: Number of episodes to run per test environment
            agent_name: Name of the agent for results tracking

        Returns:
            GeneralizationResults with in-dist and out-of-dist performance
        """
        if "test_in_dist" not in graph_splits:
            raise ValueError("graph_splits must contain 'test_in_dist' key")

        # Evaluate on in-distribution test set
        if self.verbose:
            print("\n" + "=" * 60)
            print("ZERO-SHOT GENERALIZATION EVALUATION")
            print("=" * 60)
            print(f"\nAgent: {agent_name}")
            print(f"Episodes per environment: {num_episodes_per_env}")

        in_dist_envs = self._create_environments(
            graph_splits["test_in_dist"], env_config
        )

        if self.verbose:
            print(f"\n--- In-Distribution Test Set ---")
            print(f"Number of test environments: {len(in_dist_envs)}")

        in_dist_results = self.evaluator.evaluate(
            agent=agent,
            test_envs=in_dist_envs,
            num_episodes=len(in_dist_envs) * num_episodes_per_env,
            agent_name=agent_name,
            environment_name="in_dist",
            collect_trajectories=False,
        )

        # Evaluate on out-of-distribution test sets
        out_dist_results = {}

        for split_name, graphs in graph_splits.items():
            if split_name == "test_in_dist" or split_name == "train":
                continue  # Skip in-dist and train sets

            if self.verbose:
                print(f"\n--- {split_name} Test Set ---")
                print(f"Number of test environments: {len(graphs)}")

            test_envs = self._create_environments(graphs, env_config)

            results = self.evaluator.evaluate(
                agent=agent,
                test_envs=test_envs,
                num_episodes=len(test_envs) * num_episodes_per_env,
                agent_name=agent_name,
                environment_name=split_name,
                collect_trajectories=False,
            )

            out_dist_results[split_name] = results

        # Compute performance degradation metrics
        degradation_metrics = self._compute_degradation_metrics(
            in_dist_results, out_dist_results
        )

        if self.verbose:
            self._print_degradation_summary(degradation_metrics)

        return GeneralizationResults(
            in_dist_results=in_dist_results,
            out_dist_results=out_dist_results,
            degradation_metrics=degradation_metrics,
            agent_name=agent_name,
        )

    def evaluate_distribution_shift(
        self,
        agent: Agent,
        in_dist_graphs: List,
        out_dist_graphs: List,
        env_config: EnvConfig,
        shift_name: str = "distribution_shift",
        num_episodes_per_env: int = 10,
        agent_name: str = "",
    ) -> PerformanceDegradation:
        """
        Evaluate agent on a specific distribution shift.

        Compares performance between in-distribution and a specific
        out-of-distribution test set.

        Args:
            agent: Trained agent to evaluate
            in_dist_graphs: List of in-distribution test graphs
            out_dist_graphs: List of out-of-distribution test graphs
            env_config: Environment configuration
            shift_name: Name of the distribution shift
            num_episodes_per_env: Number of episodes per environment
            agent_name: Name of the agent

        Returns:
            PerformanceDegradation metrics
        """
        # Evaluate on in-distribution
        in_dist_envs = self._create_environments(in_dist_graphs, env_config)
        in_dist_results = self.evaluator.evaluate(
            agent=agent,
            test_envs=in_dist_envs,
            num_episodes=len(in_dist_envs) * num_episodes_per_env,
            agent_name=agent_name,
            environment_name="in_dist",
        )

        # Evaluate on out-of-distribution
        out_dist_envs = self._create_environments(out_dist_graphs, env_config)
        out_dist_results = self.evaluator.evaluate(
            agent=agent,
            test_envs=out_dist_envs,
            num_episodes=len(out_dist_envs) * num_episodes_per_env,
            agent_name=agent_name,
            environment_name=shift_name,
        )

        # Compute degradation
        degradation = self._compute_single_degradation(
            in_dist_results.metrics, out_dist_results.metrics
        )

        return degradation

    def _create_environments(
        self, graphs: List, env_config: EnvConfig
    ) -> List[FlowEnvironment]:
        """
        Create FlowEnvironment instances from graphs.

        Args:
            graphs: List of NetworkX graphs
            env_config: Environment configuration

        Returns:
            List of FlowEnvironment instances
        """
        environments = []
        for graph in graphs:
            env = FlowEnvironment(graph=graph, config=env_config)
            environments.append(env)
        return environments

    def _compute_degradation_metrics(
        self,
        in_dist_results: EvaluationResults,
        out_dist_results: Dict[str, EvaluationResults],
    ) -> Dict[str, PerformanceDegradation]:
        """
        Compute performance degradation for all distribution shifts.

        Args:
            in_dist_results: In-distribution evaluation results
            out_dist_results: Dictionary of out-of-distribution results

        Returns:
            Dictionary mapping shift names to degradation metrics
        """
        degradation_metrics = {}

        for shift_name, out_results in out_dist_results.items():
            degradation = self._compute_single_degradation(
                in_dist_results.metrics, out_results.metrics
            )
            degradation_metrics[shift_name] = degradation

        return degradation_metrics

    def _compute_single_degradation(
        self,
        in_dist_metrics: AggregateMetrics,
        out_dist_metrics: AggregateMetrics,
    ) -> PerformanceDegradation:
        """
        Compute performance degradation between two metric sets.

        Args:
            in_dist_metrics: In-distribution metrics
            out_dist_metrics: Out-of-distribution metrics

        Returns:
            PerformanceDegradation object
        """
        # Success rate degradation
        success_rate_drop = in_dist_metrics.success_rate - out_dist_metrics.success_rate
        success_rate_drop_pct = (
            (success_rate_drop / in_dist_metrics.success_rate * 100)
            if in_dist_metrics.success_rate > 0
            else 0.0
        )

        # Steps to success degradation (only if both have successes)
        steps_increase = 0.0
        steps_increase_pct = 0.0
        if (
            in_dist_metrics.mean_steps_to_success > 0
            and out_dist_metrics.mean_steps_to_success > 0
        ):
            steps_increase = (
                out_dist_metrics.mean_steps_to_success
                - in_dist_metrics.mean_steps_to_success
            )
            steps_increase_pct = (
                steps_increase / in_dist_metrics.mean_steps_to_success * 100
            )

        # Coverage degradation
        coverage_drop = in_dist_metrics.mean_coverage - out_dist_metrics.mean_coverage
        coverage_drop_pct = (
            (coverage_drop / in_dist_metrics.mean_coverage * 100)
            if in_dist_metrics.mean_coverage > 0
            else 0.0
        )

        # Reward degradation
        reward_drop = in_dist_metrics.mean_reward - out_dist_metrics.mean_reward
        reward_drop_pct = (
            (reward_drop / abs(in_dist_metrics.mean_reward) * 100)
            if in_dist_metrics.mean_reward != 0
            else 0.0
        )

        return PerformanceDegradation(
            success_rate_in_dist=in_dist_metrics.success_rate,
            success_rate_out_dist=out_dist_metrics.success_rate,
            success_rate_drop=success_rate_drop,
            success_rate_drop_pct=success_rate_drop_pct,
            steps_in_dist=in_dist_metrics.mean_steps_to_success,
            steps_out_dist=out_dist_metrics.mean_steps_to_success,
            steps_increase=steps_increase,
            steps_increase_pct=steps_increase_pct,
            coverage_in_dist=in_dist_metrics.mean_coverage,
            coverage_out_dist=out_dist_metrics.mean_coverage,
            coverage_drop=coverage_drop,
            coverage_drop_pct=coverage_drop_pct,
            reward_in_dist=in_dist_metrics.mean_reward,
            reward_out_dist=out_dist_metrics.mean_reward,
            reward_drop=reward_drop,
            reward_drop_pct=reward_drop_pct,
        )

    def _print_degradation_summary(
        self, degradation_metrics: Dict[str, PerformanceDegradation]
    ):
        """
        Print summary of performance degradation across shifts.

        Args:
            degradation_metrics: Dictionary of degradation metrics
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE DEGRADATION SUMMARY")
        print("=" * 60)

        for shift_name, degradation in degradation_metrics.items():
            print(f"\n--- {shift_name} ---")

            print(f"\nSuccess Rate:")
            print(f"  In-Dist:  {degradation.success_rate_in_dist:.2%}")
            print(f"  Out-Dist: {degradation.success_rate_out_dist:.2%}")
            print(
                f"  Drop:     {degradation.success_rate_drop:.2%} "
                f"({degradation.success_rate_drop_pct:+.1f}%)"
            )

            if degradation.steps_in_dist > 0 and degradation.steps_out_dist > 0:
                print(f"\nSteps to Success:")
                print(f"  In-Dist:  {degradation.steps_in_dist:.2f}")
                print(f"  Out-Dist: {degradation.steps_out_dist:.2f}")
                print(
                    f"  Increase: {degradation.steps_increase:+.2f} "
                    f"({degradation.steps_increase_pct:+.1f}%)"
                )

            print(f"\nCoverage:")
            print(f"  In-Dist:  {degradation.coverage_in_dist:.2%}")
            print(f"  Out-Dist: {degradation.coverage_out_dist:.2%}")
            print(
                f"  Drop:     {degradation.coverage_drop:.2%} "
                f"({degradation.coverage_drop_pct:+.1f}%)"
            )

            print(f"\nMean Reward:")
            print(f"  In-Dist:  {degradation.reward_in_dist:.3f}")
            print(f"  Out-Dist: {degradation.reward_out_dist:.3f}")
            print(
                f"  Drop:     {degradation.reward_drop:+.3f} "
                f"({degradation.reward_drop_pct:+.1f}%)"
            )

        print("=" * 60 + "\n")
