"""
MetricsCollector: Computes and aggregates evaluation metrics.

This module provides functionality to compute metrics for individual episodes
and aggregate them across multiple episodes with statistical analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    success: bool
    steps_to_completion: int
    total_reward: float
    failure_type: Optional[str] = None
    visited_states: int = 0
    total_states: int = 0
    trajectory_length: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple episodes."""

    # Success metrics
    success_rate: float
    success_rate_std: float
    success_rate_ci: tuple  # 95% confidence interval

    # Steps metrics (for successful episodes)
    mean_steps_to_success: float
    std_steps_to_success: float
    steps_ci: tuple  # 95% confidence interval

    # Failure metrics
    failure_rate: float

    # Coverage metrics
    mean_coverage: float
    std_coverage: float
    coverage_ci: tuple

    # Reward metrics
    mean_reward: float
    std_reward: float
    reward_ci: tuple

    # Episode length metrics
    mean_episode_length: float
    std_episode_length: float

    # Raw data
    num_episodes: int
    num_successes: int
    num_failures: int

    # Failure types (with default)
    failure_types: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates evaluation metrics.

    This class computes metrics for individual episodes and aggregates
    them across multiple episodes with statistical analysis including
    confidence intervals.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the MetricsCollector.

        Args:
            confidence_level: Confidence level for confidence intervals (default: 0.95)
        """
        self.confidence_level = confidence_level

    def evaluate_episode(
        self,
        trajectory: List[Dict[str, Any]],
        total_states: int,
    ) -> EpisodeMetrics:
        """
        Compute metrics for a single episode.

        Args:
            trajectory: List of step dictionaries containing:
                - 'reward': float
                - 'done': bool (terminated or truncated)
                - 'info': dict with episode information
            total_states: Total number of states in the environment

        Returns:
            EpisodeMetrics object containing computed metrics
        """
        if not trajectory:
            # Empty trajectory - episode failed immediately
            return EpisodeMetrics(
                success=False,
                steps_to_completion=0,
                total_reward=0.0,
                failure_type="empty_trajectory",
                visited_states=0,
                total_states=total_states,
                trajectory_length=0,
            )

        # Extract information from trajectory
        total_reward = sum(step.get("reward", 0.0) for step in trajectory)
        trajectory_length = len(trajectory)

        # Get final step info
        final_step = trajectory[-1]
        final_info = final_step.get("info", {})

        # Determine success
        success = final_info.get("success", False)

        # Determine failure type
        failure_type = None
        if not success:
            failure_type = final_info.get("failure_type", "unknown")
            if final_info.get("timeout", False):
                failure_type = "timeout"

        # Count visited states
        visited_states = final_info.get("visited_nodes", 0)

        # Steps to completion
        steps_to_completion = trajectory_length

        # Compute coverage
        coverage = visited_states / total_states if total_states > 0 else 0.0

        return EpisodeMetrics(
            success=success,
            steps_to_completion=steps_to_completion,
            total_reward=total_reward,
            failure_type=failure_type,
            visited_states=visited_states,
            total_states=total_states,
            trajectory_length=trajectory_length,
        )

    def aggregate(self, episodes: List[EpisodeMetrics]) -> AggregateMetrics:
        """
        Aggregate metrics across multiple episodes.

        Computes mean, standard deviation, and confidence intervals for
        various metrics across all episodes.

        Args:
            episodes: List of EpisodeMetrics from multiple episodes

        Returns:
            AggregateMetrics object with aggregated statistics
        """
        if not episodes:
            # Return empty metrics if no episodes
            return AggregateMetrics(
                success_rate=0.0,
                success_rate_std=0.0,
                success_rate_ci=(0.0, 0.0),
                mean_steps_to_success=0.0,
                std_steps_to_success=0.0,
                steps_ci=(0.0, 0.0),
                failure_rate=0.0,
                failure_types={},
                mean_coverage=0.0,
                std_coverage=0.0,
                coverage_ci=(0.0, 0.0),
                mean_reward=0.0,
                std_reward=0.0,
                reward_ci=(0.0, 0.0),
                mean_episode_length=0.0,
                std_episode_length=0.0,
                num_episodes=0,
                num_successes=0,
                num_failures=0,
            )

        num_episodes = len(episodes)

        # Success metrics
        successes = [1 if ep.success else 0 for ep in episodes]
        num_successes = sum(successes)
        num_failures = num_episodes - num_successes

        success_rate = num_successes / num_episodes
        success_rate_std = np.std(successes, ddof=1) if num_episodes > 1 else 0.0
        success_rate_ci = self._compute_confidence_interval(successes)

        # Steps to success (only for successful episodes)
        successful_episodes = [ep for ep in episodes if ep.success]
        if successful_episodes:
            steps_to_success = [ep.steps_to_completion for ep in successful_episodes]
            mean_steps_to_success = np.mean(steps_to_success)
            std_steps_to_success = (
                np.std(steps_to_success, ddof=1) if len(steps_to_success) > 1 else 0.0
            )
            steps_ci = self._compute_confidence_interval(steps_to_success)
        else:
            mean_steps_to_success = 0.0
            std_steps_to_success = 0.0
            steps_ci = (0.0, 0.0)

        # Failure metrics
        failure_rate = num_failures / num_episodes
        failure_types: Dict[str, int] = {}
        for ep in episodes:
            if not ep.success and ep.failure_type:
                failure_types[ep.failure_type] = (
                    failure_types.get(ep.failure_type, 0) + 1
                )

        # Coverage metrics
        coverages = [
            ep.visited_states / ep.total_states if ep.total_states > 0 else 0.0
            for ep in episodes
        ]
        mean_coverage = np.mean(coverages)
        std_coverage = np.std(coverages, ddof=1) if num_episodes > 1 else 0.0
        coverage_ci = self._compute_confidence_interval(coverages)

        # Reward metrics
        rewards = [ep.total_reward for ep in episodes]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards, ddof=1) if num_episodes > 1 else 0.0
        reward_ci = self._compute_confidence_interval(rewards)

        # Episode length metrics
        episode_lengths = [ep.trajectory_length for ep in episodes]
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = (
            np.std(episode_lengths, ddof=1) if num_episodes > 1 else 0.0
        )

        return AggregateMetrics(
            success_rate=success_rate,
            success_rate_std=success_rate_std,
            success_rate_ci=success_rate_ci,
            mean_steps_to_success=mean_steps_to_success,
            std_steps_to_success=std_steps_to_success,
            steps_ci=steps_ci,
            failure_rate=failure_rate,
            failure_types=failure_types,
            mean_coverage=mean_coverage,
            std_coverage=std_coverage,
            coverage_ci=coverage_ci,
            mean_reward=mean_reward,
            std_reward=std_reward,
            reward_ci=reward_ci,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            num_episodes=num_episodes,
            num_successes=num_successes,
            num_failures=num_failures,
        )

    def _compute_confidence_interval(self, data: List[float]) -> tuple:
        """
        Compute confidence interval for a list of values.

        Args:
            data: List of numerical values

        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if not data or len(data) < 2:
            # Not enough data for confidence interval
            mean_val = np.mean(data) if data else 0.0
            return (mean_val, mean_val)

        data_array = np.array(data)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)  # Standard error of the mean

        # Compute confidence interval using t-distribution
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(data_array) - 1,
            loc=mean,
            scale=std_err,
        )

        return confidence_interval
