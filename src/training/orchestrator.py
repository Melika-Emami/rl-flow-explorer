"""Training orchestration for RL agents."""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agents.base import Agent
from src.environment.flow_environment import FlowEnvironment
from src.utils.data_models import TrainingConfig, Transition
from src.utils.logger import ExperimentLogger, setup_deterministic_logging


class TrainingOrchestrator:
    """
    Coordinates training across multiple episodes with logging and checkpointing.
    """

    def __init__(
        self,
        agent: Agent,
        env: FlowEnvironment,
        config: TrainingConfig,
        val_envs: Optional[List[FlowEnvironment]] = None,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        exploration_strategy=None,
    ):
        """
        Initialize training orchestrator.

        Args:
            agent: Agent to train
            env: Training environment
            config: Training configuration
            val_envs: Optional list of validation environments
            log_dir: Directory for logs and checkpoints
            experiment_name: Optional name for the experiment
            exploration_strategy: Optional exploration strategy for intrinsic rewards
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.val_envs = val_envs or []
        self.exploration_strategy = exploration_strategy
        self.log_dir = Path(log_dir)

        # Create directories
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # Initialize experiment logger
        self.exp_logger = ExperimentLogger(str(self.log_dir), experiment_name)

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_val_reward = -float("inf")
        self.episodes_since_improvement = 0

        # Metrics storage
        self.training_metrics: List[Dict[str, Any]] = []

        # Set random seeds for reproducibility
        setup_deterministic_logging(config.random_seed)
        self._set_seeds(config.random_seed)

        # Log configuration and hyperparameters
        self._log_configuration()

    def _log_configuration(self) -> None:
        """Log training configuration and hyperparameters."""
        config_dict = {
            "num_episodes": self.config.num_episodes,
            "max_steps_per_episode": self.config.max_steps_per_episode,
            "learning_rate": self.config.learning_rate,
            "discount_factor": self.config.discount_factor,
            "batch_size": self.config.batch_size,
            "eval_frequency": self.config.eval_frequency,
            "num_eval_episodes": self.config.num_eval_episodes,
            "random_seed": self.config.random_seed,
            "checkpoint_frequency": self.config.checkpoint_frequency,
            "log_frequency": self.config.log_frequency,
            "early_stopping_patience": self.config.early_stopping_patience,
            "save_dir": self.config.save_dir,
        }
        self.exp_logger.log_config(config_dict)
        self.exp_logger.log_hyperparameters(config_dict)

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def train(self) -> Dict[str, Any]:
        """
        Execute main training loop.

        Returns:
            Dictionary containing training results and metrics
        """
        self.exp_logger.log_message(
            f"Starting training for {self.config.num_episodes} episodes..."
        )
        self.exp_logger.log_message(f"Logs will be saved to: {self.log_dir}")

        start_time = time.time()

        for episode in range(self.config.num_episodes):
            self.episode = episode

            # Collect episode trajectory
            episode_metrics = self._collect_episode()

            # Log metrics
            if episode % self.config.log_frequency == 0:
                self._log_metrics(episode_metrics, episode)

            # Store metrics
            self.training_metrics.append(episode_metrics)

            # Periodic evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                val_metrics = self._evaluate_validation()
                self._log_validation_metrics(val_metrics, episode)

                # Check for improvement
                if val_metrics["mean_reward"] > self.best_val_reward:
                    self.best_val_reward = val_metrics["mean_reward"]
                    self.episodes_since_improvement = 0
                    self._save_checkpoint(episode, is_best=True)
                else:
                    self.episodes_since_improvement += self.config.eval_frequency

            # Periodic checkpoint saving
            if episode % self.config.checkpoint_frequency == 0 and episode > 0:
                self._save_checkpoint(episode, is_best=False)

            # Early stopping check
            if self.episodes_since_improvement >= self.config.early_stopping_patience:
                print(
                    f"Early stopping at episode {episode} - no improvement for {self.config.early_stopping_patience} episodes"
                )
                break

        # Final checkpoint
        self._save_checkpoint(self.episode, is_best=False, final=True)

        # Save training metrics to JSON
        self._save_metrics_json()

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Close TensorBoard writer
        self.writer.close()

        return {
            "training_metrics": self.training_metrics,
            "best_val_reward": self.best_val_reward,
            "total_episodes": self.episode + 1,
            "total_steps": self.total_steps,
            "training_time": training_time,
        }

    def _collect_episode(self) -> Dict[str, Any]:
        """
        Collect a single episode trajectory and update agent.

        Returns:
            Dictionary of episode metrics
        """
        state, info = self.env.reset()
        episode_reward = 0.0
        episode_intrinsic_reward = 0.0
        episode_steps = 0
        done = False
        truncated = False

        update_metrics_sum = {}
        update_count = 0

        # Reset exploration strategy at episode start
        if self.exploration_strategy is not None:
            self.exploration_strategy.reset()

        while (
            not (done or truncated)
            and episode_steps < self.config.max_steps_per_episode
        ):
            # Select action
            action = self.agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, truncated, info = self.env.step(action)

            # Compute intrinsic reward bonus if exploration strategy is provided
            intrinsic_reward = 0.0
            if self.exploration_strategy is not None:
                intrinsic_reward = self.exploration_strategy.compute_bonus(
                    state, action, next_state, info
                )
                episode_intrinsic_reward += intrinsic_reward

            # Combine extrinsic and intrinsic rewards
            total_reward = reward + intrinsic_reward

            # Create transition with combined reward
            transition = Transition(
                state=state,
                action=action,
                reward=total_reward,
                next_state=next_state,
                done=done or truncated,
                info=info,
            )

            # Update agent
            update_metrics = self.agent.update(transition)

            # Update exploration strategy
            if self.exploration_strategy is not None:
                self.exploration_strategy.update(state, action, next_state)

            # Accumulate update metrics
            for key, value in update_metrics.items():
                if key not in update_metrics_sum:
                    update_metrics_sum[key] = 0.0
                update_metrics_sum[key] += value
            update_count += 1

            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

        # Average update metrics
        avg_update_metrics = {
            key: value / update_count if update_count > 0 else 0.0
            for key, value in update_metrics_sum.items()
        }

        # Compile episode metrics
        episode_metrics = {
            "episode": self.episode,
            "reward": episode_reward,
            "intrinsic_reward": episode_intrinsic_reward,
            "steps": episode_steps,
            "success": info.get("success", False),
            "failure": info.get("failure", False),
            "total_steps": self.total_steps,
            **avg_update_metrics,
        }

        # Add exploration strategy statistics if available
        if self.exploration_strategy is not None and hasattr(
            self.exploration_strategy, "get_statistics"
        ):
            exploration_stats = self.exploration_strategy.get_statistics()
            for key, value in exploration_stats.items():
                episode_metrics[f"exploration_{key}"] = value

        return episode_metrics

    def _evaluate_validation(self) -> Dict[str, Any]:
        """
        Evaluate agent on validation environments.

        Returns:
            Dictionary of validation metrics
        """
        if not self.val_envs:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "success_rate": 0.0,
                "mean_steps": 0.0,
            }

        all_rewards = []
        all_successes = []
        all_steps = []

        for val_env in self.val_envs:
            for _ in range(self.config.num_eval_episodes):
                state, info = val_env.reset()
                episode_reward = 0.0
                episode_steps = 0
                done = False
                truncated = False

                while (
                    not (done or truncated)
                    and episode_steps < self.config.max_steps_per_episode
                ):
                    action = self.agent.select_action(state, training=False)
                    state, reward, done, truncated, info = val_env.step(action)
                    episode_reward += reward
                    episode_steps += 1

                all_rewards.append(episode_reward)
                all_successes.append(1.0 if info.get("success", False) else 0.0)
                all_steps.append(episode_steps)

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "success_rate": np.mean(all_successes),
            "mean_steps": np.mean(all_steps),
        }

    def _log_metrics(self, metrics: Dict[str, Any], episode: int) -> None:
        """
        Log training metrics to TensorBoard and console.

        Args:
            metrics: Dictionary of metrics to log
            episode: Current episode number
        """
        # Log to TensorBoard
        self.writer.add_scalar("train/reward", metrics["reward"], episode)
        self.writer.add_scalar("train/steps", metrics["steps"], episode)
        self.writer.add_scalar(
            "train/success", 1.0 if metrics["success"] else 0.0, episode
        )

        # Log agent-specific metrics
        for key, value in metrics.items():
            if key not in [
                "episode",
                "reward",
                "steps",
                "success",
                "failure",
                "total_steps",
            ]:
                self.writer.add_scalar(f"train/{key}", value, episode)

        # Console output
        success_str = "Y" if metrics["success"] else "N"
        print(
            f"Episode {episode:5d} | "
            f"Reward: {metrics['reward']:7.2f} | "
            f"Steps: {metrics['steps']:3d} | "
            f"Success: {success_str}"
        )

    def _log_validation_metrics(self, metrics: Dict[str, Any], episode: int) -> None:
        """
        Log validation metrics to TensorBoard and console.

        Args:
            metrics: Dictionary of validation metrics
            episode: Current episode number
        """
        # Log to TensorBoard
        self.writer.add_scalar("val/mean_reward", metrics["mean_reward"], episode)
        self.writer.add_scalar("val/success_rate", metrics["success_rate"], episode)
        self.writer.add_scalar("val/mean_steps", metrics["mean_steps"], episode)

        # Console output
        print(
            f"\n{'='*60}\n"
            f"Validation @ Episode {episode}\n"
            f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics.get('std_reward', 0):.2f}\n"
            f"Success Rate: {metrics['success_rate']:.2%}\n"
            f"Mean Steps: {metrics['mean_steps']:.1f}\n"
            f"{'='*60}\n"
        )

    def _save_checkpoint(
        self, episode: int, is_best: bool = False, final: bool = False
    ) -> None:
        """
        Save agent checkpoint.

        Args:
            episode: Current episode number
            is_best: Whether this is the best model so far
            final: Whether this is the final checkpoint
        """
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            print(f"Saving best model to {checkpoint_path}")
        elif final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
            print(f"Saving final model to {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"

        self.agent.save(str(checkpoint_path))

        # Save training state
        state_path = checkpoint_path.with_suffix(".json")
        training_state = {
            "episode": episode,
            "total_steps": self.total_steps,
            "best_val_reward": self.best_val_reward,
            "episodes_since_improvement": self.episodes_since_improvement,
        }
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

    def _save_metrics_json(self) -> None:
        """Save all training metrics to JSON file."""
        metrics_path = self.log_dir / "training_metrics.json"

        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = []
        for metric in self.training_metrics:
            serializable_metric = {}
            for key, value in metric.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metric[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_metric[key] = value.tolist()
                else:
                    serializable_metric[key] = value
            serializable_metrics.append(serializable_metric)

        # Save configuration and metrics
        output = {
            "config": {
                "num_episodes": self.config.num_episodes,
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "batch_size": self.config.batch_size,
                "random_seed": self.config.random_seed,
            },
            "metrics": serializable_metrics,
        }

        with open(metrics_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Training metrics saved to {metrics_path}")
