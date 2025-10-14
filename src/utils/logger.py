"""Logging utilities for training and evaluation."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentLogger:
    """
    Handles logging for experiments with deterministic output.
    """

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize experiment logger.

        Args:
            log_dir: Directory for logs
            experiment_name: Optional name for the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"

        # Set up Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters.

        Args:
            config: Configuration dictionary
        """
        config_path = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to {config_path}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            hyperparameters: Hyperparameter dictionary
        """
        hp_path = self.log_dir / f"{self.experiment_name}_hyperparameters.json"
        with open(hp_path, "w") as f:
            json.dump(hyperparameters, f, indent=2)

        self.logger.info(f"Hyperparameters saved to {hp_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics at a given step.

        Args:
            metrics: Metrics dictionary
            step: Training step or episode number
        """
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")

    def log_message(self, message: str, level: str = "info") -> None:
        """
        Log a custom message.

        Args:
            message: Message to log
            level: Logging level (info, warning, error)
        """
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save final results to JSON.

        Args:
            results: Results dictionary
        """
        results_path = self.log_dir / f"{self.experiment_name}_results.json"

        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of object
        """
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


def setup_deterministic_logging(seed: int) -> None:
    """
    Set up deterministic logging with fixed random seed.

    Args:
        seed: Random seed for reproducibility
    """
    import random
    import numpy as np
    import torch

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set Python hash seed for deterministic hashing
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
