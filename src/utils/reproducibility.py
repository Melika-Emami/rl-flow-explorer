"""
Reproducibility utilities for ensuring deterministic experiments.

This module provides functions to set random seeds and make all
stochastic operations deterministic across Python, NumPy, and PyTorch.
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set environment variables for additional determinism
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable PyTorch deterministic mode (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch versions don't have this function
            pass


def save_experiment_config(
    config: Dict[str, Any], output_dir: str, filename: str = "experiment_config.json"
) -> str:
    """
    Save experiment configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save configuration
        filename: Name of configuration file

    Returns:
        Path to saved configuration file
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    config_with_metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }

    # Convert any non-serializable types
    serializable_config = make_json_serializable(config_with_metadata)

    with open(output_path, "w") as f:
        json.dump(serializable_config, f, indent=2)

    return str(output_path)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        data = json.load(f)

    # Return just the config part if metadata is present
    if "config" in data:
        return data["config"]
    return data


def make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of object
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "__dict__"):
        # Handle dataclass or custom objects
        return make_json_serializable(obj.__dict__)
    else:
        return obj


def create_experiment_metadata(
    agent_type: str,
    env_config: Dict[str, Any],
    training_config: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create comprehensive experiment metadata dictionary.

    Args:
        agent_type: Type of agent being trained
        env_config: Environment configuration
        training_config: Training configuration
        agent_config: Agent-specific configuration (optional)
        additional_info: Any additional information (optional)

    Returns:
        Complete metadata dictionary
    """
    metadata = {
        "agent_type": agent_type,
        "environment": env_config,
        "training": training_config,
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["cudnn_version"] = torch.backends.cudnn.version()

    if agent_config:
        metadata["agent"] = agent_config

    if additional_info:
        metadata["additional_info"] = additional_info

    return metadata


def verify_reproducibility(
    func, seed: int, num_runs: int = 3, tolerance: float = 1e-6
) -> bool:
    """
    Verify that a function produces reproducible results.

    Args:
        func: Function to test (should take no arguments)
        seed: Random seed to use
        num_runs: Number of times to run the function
        tolerance: Tolerance for numerical differences

    Returns:
        True if all runs produce identical results within tolerance
    """
    results = []

    for _ in range(num_runs):
        set_global_seeds(seed)
        result = func()
        results.append(result)

    # Compare all results
    first_result = results[0]

    for result in results[1:]:
        if isinstance(first_result, np.ndarray):
            if not np.allclose(first_result, result, atol=tolerance):
                return False
        elif isinstance(first_result, torch.Tensor):
            if not torch.allclose(first_result, result, atol=tolerance):
                return False
        elif isinstance(first_result, (int, float)):
            if abs(first_result - result) > tolerance:
                return False
        else:
            if first_result != result:
                return False

    return True


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.

    Usage:
        with ReproducibilityContext(seed=42):
            # Your code here will be reproducible
            result = train_model()
    """

    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize reproducibility context.

        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = {}

    def __enter__(self):
        """Enter context and set seeds."""
        # Save original random states
        self.original_state["python"] = random.getstate()
        self.original_state["numpy"] = np.random.get_state()
        self.original_state["torch"] = torch.get_rng_state()

        if torch.cuda.is_available():
            self.original_state["cuda"] = torch.cuda.get_rng_state_all()

        # Set seeds
        set_global_seeds(self.seed, self.deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and optionally restore original states."""
        # Note: We don't restore states by default to maintain reproducibility
        # If you need to restore, uncomment the following:
        # random.setstate(self.original_state['python'])
        # np.random.set_state(self.original_state['numpy'])
        # torch.set_rng_state(self.original_state['torch'])
        # if torch.cuda.is_available():
        #     torch.cuda.set_rng_state_all(self.original_state['cuda'])
        pass


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility tracking.

    Returns:
        Dictionary with system information
    """
    info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["num_gpus"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return info


def save_hyperparameters(
    hyperparameters: Dict[str, Any],
    output_dir: str,
    filename: str = "hyperparameters.json",
) -> str:
    """
    Save hyperparameters to JSON file.

    Args:
        hyperparameters: Hyperparameter dictionary
        output_dir: Directory to save hyperparameters
        filename: Name of hyperparameters file

    Returns:
        Path to saved hyperparameters file
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add system info
    full_config = {
        "hyperparameters": hyperparameters,
        "system_info": get_system_info(),
        "timestamp": datetime.now().isoformat(),
    }

    # Convert to JSON-serializable format
    serializable_config = make_json_serializable(full_config)

    with open(output_path, "w") as f:
        json.dump(serializable_config, f, indent=2)

    return str(output_path)
