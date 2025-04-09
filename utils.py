# utils.py
"""Utility functions, including logger setup."""

import logging
import os
import yaml
import sys
from constants import LOG_FILE, LOG_FORMAT
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration schema for the Snake DQN training and evaluation."""

    episodes: int = 2000  # Number of training episodes
    patience: int = 250  # Early stopping patience
    save_path: str = "snake_dqn_best.pth"  # Path to save the best model
    plot_filename: str = "training_progress.png"  # Filename for training plot
    evaluation_games: int = 50  # Number of games for evaluation
    load_model_path: Optional[str] = (
        None  # Path to load a pre-trained model (None skips training)
    )

    @staticmethod
    def from_yaml(file_path: str = "config.yaml") -> "TrainConfig":
        """Loads training configuration from a YAML file with validation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found -> {file_path}")

        with open(file_path) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML: {e}")

        # Validate expected keys and types
        required_keys = {
            "episodes": int,
            "patience": int,
            "save_path": str,
            "plot_filename": str,
            "evaluation_games": int,
            "load_model_path": (str, type(None)),  # Can be string or None
        }

        for key, expected_type in required_keys.items():
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
            if not isinstance(config[key], expected_type):
                raise TypeError(
                    f"Expected `{key}` to be {expected_type}, got {type(config[key])}"
                )
        return TrainConfig(**config)


def ensure_directory(dir_path: str) -> Path:
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)  # Creates dir and parents if needed
    return path


def setup_logger():
    """Configures the root logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Create console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.ERROR)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create file handler
    logs_dir = ensure_directory("logs")
    file_handler = logging.FileHandler(logs_dir / LOG_FILE, mode="w", delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logger initialized. Logging to console and %s", LOG_FILE)
