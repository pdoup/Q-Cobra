# main.py
"""Main script to run the DQN training and evaluation for Snake."""

import argparse
from pathlib import Path
from pprint import pformat
import torch
import logging
import os

from utils import setup_logger, TrainConfig
from constants import STATE_SIZE, ACTION_SIZE
from dqn_agent import AdvancedDQNAgent
from train import train_agent
from evaluation import evaluate_agent


def main():

    def parse_args():
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(description="DQN Training & Evaluation")
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="config.yaml",  # Default YAML file if not specified
            help="Path to the YAML configuration file",
        )
        return parser.parse_args()

    # --- Setup ---
    setup_logger()  # Configure logging to console and file
    logger = logging.getLogger(__name__)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Configuration ---
    # Load config from YAML
    try:
        args = parse_args()
        config = TrainConfig.from_yaml(args.config)
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        argparse.ArgumentError,
    ) as e:
        logger.error(f"Configuration error: {e}")
        return  # Exit on config failure

    logger.info(
        f"Configuration loaded from {Path(args.config).name}: {pformat(dict(config.__dict__), indent=2, compact=True)}"
    )

    # --- Training ---
    final_agent = None
    if config.load_model_path is None:
        logger.info("Starting new training session...")
        # Train the agent. train_agent handles saving the best model internally.
        # It returns the agent's state at the *end* of training.
        final_agent, _, _ = train_agent(
            device=device,
            episodes=config.episodes,
            save_path=config.save_path,
            patience=config.patience,
            plot_filename=config.plot_filename,
        )
    else:
        logger.info("Skipping training. Loading model from: %s", config.load_model_path)
        # If we skip training, we need to load a model for evaluation later
        if not os.path.exists(config.load_model_path):
            logger.error(
                "Load model path specified, but file not found: %s",
                config.load_model_path,
            )
            return  # Exit if specified model doesn't exist
        # final_agent will remain None if only loading for eval below

    # --- Evaluation ---
    logger.info("Starting evaluation...")

    # Determine which model weights to evaluate
    model_to_evaluate_path = (
        config.load_model_path
        if config.load_model_path
        else Path("models", config.save_path)
    )

    if not os.path.exists(model_to_evaluate_path):
        logger.error("Model file for evaluation not found: %s", model_to_evaluate_path)
        logger.error("Cannot perform evaluation.")
        if final_agent:
            logger.warning(
                "Attempting to evaluate the agent from the end of the (interrupted?) training session..."
            )
            evaluate_agent(final_agent, device, num_games=config.evaluation_games)
        else:
            logger.error("No trained agent available in memory either.")

    else:
        logger.info("Loading model for evaluation from: %s", model_to_evaluate_path)
        # Create a new agent instance and load the saved weights
        eval_agent = AdvancedDQNAgent(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, device=device
        )
        try:
            eval_agent.policy_net.load_state_dict(
                torch.load(model_to_evaluate_path, map_location=device)
            )
            logger.info("Model weights loaded successfully.")
            evaluate_agent(eval_agent, device, num_games=config.evaluation_games)
        except Exception as e:
            logger.error(
                "Failed to load model weights from %s: %s",
                model_to_evaluate_path,
                e,
                exc_info=True,
            )

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
