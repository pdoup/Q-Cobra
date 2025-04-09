# plotting.py
"""Contains functions for plotting training results."""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_training_progress(
    episode_rewards, avg_scores, losses, filename="training_progress.png"
):
    """Generates and saves plots for training progress."""
    if not episode_rewards:
        logger.warning("No data provided for plotting.")
        return

    logger.info("Generating training progress plot...")
    try:
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        episodes = range(1, len(episode_rewards) + 1)

        # Plot 1: Score per Episode & Rolling Average
        axs[0].plot(
            episodes,
            episode_rewards,
            label="Score per Episode",
            alpha=0.6,
            color="blue",
        )
        axs[0].plot(
            episodes,
            avg_scores,
            label="Average Score (100 episodes)",
            linewidth=2,
            color="red",
        )
        axs[0].set_ylabel("Score")
        axs[0].set_title("Episode Score and Moving Average Score")
        axs[0].legend()
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # Plot 2: Average Score (Smoother view)
        axs[1].plot(
            episodes, avg_scores, label="Average Score (100 episodes)", color="red"
        )
        axs[1].set_ylabel("Average Score")
        axs[1].set_title("Average Score Over Time")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.6)

        # Plot 3: Loss per Episode
        valid_losses = [
            (i + 1, loss)
            for i, loss in enumerate(losses)
            if loss is not None and loss > 0
        ]  # Filter 0 loss too for log scale
        if valid_losses:
            loss_episodes, loss_values = zip(*valid_losses)
            axs[2].plot(
                loss_episodes,
                loss_values,
                label="Average Loss per Episode",
                color="orange",
                alpha=0.8,
                linestyle="-",
                marker=".",
                markersize=2,
            )

        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Loss (Log Scale)")
        axs[2].set_title("Training Loss Over Time")
        axs[2].legend()
        axs[2].grid(True, linestyle="--", alpha=0.6)
        # Use log scale only if there are valid losses > 0
        if valid_losses:
            axs[2].set_yscale("log")
        else:
            axs[2].set_ylabel("Loss")  # Linear scale if no valid data or all zero

        plt.tight_layout()
        plt.savefig(filename)
        logger.info("Training progress plot saved to %s", filename)
        plt.close(fig)

    except Exception as e:
        logger.error("Failed to generate plot: %s", e, exc_info=True)
