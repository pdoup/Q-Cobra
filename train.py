# training.py
"""Contains the main training loop for the DQNAgent."""

import torch
from collections import deque
from statistics import mean
from tqdm import tqdm
import logging
from time import time_ns

from snake_game import SnakeGame
from dqn_agent import AdvancedDQNAgent
from plotting import plot_training_progress
from constants import GRID_HEIGHT, GRID_WIDTH, ACTION_SIZE
from utils import ensure_directory

logger = logging.getLogger(__name__)


def train_agent(
    device,
    episodes=2000,
    save_path="snake_dqn_best.pth",
    patience=300,
    plot_filename="training_progress.png",
):
    logger.info("Starting training process...")
    logger.info(
        f"Max Episodes: {episodes}, Patience: {patience}, Save Path: {save_path}"
    )

    game = SnakeGame()
    agent = AdvancedDQNAgent(
        state_size=game.state_size, action_size=ACTION_SIZE, device=device
    )

    scores_window = deque(maxlen=100)
    all_episode_scores = []
    all_avg_scores = []
    all_losses = []
    best_avg_score = -float("inf")
    episodes_since_best = 0
    training_complete = False
    save_dir = ensure_directory("models")
    plots_dir = ensure_directory("plots")

    try:
        with tqdm(
            range(1, episodes + 1),
            desc="Training",
            unit="episode",
            ncols=100,
            colour="blue",
            dynamic_ncols=True,
        ) as pbar:
            for episode in pbar:
                state = game.reset()
                episode_losses = []
                done = False
                step = 0

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done = game.step(action)
                    agent.store(state, action, reward, next_state, done)
                    loss = agent.train()
                    if loss is not None:
                        episode_losses.append(loss)

                    state = next_state
                    step += 1
                    # Add a max step limit per episode to prevent potential infinite loops
                    if step > (GRID_WIDTH * GRID_HEIGHT * 3):
                        logger.warning(
                            "Episode %d exceeded max steps, ending episode.", episode
                        )
                        done = True  # Force end episode

                episode_score = game.score  # Use final score from the game object
                scores_window.append(episode_score)
                all_episode_scores.append(episode_score)
                avg_score = mean(scores_window) if scores_window else 0
                all_avg_scores.append(avg_score)
                avg_loss = mean(episode_losses) if episode_losses else 0
                all_losses.append(avg_loss if episode_losses else None)
                training_complete = True  # Mark that at least one episode ran

                pbar.set_postfix(
                    {
                        "Score": episode_score,
                        "Avg": f"{avg_score:.2f}",
                        "Best": f"{best_avg_score:.2f}",
                        "Eps": f"{agent.epsilon:.3f}",
                        "LR": f"{agent.scheduler.get_last_lr()[0]:.6f}",
                        "Loss": f"{avg_loss:.4f}" if avg_loss != 0 else "N/A",
                        "Pat": f"{episodes_since_best}/{patience}",
                        "Steps": agent.steps_done,
                    }
                )

                # Checkpointing and Early Stopping
                if len(scores_window) == scores_window.maxlen:
                    if avg_score > best_avg_score:
                        logger.info(
                            "Average score improved: %.2f -> %.2f. Saving model...",
                            best_avg_score,
                            avg_score,
                        )
                        best_avg_score = avg_score
                        torch.save(agent.policy_net.state_dict(), save_dir / save_path)
                        episodes_since_best = 0
                    else:
                        episodes_since_best += 1

                    if episodes_since_best >= patience:
                        logger.info(
                            "Early stopping triggered: No improvement in avg score for %d episodes.",
                            patience,
                        )
                        break  # Exit loop

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        logger.info("Training loop finished.")
        if training_complete:
            if best_avg_score > -float("inf"):
                logger.info("Best average score achieved: %.2f", best_avg_score)
                logger.info("Model saved to %s", save_path)
            else:
                logger.warning(
                    "Training finished, but no improvement recorded or model saved."
                )

            # Plotting after training, regardless of how it finished
            plot_training_progress(
                all_episode_scores,
                all_avg_scores,
                all_losses,
                filename=plots_dir / plot_filename,
            )
        else:
            logger.warning("No training episodes completed.")

    # Return the agent state at the end of training, and history ('best' model weights are in the saved file)
    return agent, all_episode_scores, all_avg_scores
