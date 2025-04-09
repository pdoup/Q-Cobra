# evaluation.py
"""Contains the function to evaluate a trained agent."""

import torch
from statistics import mean
from tqdm import tqdm
import logging

from constants import GRID_HEIGHT, GRID_WIDTH
from snake_game import SnakeGame

logger = logging.getLogger(__name__)


def evaluate_agent(agent, device, num_games=100):
    logger.info("Starting evaluation...")
    logger.info(f"Number of games: {num_games}")

    game = SnakeGame()
    scores = []
    steps_list = []

    if agent is None:
        logger.error("No agent provided for evaluation.")
        return None, []

    agent.epsilon = 0  # Turn off exploration
    agent.policy_net.eval()  # Set model to evaluation mode

    try:
        with tqdm(
            range(num_games),
            desc="Evaluating Agent",
            unit="game",
            dynamic_ncols=True,
            colour="green",
        ) as pbar:
            for i in pbar:
                state = game.reset()
                steps = 0
                done = False

                while not done:
                    # Convert state numpy array to tensor for the agent
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = agent.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()

                    next_state, _, done = game.step(action)
                    state = next_state
                    steps += 1

                    # Add step limit for evaluation robustness
                    if steps > (GRID_WIDTH * GRID_HEIGHT * 3):
                        logger.warning("Evaluation game %d exceeded max steps.", i + 1)
                        break  # Stop this game

                scores.append(game.score)
                steps_list.append(steps)

                avg_score_eval = mean(scores) if scores else 0
                max_score_eval = max(scores) if scores else 0
                avg_steps_eval = mean(steps_list) if steps_list else 0
                pbar.set_postfix(
                    {
                        "Last Score": game.score,
                        "Avg": f"{avg_score_eval:.2f}",
                        "Max": max_score_eval,
                        "Avg Steps": f"{avg_steps_eval:.1f}",
                    }
                )

        # Calculate final results
        avg_score_final = mean(scores) if scores else 0
        max_score_final = max(scores) if scores else 0
        avg_steps_final = mean(steps_list) if steps_list else 0

        print("\n------ Evaluation Results (%d games) ------" % num_games)
        print("Average Score: %.2f" % avg_score_final)
        print("Max Score: %d" % max_score_final)
        print("Average Steps per Game: %.1f" % avg_steps_final)
        print("-" * 40)

        return avg_score_final, scores

    except Exception as e:
        logger.error("An error occurred during evaluation: %s", e, exc_info=True)
        return None, scores  # Return None for avg score on error

    finally:
        agent.policy_net.train()
