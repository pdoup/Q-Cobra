# play_game.py
"""
Loads a trained DQN agent and runs the Snake game visually in the terminal.
Provides statistics after the games are played.
"""

import torch
import time
import argparse
import os
from statistics import mean, median, stdev

# --- Import necessary components from project files ---
try:
    from snake_game import SnakeGame
    from dqn_agent import AdvancedDQNAgent
    from constants import STATE_SIZE, ACTION_SIZE, GRID_HEIGHT, GRID_WIDTH
except ImportError as e:
    print(f"Error importing project files: {e}")
    print(
        "Please ensure constants.py, snake_game.py, dqn_model.py, and dqn_agent.py are in the same directory or Python path."
    )
    exit(1)
# --- End Imports ---


def play(model_path, num_games, delay):
    """Loads the agent and plays the specified number of games."""

    print(f"--- Snake Game Playback ---")
    print(f"Loading model from: {model_path}")
    print(f"Number of games: {num_games}")
    print(f"Render delay: {delay} seconds")
    print("-" * 30)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent structure
    agent = AdvancedDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        device=device,
        epsilon_start=0,
        epsilon_min=0,
    )  # Epsilon forced to 0

    # Load the trained weights
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- Game Play Loop ---
    game = SnakeGame()
    all_scores = []
    all_steps = []

    for i in range(num_games):
        print(f"\n--- Starting Game {i + 1} of {num_games} ---")
        state = game.reset()
        steps = 0
        done = False
        max_steps_per_game = GRID_WIDTH * GRID_HEIGHT * 3  # Safety limit

        while not done:
            # Render the current game state
            game.render(delay=delay)

            # Agent chooses action based on learned policy (epsilon=0)
            action = agent.get_action(state)

            # Perform action in the game
            next_state, _, done = game.step(action)

            # Update state
            state = next_state
            steps += 1

            # Check safety limit
            if steps >= max_steps_per_game:
                print(
                    f"Warning: Game {i + 1} reached max step limit ({max_steps_per_game}). Ending game."
                )
                done = True  # Force end

        # Final render after game ends
        game.render(delay=delay)
        time.sleep(0.5)  # Pause slightly after game over

        # Record results for this game
        score = game.score
        all_scores.append(score)
        all_steps.append(steps)

    # --- Final Statistics ---
    print("\n" + "=" * 30)
    print(f"--- Overall Statistics ({num_games} Games) ---")
    print(f"Scores: {all_scores}")
    if all_scores:
        print(f"Max Score:   {max(all_scores)}")
        print(f"Min Score:   {min(all_scores)}")
        print(f"Avg Score:   {mean(all_scores):.2f}")
        if len(all_scores) > 1:
            print(f"Median Score:{median(all_scores):.2f}")
            print(f"Std Dev Score:{stdev(all_scores):.2f}")
        print("-" * 15)
        print(f"Avg Steps:   {mean(all_steps):.1f}")
    else:
        print("No games were completed.")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Snake using a trained DQN agent."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="snake_dqn_best.pth",
        help="Path to the saved model state dictionary (.pth file).",
    )
    parser.add_argument(
        "--num-games", type=int, default=5, help="Number of games to play."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between game steps in seconds for rendering.",
    )

    args = parser.parse_args()

    play(args.model_path, args.num_games, args.delay)
