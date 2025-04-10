import torch
import time
import argparse
import os
import sys
import pygame
import math
from statistics import mean, median, stdev

try:
    from snake_game import SnakeGame
    from dqn_agent import AdvancedDQNAgent
    from constants import (
        STATE_SIZE,
        ACTION_SIZE,
        CELL_SIZE,
        SCORE_AREA_HEIGHT,
        GRID_WIDTH_H,
        GRID_HEIGHT_H,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        COLOR_SNAKE_BODY,
        COLOR_SNAKE_HEAD,
        COLOR_FOOD,
        COLOR_TEXT,
        COLOR_GAMEOVER,
        OFFSET,
        CELL_RADIUS,
    )
except ImportError as e:
    print(f"Error importing project files: {e}")
    sys.exit(1)


def draw_elements(
    screen,
    game,
    font,
    transition_alpha,
    elapsed_time,
    eat_animation=None,
    gameover_scale=1.0,
    gameover_text=None,
):
    surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    surface.set_alpha(transition_alpha)

    for segment in game.snake[1:]:
        center = (segment[0] * CELL_SIZE + OFFSET, segment[1] * CELL_SIZE + OFFSET)
        pygame.draw.circle(surface, COLOR_SNAKE_BODY, center, CELL_RADIUS - 2)

    head = game.snake[0]
    head_center = (head[0] * CELL_SIZE + OFFSET, head[1] * CELL_SIZE + OFFSET)
    pygame.draw.circle(surface, COLOR_SNAKE_HEAD, head_center, CELL_RADIUS)

    food = game.food
    food_center = (food[0] * CELL_SIZE + OFFSET, food[1] * CELL_SIZE + OFFSET)

    pulse_scale = 1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.005)
    pygame.draw.circle(
        surface, (100, 0, 0), food_center, int((CELL_RADIUS - 2) * pulse_scale)
    )
    pygame.draw.circle(
        surface, COLOR_FOOD, food_center, int((CELL_RADIUS - 4) * pulse_scale)
    )

    if eat_animation and eat_animation["active"]:
        scale = eat_animation["scale"]
        scaled_radius = int((CELL_RADIUS - 4) * scale)
        pygame.draw.circle(surface, (255, 80, 80), food_center, scaled_radius)

    hud_rect = pygame.Rect(
        0, WINDOW_HEIGHT - SCORE_AREA_HEIGHT, WINDOW_WIDTH, SCORE_AREA_HEIGHT
    )
    pygame.draw.rect(surface, (30, 30, 30), hud_rect, border_radius=12)

    score_text = font.render(f"Score: {game.score}", True, COLOR_TEXT)
    score_rect = score_text.get_rect(
        midleft=(10, WINDOW_HEIGHT - SCORE_AREA_HEIGHT // 2)
    )
    surface.blit(score_text, score_rect)

    timer_text = font.render(f"Time: {elapsed_time:.1f}s", True, COLOR_TEXT)
    timer_rect = timer_text.get_rect(
        midright=(WINDOW_WIDTH - 10, WINDOW_HEIGHT - SCORE_AREA_HEIGHT // 2)
    )
    surface.blit(timer_text, timer_rect)

    screen.blit(surface, (0, 0))

    if gameover_text:
        scaled_surface = gameover_text.copy()
        w, h = scaled_surface.get_size()
        scaled_surface = pygame.transform.scale(
            scaled_surface, (int(w * gameover_scale), int(h * gameover_scale))
        )
        rect = scaled_surface.get_rect(
            center=(WINDOW_WIDTH // 2, (WINDOW_HEIGHT - SCORE_AREA_HEIGHT) // 2)
        )
        screen.blit(scaled_surface, rect)


def play_gui(model_path, num_games, delay_ms):
    print(f"--- Snake Game GUI Playback ---")
    print(f"Loading model from: {model_path}")
    print(f"Number of games: {num_games}")
    print(f"Step delay: {delay_ms} ms")
    print("-" * 30)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = AdvancedDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        device=device,
        epsilon_start=0,
        epsilon_min=0,
    )

    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    pygame.init()
    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Cobra")
        font = pygame.font.SysFont("Orbitron", 24)
        gameover_font = pygame.font.SysFont("Orbitron", 48, bold=True)
    except pygame.error as e:
        print(f"Error initializing Pygame: {e}")
        return

    game = SnakeGame()
    all_scores = []
    all_steps = []
    running = True

    for i in range(num_games):
        if not running:
            break
        print(f"\n--- Starting Game {i + 1} of {num_games} ---")
        state = game.reset()
        steps = 0
        done = False
        max_steps_per_game = GRID_WIDTH_H * GRID_HEIGHT_H * 4
        start_time = time.time()
        transition_alpha = 0
        eat_animation = {"active": False, "scale": 1.0, "frame": 0}

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("Quit event detected. Exiting...")
            if not running:
                break

            action = agent.get_action(state)
            next_state, reward, done = game.step(action)

            if reward > 0:
                eat_animation["active"] = True
                eat_animation["scale"] = 1.8
                eat_animation["frame"] = 0

            state = next_state
            steps += 1

            transition_alpha = min(transition_alpha + 25, 255)

            if eat_animation["active"]:
                eat_animation["scale"] *= 0.9
                eat_animation["frame"] += 1
                if eat_animation["frame"] > 5:
                    eat_animation["active"] = False

            elapsed_time = time.time() - start_time
            draw_elements(
                screen, game, font, transition_alpha, elapsed_time, eat_animation
            )
            pygame.display.flip()
            pygame.time.wait(delay_ms)

            if steps >= max_steps_per_game:
                print(
                    f"Warning: Game {i + 1} reached max step limit ({max_steps_per_game}). Ending game."
                )
                done = True

        if running:
            score = game.score
            all_scores.append(score)
            all_steps.append(steps)
            print(f"- Game {i + 1} Finished => Score: {score}, Steps: {steps}")

            elapsed_time = time.time() - start_time
            gameover_text = gameover_font.render(
                f"Game Over! Score: {score}", True, COLOR_GAMEOVER
            )

            for scale in [1 + 0.02 * i for i in range(15)]:
                draw_elements(
                    screen,
                    game,
                    font,
                    255,
                    elapsed_time,
                    gameover_text=gameover_text,
                    gameover_scale=scale,
                )
                pygame.display.flip()
                pygame.time.wait(30)

            pygame.time.wait(1000)

    pygame.quit()
    print("\n" + "=" * 30)
    print(f"--- Overall Statistics ({len(all_scores)} Games Completed) ---")
    print(f"Scores: {all_scores}")
    if all_scores:
        print(f"Max Score: {max(all_scores)}")
        print(f"Min Score: {min(all_scores)}")
        print(f"Avg Score: {mean(all_scores):.2f}")
        if len(all_scores) > 1:
            print(f"Median Score: {median(all_scores):.2f}")
            print(f"Std Dev Score: {stdev(all_scores):.2f}")
        print("-" * 20)
        print(f"Avg Steps: {mean(all_steps):.0f}")
    else:
        print("No games were completed.")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Snake using a trained DQN agent with Pygame GUI."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/snake_dqn_advanced_best.pth",
        help="Path to the saved model state dictionary (.pth file).",
    )
    parser.add_argument(
        "--num-games", type=int, default=5, help="Number of games to play."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=70,
        help="Delay between game steps in milliseconds.",
    )

    args = parser.parse_args()
    play_gui(args.model_path, args.num_games, args.delay)
