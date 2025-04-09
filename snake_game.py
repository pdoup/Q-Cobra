# snake_game.py
"""Defines the Snake game environment."""

import random
import numpy as np
import time

from constants import (
    GRID_WIDTH,
    GRID_HEIGHT,
    ACTION_LEFT,
    ACTION_RIGHT,
    REWARD_CLOSER,
    REWARD_DIE,
    REWARD_FARTHER,
    REWARD_FOOD,
)


class SnakeGame:
    def __init__(self):
        self.state_size = -1
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)  # Start moving right
        self.food = self.spawn_food()
        self.score = 0
        self.done = False
        state = self.get_state()
        self.state_size = len(state)  # Determine state size dynamically
        return state

    def spawn_food(self):
        while True:
            food = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            if food not in self.snake:
                return food

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dist_x = food_x - head_x
        dist_y = food_y - head_y

        # Danger signals
        danger_straight = self._will_collide(
            (head_x + self.direction[0], head_y + self.direction[1])
        )
        dir_left = (-self.direction[1], self.direction[0])
        dir_right = (self.direction[1], -self.direction[0])
        danger_left = self._will_collide((head_x + dir_left[0], head_y + dir_left[1]))
        danger_right = self._will_collide(
            (head_x + dir_right[0], head_y + dir_right[1])
        )

        state = [
            # Danger signals (binary)
            int(danger_straight),
            int(danger_left),
            int(danger_right),
            # Direction (one-hot encoded)
            int(self.direction == (1, 0)),
            int(self.direction == (-1, 0)),
            int(self.direction == (0, -1)),
            int(self.direction == (0, 1)),
            # Food location relative to head (binary)
            int(food_x < head_x),
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
            # Normalized distance
            dist_x / GRID_WIDTH,
            dist_y / GRID_HEIGHT,
        ]
        return np.array(state, dtype=np.float32)

    def _will_collide(self, pos):
        """Internal helper for collision check."""
        x, y = pos
        # Check boundaries or collision with self (excluding the current head position)
        return (
            x < 0
            or x >= GRID_WIDTH
            or y < 0
            or y >= GRID_HEIGHT
            or pos in self.snake[1:]
        )

    def step(self, action):
        original_direction = self.direction
        if action == ACTION_LEFT:
            self.direction = (-original_direction[1], original_direction[0])
        elif action == ACTION_RIGHT:
            self.direction = (original_direction[1], -original_direction[0])

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Calculate distances for reward shaping
        old_dist = abs(self.food[0] - head_x) + abs(self.food[1] - head_y)
        new_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])

        # Check for collision BEFORE moving
        if self._will_collide(new_head):
            self.done = True
            reward = REWARD_DIE
            # Return current state before crash
            return self.get_state(), reward, self.done

        # No collision, proceed with move
        self.snake.insert(0, new_head)

        # Check for food eating AFTER moving
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward = REWARD_FOOD
        else:
            self.snake.pop()  # Remove tail segment if no food eaten
            # Reward based on distance change
            if new_dist < old_dist:
                reward = REWARD_CLOSER
            else:
                # Penalize moving away or staying same distance (e.g., moving parallel)
                reward = REWARD_FARTHER

        return self.get_state(), reward, self.done

    def render(self, delay=0.01):
        grid = [[" " for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        # Draw snake body first
        for x, y in self.snake[1:]:
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                grid[y][x] = "s"
        # Draw snake head
        hx, hy = self.snake[0]
        if 0 <= hy < GRID_HEIGHT and 0 <= hx < GRID_WIDTH:
            grid[hy][hx] = "S"
        # Draw food
        fx, fy = self.food
        if 0 <= fy < GRID_HEIGHT and 0 <= fx < GRID_WIDTH:
            grid[fy][fx] = "F"

        # Clear console (works better on Linux/macOS terminals)
        print("\033[H\033[J", end="")
        print(f"Score: {self.score}")
        print("+" + "-" * GRID_WIDTH + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * GRID_WIDTH + "+")
        if delay > 0:
            time.sleep(delay)
