"""Stores constants used across the project."""

from time import time_ns

GRID_WIDTH = 80
GRID_HEIGHT = 20

# Action definitions
# 0: straight, 1: left turn, 2: right turn
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_SIZE = 3  # Total number of actions

# Reward shaping values
REWARD_FOOD = 10.0
REWARD_DIE = -10.0
REWARD_CLOSER = 0.1
REWARD_FARTHER = -0.2

# State size (Update if get_state() changes)
# Danger (straight, left, right), Direction (4), Food location (4), Dist (2)
STATE_SIZE = 3 + 4 + 4 + 2

# Logging configuration
LOG_FILE = f"snake_dqn_training_{time_ns()}.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s - [Process:%(process)d Thread:%(threadName)s]"

# GUI configuration
COLOR_BACKGROUND = (40, 40, 40)
COLOR_SNAKE_BODY = (0, 200, 0)
COLOR_SNAKE_HEAD = (0, 255, 0)
COLOR_FOOD = (220, 0, 0)
COLOR_TEXT = (230, 230, 230)
COLOR_GAMEOVER = (255, 60, 60)

CELL_SIZE = 20
SCORE_AREA_HEIGHT = 30

GRID_WIDTH_H = GRID_WIDTH
GRID_HEIGHT_H = GRID_HEIGHT + GRID_HEIGHT * .9
WINDOW_WIDTH = GRID_WIDTH_H * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT_H * CELL_SIZE + SCORE_AREA_HEIGHT

CELL_RADIUS = CELL_SIZE // 2
OFFSET = CELL_SIZE // 2