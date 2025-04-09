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
