# dqn_agent.py
"""Defines the Advanced DQNAgent class using Dueling, PER, and Multi-step."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

from dqn_model import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class AdvancedDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        lr=0.00025,  # Learning rate
        gamma=0.99,  # Discount factor
        epsilon_start=1.0,  # Initial exploration rate
        epsilon_min=0.01,  # Minimum exploration rate
        epsilon_decay_steps=30000,
        buffer_size=50000,  # PER buffer size
        batch_size=64,  # Batch size
        target_update=1500,  # Target network update frequency
        multi_step_n=3,  # Number of steps for multi-step learning
        per_alpha=0.6,  # PER alpha
        per_beta_start=0.4,  # PER beta initial value
        per_beta_frames=50000,  # Steps over which beta anneals to 1
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.lr = lr
        self.multi_step_n = multi_step_n

        logger.info("Initializing AdvancedDQNAgent (Dueling + PER + Multi-step):")
        logger.info(
            f"State Size: {state_size}, Action Size: {action_size}, Device: {self.device}"
        )
        logger.info(f"LR: {self.lr}, Gamma: {self.gamma}, N-step: {self.multi_step_n}")
        logger.info(
            f"Epsilon Start: {self.epsilon}, Min: {self.epsilon_min}, Decay Steps: {self.epsilon_decay_steps}"
        )
        logger.info(
            f"Buffer Size: {buffer_size}, Batch Size: {self.batch_size}, Target Update: {self.target_update}"
        )
        logger.info(
            f"PER Alpha: {per_alpha}, PER Beta Start: {per_beta_start}, Beta Frames: {per_beta_frames}"
        )

        # Networks (Using DuelingDQN)
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=max(1, epsilon_decay_steps // 10), gamma=0.99
        )

        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.memory.alpha = per_alpha
        self.memory.beta = per_beta_start
        # Calculate beta increment based on total frames/steps for annealing
        self.beta_increment = (
            (1.0 - per_beta_start) / per_beta_frames if per_beta_frames > 0 else 0
        )

        # Temporary storage for multi-step transitions
        self.n_step_buffer = deque(maxlen=self.multi_step_n)
        self.steps_done = 0

    def _store_multi_step(self):
        """Processes the n_step_buffer to store a multi-step transition in PER."""
        if len(self.n_step_buffer) < self.multi_step_n:
            return  # Not enough steps yet

        # Get the state, action from the beginning of the buffer
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        # Calculate accumulated reward and find the final state
        multi_step_reward = 0
        last_state = None
        is_terminal = False
        for i in range(self.multi_step_n):
            _, _, r, s_next, done = self.n_step_buffer[i]
            multi_step_reward += (self.gamma**i) * r
            last_state = s_next
            if done:
                is_terminal = True
                break  # Stop accumulating if episode ended

        # The transition to store: (start_state, start_action, n_step_reward, n_step_final_state, n_step_terminal)
        multi_step_transition = (
            first_state,
            first_action,
            multi_step_reward,
            last_state,
            is_terminal,
        )

        # --- Calculate initial priority (TD error) ---
        # We need an estimate of the TD error to add to PER.
        # We can do a quick calculation here based on the *current* networks.
        # It's an approximation but better than max priority.
        with torch.no_grad():
            state_t = torch.FloatTensor(first_state).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)

            # Use Double DQN logic for target Q value calculation
            next_q_policy = self.policy_net(next_state_t)
            best_next_action = next_q_policy.max(1)[1].unsqueeze(1)
            next_q_target = self.target_net(next_state_t).gather(1, best_next_action)

            target_q = (
                multi_step_reward
                + (1 - float(is_terminal))
                * (self.gamma**self.multi_step_n)
                * next_q_target.item()
            )

            current_q = (
                self.policy_net(state_t)
                .gather(1, torch.LongTensor([[first_action]]).to(self.device))
                .item()
            )
            td_error = target_q - current_q

        # Add to PER buffer with calculated priority
        self.memory.add(td_error, multi_step_transition)

    def store(self, state, action, reward, next_state, done):
        """Stores experience in temporary buffer and processes for multi-step."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If buffer has enough steps, process the oldest transition for PER storage
        if len(self.n_step_buffer) >= self.multi_step_n:
            self._store_multi_step()

        # If the episode ended, clear the multi-step buffer by processing remaining steps
        if done:
            while len(self.n_step_buffer) > 0:
                # Process remaining steps, adjusting n for shorter sequences at the end
                temp_n = len(self.n_step_buffer)
                first_state, first_action, _, _, _ = self.n_step_buffer[0]
                multi_step_reward = 0
                last_state = None
                is_terminal = False
                for i in range(temp_n):
                    _, _, r, s_next, d = self.n_step_buffer[i]
                    multi_step_reward += (self.gamma**i) * r
                    last_state = s_next
                    if d:
                        is_terminal = True
                        break

                multi_step_transition = (
                    first_state,
                    first_action,
                    multi_step_reward,
                    last_state,
                    is_terminal,
                )
                # Calculate priority as before
                with torch.no_grad():
                    state_t = (
                        torch.FloatTensor(first_state).unsqueeze(0).to(self.device)
                    )
                    next_state_t = (
                        torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    )
                    next_q_policy = self.policy_net(next_state_t)
                    best_next_action = next_q_policy.max(1)[1].unsqueeze(1)
                    next_q_target = self.target_net(next_state_t).gather(
                        1, best_next_action
                    )
                    target_q = (
                        multi_step_reward
                        + (1 - float(is_terminal))
                        * (self.gamma**temp_n)
                        * next_q_target.item()
                    )
                    current_q = (
                        self.policy_net(state_t)
                        .gather(1, torch.LongTensor([[first_action]]).to(self.device))
                        .item()
                    )
                    td_error = target_q - current_q
                self.memory.add(td_error, multi_step_transition)

                # Remove the processed step from the front
                self.n_step_buffer.popleft()
            # Ensure buffer is fully cleared after episode end
            self.n_step_buffer.clear()

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def train(self):
        """Samples from PER, computes loss with IS weights and n-step targets, and updates."""
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples

        # Sample batch from PER buffer
        # batch contains (state, action, n_reward, n_next_state, n_done) tuples
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

        # Unzip batch
        states, actions, n_rewards, n_next_states, n_dones = zip(*mini_batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        n_rewards = torch.FloatTensor(np.array(n_rewards)).unsqueeze(1).to(self.device)
        n_next_states = torch.FloatTensor(np.array(n_next_states)).to(self.device)
        n_dones = torch.FloatTensor(np.array(n_dones)).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # --- Calculate Target Q values (using n-step rewards and Double DQN principle) ---
        with torch.no_grad():
            # Select best actions for n_next_states using the policy network
            next_q_policy = self.policy_net(n_next_states)
            best_next_actions = next_q_policy.max(1)[1].unsqueeze(1)

            # Evaluate these actions using the target network
            next_q_target = self.target_net(n_next_states).gather(1, best_next_actions)

            # Calculate the n-step target value
            # target = n_reward + (1 - n_done) * (gamma^n) * Q_target(s_n, argmax_a Q_policy(s_n, a))
            target_q_values = (
                n_rewards
                + (1 - n_dones) * (self.gamma**self.multi_step_n) * next_q_target
            )

        # --- Calculate Current Q values ---
        current_q_values = self.policy_net(states).gather(1, actions)

        # --- Calculate Loss (weighted by IS weights) ---
        td_errors = target_q_values - current_q_values
        loss = nn.SmoothL1Loss(reduction="none")(current_q_values, target_q_values)
        weighted_loss = (is_weights * loss).mean()  # Apply IS weights

        # --- Optimization ---
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        # --- Update Priorities in PER Buffer ---
        abs_td_errors = td_errors.abs().detach().cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, abs_td_errors[i][0])

        # --- Epsilon Decay and Target Network Update ---
        self._update_epsilon_and_target()

        return weighted_loss.item()

    def _update_epsilon_and_target(self):
        """Handles epsilon decay and target network updates."""
        # Linear epsilon decay
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (
                self.epsilon_start - self.epsilon_min
            ) * (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min

        # Anneal PER beta
        self.memory.beta = min(1.0, self.memory.beta + self.beta_increment)

        # Update target network
        if self.steps_done % self.target_update == 0:
            logger.info("Updating target network at step %d", self.steps_done)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
