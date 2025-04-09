# dqn_model.py
"""Defines the DQN network architecture."""

import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size  # Number of actions

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        # State Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),  # Outputs a single value for the state
        )

        # Action Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),  # Outputs advantage for each action
        )

    def forward(self, x):
        # Ensure input is flattened if needed (should be for vector state)
        features = self.feature_layer(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine V and A streams to get Q values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)
