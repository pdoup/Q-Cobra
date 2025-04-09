# replay_buffer.py
"""Prioritized Experience Replay Buffer using a SumTree."""

import random
import numpy as np


class SumTree:
    """
    Simple SumTree implementation for prioritized sampling.
    Stores priorities and allows efficient sampling based on them.
    """

    write = 0  # Current index in the circular buffer part

    def __init__(self, capacity):
        self.capacity = capacity  # Max number of items (transitions)
        # Tree structure: Stores priorities. Leaf nodes are priorities, parents are sums.
        # Size is 2*capacity - 1 (leaves + internal nodes)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Data structure: Stores the actual transitions corresponding to leaf nodes.
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0  # Current number of entries in the buffer

    def _propagate(self, idx, change):
        """Update sum upwards from leaf node."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample leaf index given a cumulative sum 's'."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # Reached leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Total sum of priorities."""
        return self.tree[0]

    def add(self, p, data):
        """Add priority 'p' and associated data 'data'."""
        idx = self.write + self.capacity - 1  # Get leaf index in the tree

        self.data[self.write] = data  # Store data in circular buffer part
        self.update(idx, p)  # Update priority in tree

        self.write += 1
        if self.write >= self.capacity:  # Wrap around if capacity reached
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority 'p' for leaf index 'idx'."""
        change = p - self.tree[idx]  # Calculate change in priority
        self.tree[idx] = p  # Set new priority
        self._propagate(idx, change)  # Propagate change up the tree

    def get(self, s):
        """Get leaf index, priority, and data for a cumulative sum 's'."""
        idx = self._retrieve(0, s)  # Find leaf index based on cumulative priority 's'
        dataIdx = idx - self.capacity + 1  # Map tree leaf index back to data index
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Uses a SumTree for efficient prioritized sampling.
    """

    epsilon = 0.01  # Small amount to avoid zero priority
    alpha = 0.6  # [0~1] converts TD error to priority (0 = uniform, 1 = full priority)
    beta = 0.4  # [0~1] importance sampling exponent (starts low, anneals to 1)
    beta_increment_per_sampling = 0.0001  # Rate at which beta anneals
    abs_err_upper = 1.0  # Clipped abs error (max priority)

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        """Calculate priority from TD error."""
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        """Add a new sample with its initial TD error."""
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        """Sample a batch of transitions based on priority."""
        batch = []
        idxs = []  # Tree indices
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min(
            [1.0, self.beta + self.beta_increment_per_sampling]
        )  # Anneal beta

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # Sample within segment
            (idx, p, data) = self.tree.get(s)

            if data == 0:  # Should not happen if buffer is reasonably full
                # Fallback: sample uniformly if data is missing (e.g., buffer just started)
                idx, p, data = self.tree.get(random.uniform(0, self.tree.total()))
                if data == 0:
                    continue  # Skip if still no valid data

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()  # Normalize weights

        return batch, idxs, is_weight

    def update(self, idx, error):
        """Update the priority of a transition after it has been used in training."""
        p = self._get_priority(error)
        clipped_p = np.minimum(p, self.abs_err_upper)  # Clip priority
        self.tree.update(idx, clipped_p)

    def __len__(self):
        return self.tree.n_entries
