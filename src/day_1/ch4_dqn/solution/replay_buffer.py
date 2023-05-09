"""
Replay Buffer.

Reference:
    - https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
    - https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
