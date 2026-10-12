from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

class ReplayBuffer:

    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros_like(self._states)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0
        self._rng = rng

    def __len__(self) -> int:
        return self._size

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self._cursor
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = float(done)
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size < batch_size:
            raise ValueError(f'Cannot sample {batch_size} transitions from buffer of size {self._size}')
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(states=self._states[idx], actions=self._actions[idx], rewards=self._rewards[idx], next_states=self._next_states[idx], dones=self._dones[idx])
