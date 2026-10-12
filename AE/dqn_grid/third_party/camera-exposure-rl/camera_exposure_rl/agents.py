from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from .models import QNetwork
from .replay_buffer import Batch, ReplayBuffer

@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    gamma: float = 0.99
    lr: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 100000
    warmup_steps: int = 1000
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    grad_clip: float = 10.0

class RandomAgent:

    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self.n_actions = n_actions
        self._rng = rng

    def act(self, state: np.ndarray, *, greedy: bool=False) -> int:
        return int(self._rng.integers(self.n_actions))

class DQNAgent:

    def __init__(self, config: DQNConfig, device: torch.device, rng: np.random.Generator) -> None:
        self.config = config
        self.device = device
        self._rng = rng
        self._step = 0
        self.q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_sizes).to(device)
        self.target_net = copy.deepcopy(self.q_net).eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(capacity=config.buffer_size, state_dim=config.state_dim, rng=rng)

    @property
    def epsilon(self) -> float:
        c = self.config
        progress = min(1.0, self._step / max(1, c.epsilon_decay_steps))
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress

    def act(self, state: np.ndarray, *, greedy: bool=False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.config.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s)
            return int(q.argmax(dim=-1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict[str, float] | None:
        self._step += 1
        if len(self.buffer) < self.config.batch_size or self._step < self.config.warmup_steps:
            return None
        batch = self.buffer.sample(self.config.batch_size)
        loss, q_mean = self._compute_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if self._step % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {'loss': float(loss.item()), 'q_mean': float(q_mean), 'epsilon': self.epsilon}

    def _compute_loss(self, batch: Batch) -> tuple[torch.Tensor, float]:
        device = self.device
        states = torch.as_tensor(batch.states, device=device)
        next_states = torch.as_tensor(batch.next_states, device=device)
        actions = torch.as_tensor(batch.actions, device=device)
        rewards = torch.as_tensor(batch.rewards, device=device)
        dones = torch.as_tensor(batch.dones, device=device)
        q_all = self.q_net(states)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            target = rewards + self.config.gamma * q_next * (1.0 - dones)
        loss = F.smooth_l1_loss(q_taken, target)
        return (loss, q_all.mean().item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'q_net': self.q_net.state_dict(), 'config': self.config.__dict__, 'step': self._step}, path)

    @classmethod
    def load(cls, path: Path, device: torch.device, rng: np.random.Generator) -> 'DQNAgent':
        ckpt = torch.load(path, map_location=device)
        config = DQNConfig(**ckpt['config'])
        agent = cls(config, device, rng)
        agent.q_net.load_state_dict(ckpt['q_net'])
        agent.target_net.load_state_dict(ckpt['q_net'])
        agent._step = int(ckpt.get('step', 0))
        return agent
