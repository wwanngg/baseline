from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .dataset import SceneGrid
from .rewards import RewardConfig, combined_reward
from .state import IntensityHistoryEncoder, StateConfig
ACTION_NOOP = 0
ACTION_ISO_DOWN = 1
ACTION_ISO_UP = 2
ACTION_EXP_DOWN = 3
ACTION_EXP_UP = 4
N_ACTIONS = 5

@dataclass
class EnvConfig:
    max_episode_steps: int = 30
    early_termination: bool = True
    convergence_threshold: float = 0.04
    convergence_patience: int = 3

class ExposureEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scene: SceneGrid, state_config: StateConfig, reward_config: RewardConfig, env_config: EnvConfig) -> None:
        super().__init__()
        self.scene = scene
        self.state_config = state_config
        self.reward_config = reward_config
        self.env_config = env_config
        self.encoder = IntensityHistoryEncoder(state_config)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.encoder.state_dim,), dtype=np.float32)
        self._iso_idx: int = 0
        self._exp_idx: int = 0
        self._step_count: int = 0
        self._consecutive_converged: int = 0
        self._prev_roi: np.ndarray | None = None

    def reset(self, *, seed: int | None=None, options: dict[str, Any] | None=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        self._iso_idx = int(options.get('iso_idx', rng.integers(self.scene.spec.n_iso)))
        self._exp_idx = int(options.get('exp_idx', rng.integers(self.scene.spec.n_exp)))
        self._step_count = 0
        self._consecutive_converged = 0
        self._prev_roi = None
        self.encoder.reset()
        observation, _, _ = self._observe()
        return (observation, self._info())

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f'Invalid action {action!r}; expected 0..{N_ACTIONS - 1}')
        self._apply_action(int(action))
        self._step_count += 1
        observation, reward, components = self._observe()
        terminated = self._check_convergence(components['r_mean'])
        truncated = self._step_count >= self.env_config.max_episode_steps
        info = self._info()
        info.update(components)
        info['action'] = int(action)
        return (observation, reward, terminated, truncated, info)

    def _apply_action(self, action: int) -> None:
        n_iso = self.scene.spec.n_iso
        n_exp = self.scene.spec.n_exp
        if action == ACTION_ISO_DOWN:
            self._iso_idx = max(0, self._iso_idx - 1)
        elif action == ACTION_ISO_UP:
            self._iso_idx = min(n_iso - 1, self._iso_idx + 1)
        elif action == ACTION_EXP_DOWN:
            self._exp_idx = max(0, self._exp_idx - 1)
        elif action == ACTION_EXP_UP:
            self._exp_idx = min(n_exp - 1, self._exp_idx + 1)

    def _observe(self) -> tuple[np.ndarray, float, dict[str, float]]:
        frame = self.scene.load(self._iso_idx, self._exp_idx)
        observation = self.encoder.encode(frame)
        roi = self.encoder.last_roi(frame)
        reward, components = combined_reward(roi, self._prev_roi, self.reward_config)
        self._prev_roi = roi
        return (observation, reward, components)

    def _check_convergence(self, brightness_penalty: float) -> bool:
        if not self.env_config.early_termination:
            return False
        if brightness_penalty <= self.env_config.convergence_threshold:
            self._consecutive_converged += 1
        else:
            self._consecutive_converged = 0
        return self._consecutive_converged >= self.env_config.convergence_patience

    def _info(self) -> dict[str, Any]:
        return {'scene': self.scene.name, 'iso_idx': self._iso_idx, 'exp_idx': self._exp_idx, 'step': self._step_count}

def make_env(scene_dir: Path, side: str, state_config: StateConfig, reward_config: RewardConfig, env_config: EnvConfig) -> ExposureEnv:
    scene = SceneGrid(scene_dir, side=side)
    return ExposureEnv(scene, state_config, reward_config, env_config)
