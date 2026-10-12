from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dataset import SceneGrid, read_image
from .rewards import RewardConfig, reward_terms
from .state import Encoder, StateConfig

@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 200
    action_fraction: float = 0.1
    flicker_threshold: float = 0.01
    convergence_patience: int = 3
    brightness_tolerance: float = 0.1
    outward_clip_penalty: float = 0.0
    terminate_on_success: bool = False

    def __post_init__(self):
        if self.max_steps < 1 or self.convergence_patience < 1 or (not 0 < self.action_fraction <= 1):
            raise ValueError('Invalid episode budget, patience or action fraction')
        if not 0 <= self.flicker_threshold <= 1 or not 0 <= self.brightness_tolerance <= 0.5:
            raise ValueError('Invalid convergence thresholds')
        if not np.isfinite(self.outward_clip_penalty) or self.outward_clip_penalty < 0:
            raise ValueError('outward_clip_penalty must be finite and nonnegative')
        if not isinstance(self.terminate_on_success, bool):
            raise ValueError('terminate_on_success must be boolean')

@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = True
    easy_until: int = 25000
    normal_until: int = 45000
    final_probabilities: tuple[float, ...] = (0.1, 0.4, 0.5)
    reset_candidates: int = 8
    lighting_proxy: str = 'exposure_lookup'
    change_every: int = 20
    illumination_range: tuple[float, ...] = (0.25, 4.0)

    def __post_init__(self):
        if not 0 <= self.easy_until < self.normal_until or self.reset_candidates < 1 or self.change_every < 1:
            raise ValueError('Invalid curriculum thresholds')
        probabilities = np.asarray(self.final_probabilities)
        if probabilities.shape != (3,) or np.any(probabilities < 0) or (not np.isfinite(probabilities).all()) or (not np.isclose(probabilities.sum(), 1)):
            raise ValueError('Curriculum probabilities must sum to one')
        if self.lighting_proxy not in ('none', 'exposure_lookup'):
            raise ValueError('Unknown lighting proxy')
        low, high = self.illumination_range
        if not 0 < low <= 1 <= high or not np.isfinite([low, high]).all():
            raise ValueError('Invalid illumination range')

    def probabilities(self, step: int) -> np.ndarray:
        if step < self.easy_until:
            return np.array([1.0, 0.0, 0.0])
        if step < self.normal_until:
            fraction = (step - self.easy_until) / (self.normal_until - self.easy_until)
            return np.array([1 - fraction, fraction, 0.0])
        return np.asarray(self.final_probabilities, dtype=float)

class ExposureEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scene: SceneGrid, state: StateConfig | None=None, reward: RewardConfig | None=None, config: EnvConfig | None=None, curriculum: CurriculumConfig | None=None):
        self.scene = scene
        self.encoder = Encoder(state or StateConfig())
        self.reward_config = reward or RewardConfig()
        self.config = config or EnvConfig()
        self.curriculum = curriculum or CurriculumConfig()
        self.state_dim = self.encoder.state_dim + (2 if self.encoder.config.include_controls else 0)
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, (self.state_dim,), dtype=np.float32)
        self.active = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.training = bool(options.get('training', False))
        self.encoder.reset(self.np_random, augment=self.training)
        if self.training and self.curriculum.enabled:
            self.level = int(self.np_random.choice(3, p=self.curriculum.probabilities(int(options.get('training_step', 0)))))
        else:
            self.level = -1
        self.position = self.scene.random_setting(self.np_random)
        if 'exposure_us' in options or 'gain_db' in options:
            if not {'exposure_us', 'gain_db'} <= options.keys():
                raise ValueError('Provide both exposure_us and gain_db')
            self.position = np.array([options['exposure_us'], options['gain_db']], dtype=float)
            if not np.isfinite(self.position).all() or np.any(self.position < self.scene.low) or np.any(self.position > self.scene.high):
                raise ValueError('Initial parameters are outside measured bounds')
        elif self.level in (0, 1):
            candidates = []
            for _ in range(self.curriculum.reset_candidates):
                candidate = self.scene.random_setting(self.np_random)
                _, _, path = self.scene.nearest(*candidate)
                intensity = float(self.encoder.roi(read_image(path)).mean())
                candidates.append((abs(intensity - 0.5), candidate))
            self.position = (min if self.level == 0 else max)(candidates, key=lambda item: item[0])[1].copy()
        self.steps = self.stable_steps = 0
        self.illumination = 1.0
        self.last_roi = self.previous_roi = None
        self.active = True
        observation, info = self._observe()
        return (observation, info)

    def _observe(self):
        exposure, gain = self.position
        effective = float(np.clip(exposure * self.illumination, self.scene.low[0], self.scene.high[0]))
        measured_exposure, measured_gain, path = self.scene.nearest(effective, gain)
        roi = self.encoder.roi(read_image(path))
        self.previous_roi, self.last_roi = (self.last_roi, roi)
        terms = reward_terms(roi, self.previous_roi, self.reward_config)
        observation = self.encoder.encode_roi(roi)
        if self.encoder.config.include_controls:
            controls = ((self.position - self.scene.low) / (self.scene.high - self.scene.low)).astype(np.float32)
            observation = np.concatenate([observation, controls])
        well_exposed = terms['brightness_error'] <= self.config.brightness_tolerance
        info = {'scene': self.scene.name, 'step': self.steps, 'exposure_us': float(exposure), 'gain_db': float(gain), 'measured_exposure_us': measured_exposure, 'measured_gain_db': measured_gain, 'source_filename': path.name, 'illumination_proxy': self.illumination, 'curriculum_level': self.level, 'well_exposed': bool(well_exposed), 'exposure_at_boundary': bool(measured_exposure in (self.scene.low[0], self.scene.high[0])), 'gain_at_boundary': bool(measured_gain in (self.scene.low[1], self.scene.high[1])), **terms}
        info['at_boundary'] = info['exposure_at_boundary'] or info['gain_at_boundary']
        return (observation, info)

    def step(self, action):
        if not self.active:
            raise RuntimeError('Call reset before stepping a new episode')
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Expected two finite actions in [-1,1], ordered exposure then gain')
        requested_delta = action.astype(float) * (self.scene.high - self.scene.low) * self.config.action_fraction
        requested_position = self.position + requested_delta
        self.position = np.clip(requested_position, self.scene.low, self.scene.high)
        clipped = requested_position - self.position
        step_scale = (self.scene.high - self.scene.low) * self.config.action_fraction
        clipped_fraction = float(np.mean(np.abs(clipped) / step_scale))
        self.steps += 1
        if self.level == 2 and self.curriculum.lighting_proxy == 'exposure_lookup' and (self.steps % self.curriculum.change_every == 0):
            low, high = self.curriculum.illumination_range
            self.illumination = float(np.exp(self.np_random.uniform(np.log(low), np.log(high))))
            self.stable_steps = 0
        observation, info = self._observe()
        penalty = self.config.outward_clip_penalty * clipped_fraction
        info['reward'] -= penalty
        info.update(reward_clip=-penalty, action_clipped=bool(np.any(np.abs(clipped) > 1e-08)), clipped_fraction=clipped_fraction, requested_delta_exposure_us=float(requested_delta[0]), requested_delta_gain_db=float(requested_delta[1]))
        self.stable_steps = self.stable_steps + 1 if info['flicker'] <= self.config.flicker_threshold else 0
        info.update(converged=self.stable_steps >= self.config.convergence_patience, success=bool(self.stable_steps >= self.config.convergence_patience and info['well_exposed']), action_exposure=float(action[0]), action_gain=float(action[1]))
        terminated = bool(self.config.terminate_on_success and info['success'])
        truncated = self.steps >= self.config.max_steps and (not terminated)
        self.active = not (terminated or truncated)
        return (observation, info['reward'], terminated, truncated, info)
