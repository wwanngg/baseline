from dataclasses import dataclass
import numpy as np
from .estimator import INITIAL

@dataclass
class Rollout:
    illuminant: np.ndarray
    initial_illuminant: np.ndarray
    trace: list
    stop_reason: str
    steps: int

class Controller:

    def __init__(self, estimator):
        self.estimator = estimator

    def observation(self):
        return np.concatenate((self.estimator.feature, self.history_action.flatten(), np.array([self.current_step], dtype=np.float32))).astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.parameters = np.array(INITIAL[self.estimator.model_name], dtype=np.float32)
        self.history_illum = np.zeros((5, 3), dtype=np.float32)
        self.history_action = np.zeros((5, 2), dtype=np.float32)
        estimate = self.estimator.estimate(self.parameters)
        self.history_illum[-1] = estimate.illuminant
        self.history_action[-1] = [1.7, 10]
        return (self.observation(), estimate)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Policy must return two finite physical actions')
        if np.any(action < np.array([-0.6, -4], np.float32) - 1e-06) or np.any(action > np.array([0.6, 4], np.float32) + 1e-06):
            raise ValueError('Policy action outside saved physical bounds')
        self.current_step += 1
        self.parameters[1] = np.clip(self.parameters[1] + action[0], 1.0, 2.0)
        self.parameters[4] = np.clip(self.parameters[4] + action[1], 1.0, 40.0)
        estimate = self.estimator.estimate(self.parameters)
        self.history_illum[:-1] = self.history_illum[1:]
        self.history_illum[-1] = estimate.illuminant
        self.history_action[:-1] = self.history_action[1:]
        self.history_action[-1] = [self.parameters[1], self.parameters[4]]
        differences = np.linalg.norm(np.diff(self.history_illum, axis=0), axis=1)
        converged = bool(np.all(differences[-3:] < 0.05))
        return (self.observation(), estimate, converged)

    def trace_row(self, estimate, action=None):
        return {'step': self.current_step, 'illuminant': estimate.illuminant.tolist(), 'action_delta': None if action is None else np.asarray(action).tolist(), 'gray_pixel_percent': float(self.parameters[1]), 'minkowski_p': float(self.parameters[4]), 'for_rl': estimate.for_rl, 'selected_pixels': estimate.selected_pixels, 'sgp_pixels': estimate.sgp_pixels, 'min_sgp_pixels': estimate.min_sgp_pixels}

def rollout(model, estimator, max_steps=100):
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    controller = Controller(estimator)
    obs, initial = controller.reset()
    trace = [controller.trace_row(initial)]
    estimate = initial
    reason = 'initial_gp_fallback'
    if initial.for_rl:
        reason = 'max_steps'
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, estimate, converged = controller.step(action)
            trace.append(controller.trace_row(estimate, action))
            if converged:
                reason = 'converged'
                break
    return Rollout(estimate.illuminant, initial.illuminant, trace, reason, controller.current_step)
