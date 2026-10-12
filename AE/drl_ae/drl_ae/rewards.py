from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class RewardConfig:
    mode: str = 'mean_target'
    target: float = 0.5
    exponent: float = 1.0
    brightness_weight: float = 1.5
    flicker_weight: float = 1.0
    noise_weight: float = 0.1

    def __post_init__(self):
        if self.mode != 'mean_target':
            raise ValueError('最终 SAC 配置使用 mean_target 奖励')
        values = (self.target, self.exponent, self.brightness_weight, self.flicker_weight, self.noise_weight)
        if not np.isfinite(values).all() or not 0 <= self.target <= 1 or self.exponent <= 0 or (min(values[2:]) < 0):
            raise ValueError('Invalid reward parameters')

def reward_terms(roi: np.ndarray, previous: np.ndarray | None, config: RewardConfig) -> dict[str, float]:
    if roi.ndim != 2 or not np.isfinite(roi).all() or roi.min() < 0 or (roi.max() > 1):
        raise ValueError('Expected normalized 2-D grayscale RoI')
    intensity = float(roi.mean())
    brightness_error = abs(intensity - config.target)
    distance = brightness_error ** config.exponent
    if previous is not None and previous.shape != roi.shape:
        raise ValueError('Consecutive RoI dimensions must match')
    flicker = float(np.abs(roi - previous).mean()) if previous is not None else 0.0
    dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    sobel = float(np.hypot(dx, dy).mean())
    sign = -1
    brightness = sign * config.brightness_weight * distance
    flicker_term = -config.flicker_weight * flicker
    noise_term = -config.noise_weight * sobel
    return {'brightness_distance': float(distance), 'brightness_error': brightness_error, 'flicker': flicker, 'sobel_noise_proxy': sobel, 'reward_brightness': float(brightness), 'reward_flicker': float(flicker_term), 'reward_noise': float(noise_term), 'reward': float(brightness + flicker_term + noise_term), 'mean_intensity': intensity}
