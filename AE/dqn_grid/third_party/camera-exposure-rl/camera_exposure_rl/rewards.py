from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class RewardConfig:
    brightness_target: float = 0.5
    brightness_p: float = 0.5
    w_mean: float = 1.5
    w_flicker: float = 1.0
    w_sharpness: float = 0.1

def brightness_penalty(roi: np.ndarray, *, target: float, p: float) -> float:
    return float(np.mean(np.abs(roi - target) ** p))

def flicker_penalty(roi_t: np.ndarray, roi_prev: np.ndarray) -> float:
    return float(np.mean(np.abs(roi_t - roi_prev)))

def sharpness_bonus(roi: np.ndarray) -> float:
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))

def combined_reward(roi_t: np.ndarray, roi_prev: np.ndarray | None, config: RewardConfig) -> tuple[float, dict[str, float]]:
    r_mean = brightness_penalty(roi_t, target=config.brightness_target, p=config.brightness_p)
    r_sharp = sharpness_bonus(roi_t)
    r_flk = flicker_penalty(roi_t, roi_prev) if roi_prev is not None else 0.0
    total = -config.w_mean * r_mean - config.w_flicker * r_flk + config.w_sharpness * r_sharp
    components = {'r_mean': r_mean, 'r_flicker': r_flk, 'r_sharpness': r_sharp, 'r_total': total}
    return (total, components)
