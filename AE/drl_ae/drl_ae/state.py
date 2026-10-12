from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class StateConfig:
    width: int = 128
    height: int = 128
    history: int = 3
    randomization: bool = True
    min_crop_fraction: float = 0.5
    include_controls: bool = False

    def __post_init__(self):
        if min(self.width, self.height, self.history) < 1 or not 0 < self.min_crop_fraction <= 1:
            raise ValueError('Invalid state dimensions or crop fraction')
        if not isinstance(self.include_controls, bool):
            raise ValueError('include_controls must be boolean')

class Encoder:

    def __init__(self, config: StateConfig):
        self.config = config
        self.frames = deque(maxlen=config.history)
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.flip_x = self.flip_y = False
        self.rotation = 0

    @property
    def state_dim(self) -> int:
        return self.config.width * self.config.history

    def reset(self, rng: np.random.Generator, augment: bool=False):
        self.frames.clear()
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.flip_x = self.flip_y = False
        self.rotation = 0
        if augment and self.config.randomization:
            height, width = rng.uniform(self.config.min_crop_fraction, 1, 2)
            y, x = (rng.uniform(0, 1 - height), rng.uniform(0, 1 - width))
            self.crop = (float(y), float(x), float(height), float(width))
            self.flip_x, self.flip_y = map(bool, rng.integers(0, 2, 2))
            self.rotation = int(rng.integers(4))

    def roi(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 BGR image')
        h, w = image.shape[:2]
        y, x, rh, rw = self.crop
        y0, x0 = (min(h - 1, int(y * h)), min(w - 1, int(x * w)))
        crop = image[y0:min(h, y0 + max(1, int(rh * h))), x0:min(w, x0 + max(1, int(rw * w)))]
        if self.flip_x:
            crop = crop[:, ::-1]
        if self.flip_y:
            crop = crop[::-1]
        crop = np.ascontiguousarray(np.rot90(crop, self.rotation))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255

    def encode_roi(self, roi: np.ndarray) -> np.ndarray:
        profile = roi.mean(axis=0).astype(np.float32)
        if not self.frames:
            self.frames.extend((profile.copy() for _ in range(self.config.history)))
        else:
            self.frames.append(profile.copy())
        return np.concatenate(self.frames).astype(np.float32)
