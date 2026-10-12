from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class StateConfig:
    roi_height: int = 600
    roi_width: int = 800
    resize_width: int = 128
    history: int = 3

class IntensityHistoryEncoder:

    def __init__(self, config: StateConfig) -> None:
        self.config = config
        self._buffer: deque[np.ndarray] = deque(maxlen=config.history)

    @property
    def state_dim(self) -> int:
        return self.config.resize_width * self.config.history

    def reset(self) -> None:
        self._buffer.clear()

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        profile = _intensity_profile(frame_bgr, roi_h=self.config.roi_height, roi_w=self.config.roi_width, resize_w=self.config.resize_width)
        if not self._buffer:
            for _ in range(self.config.history):
                self._buffer.append(profile)
        else:
            self._buffer.append(profile)
        return np.concatenate(list(self._buffer), dtype=np.float32)

    def last_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _grayscale_roi(frame_bgr, roi_h=self.config.roi_height, roi_w=self.config.roi_width)

def _grayscale_roi(frame_bgr: np.ndarray, *, roi_h: int, roi_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    rh = min(roi_h, h)
    rw = min(roi_w, w)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    crop = frame_bgr[y0:y0 + rh, x0:x0 + rw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def _intensity_profile(frame_bgr: np.ndarray, *, roi_h: int, roi_w: int, resize_w: int) -> np.ndarray:
    gray = _grayscale_roi(frame_bgr, roi_h=roi_h, roi_w=roi_w)
    column_mean = gray.mean(axis=0)
    resized = cv2.resize(column_mean[None, :], (resize_w, 1), interpolation=cv2.INTER_AREA)
    return resized[0].astype(np.float32)
