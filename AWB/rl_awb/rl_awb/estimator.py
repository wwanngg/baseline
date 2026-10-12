from dataclasses import dataclass
import numpy as np
from .upstream import MODEL_NAMES, upstream_modules
INITIAL = {'NCC': [0.5, 3.5, 0.045, 0.3, 10, 3, 0, 0.9, 7, 3], 'LEVI': [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3], 'Gehler': [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3]}

@dataclass
class Estimate:
    illuminant: np.ndarray
    for_rl: bool
    selected_pixels: int
    sgp_pixels: int
    min_sgp_pixels: int

class Estimator:

    def __init__(self, image, invalid, model_name, bits=14):
        if model_name not in MODEL_NAMES:
            raise ValueError(f'Unknown checkpoint preset: {model_name}')
        self.image = np.asarray(image, dtype=np.float64)
        self.invalid = np.asarray(invalid, dtype=bool)
        if self.image.ndim != 3 or self.image.shape[2] != 3 or self.invalid.shape != self.image.shape[:2] or (not np.isfinite(self.image).all()) or np.any(self.image < 0) or np.any(self.image > 2 ** bits - 1):
            raise ValueError('SGP-LRD expects finite nonnegative working RGB codes and an aligned mask')
        self.model_name, self.bits = (model_name, bits)
        _, self.algorithm, histogram = upstream_modules()
        if not np.any((self.image > 0).all(axis=2) & ~self.invalid):
            raise ValueError('No unmasked positive RGB pixels for the policy histogram')
        with np.errstate(divide='ignore', invalid='ignore'):
            feature = histogram.rgb_uv_hist(self.image * (~self.invalid)[..., None]).flatten()
        if feature.shape != (10800,) or not np.isfinite(feature).all():
            raise ValueError('RGB-uv histogram has empty bins/channels or nonfinite values')
        self.feature = feature.astype(np.float32)
        saturated = np.max(self.image, axis=2) >= np.max(self.image) * 0.95
        dilated = self.algorithm.dilation33(saturated.astype(np.float64))
        valid = (self.invalid.astype(np.float64) + dilated == 0).astype(np.float64)
        self.gray_invalid = 1 - self.algorithm.set_border(valid, 1, method=0)
        action = np.array(INITIAL[model_name], dtype=np.float32)
        self.local_variance, self.saliency = self.algorithm.ComputeSaliencyMap(self.image, 1 - self.invalid, action[2], action[3])
        self.min_sgp_pixels = int(np.floor(0.05 * np.prod(self.image.shape[:2]) / 100))
        self._gray_cache = {}

    def estimate(self, parameters):
        a = np.asarray(parameters, dtype=np.float32)
        key = (float(a[0]), float(a[1]))
        if key not in self._gray_cache:
            self._gray_cache.clear()
            self._gray_cache[key] = self.algorithm.gray_index_angular(self.image, self.gray_invalid, a[0], a[1])
        gp1, gp2 = self._gray_cache[key]
        if self.model_name == 'Gehler':
            gp, sgp = (gp2, gp1)
        else:
            gp, sgp = (gp2 * self.local_variance, gp1 * self.saliency)
        count = int(np.count_nonzero(sgp))
        for_rl = count >= self.min_sgp_pixels
        selected = sgp if for_rl else gp
        illuminant = self.algorithm.estimate_illuminant_pixelwise_accelerated(self.image * selected[..., None], 2, a[4], a[5], selected, a[6], a[7], int(a[9]), bitDepth=self.bits)
        if not np.isfinite(illuminant).all() or np.any(illuminant <= 0) or np.linalg.norm(illuminant) <= 0:
            raise ValueError('SGP-LRD produced no valid positive illuminant; sample is recorded as failed')
        return Estimate(illuminant, bool(for_rl), int(np.count_nonzero(selected)), count, self.min_sgp_pixels)
