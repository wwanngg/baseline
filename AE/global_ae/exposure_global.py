import numpy as np
from baseline_utils import normalized_plane
from upstream import produce_map

class ExposureGlobal:

    def __init__(self, raw_images=None, target_intensity=0.18, high_threshold=1.0, low_threshold=0.0, start_index=20, smoothness_number=1, saturated_weight=0.0):
        values = [target_intensity, high_threshold, low_threshold, saturated_weight]
        if not np.isfinite(values).all() or not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError('Require 0 <= low_threshold <= high_threshold <= 1')
        if not 0 <= target_intensity <= 1 or not 0 <= saturated_weight <= 1:
            raise ValueError('Invalid target or saturated weight')
        self.raw_imgs = raw_images
        self.target_intensity = float(target_intensity)
        self.high_threshold, self.low_threshold = (float(high_threshold), float(low_threshold))
        self.start_index = start_index
        self.smoothness_number = smoothness_number
        self.saturated_weight = float(saturated_weight)

    def produce_map(self, image):
        image = normalized_plane(image)
        weights, _ = produce_map(self, image)
        weights = weights.astype(np.float64)
        weights[image > self.high_threshold] = self.saturated_weight
        return (weights, float(weights.sum()))

    def measure(self, image):
        image = normalized_plane(image)
        weights, denominator = self.produce_map(image)
        mean = float(np.sum(image * weights) / denominator) if denominator > 0 else None
        return {'target_distance': abs(mean - self.target_intensity) if mean is not None else None}
