from upstream import luminance
import numpy as np

def rgb_luminance(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim < 3 or image.shape[-1] != 3 or (image.size == 0):
        raise ValueError('Expected nonempty uint8 RGB images')
    return luminance(image)

def shannon_entropy(values):
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Entropy input must be nonempty and finite')
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / values.size
    return float(-np.sum(probabilities * np.log2(probabilities)))
