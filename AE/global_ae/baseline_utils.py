import numpy as np

def normalized_plane(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0 or (not np.isfinite(image).all()) or (image.min() < 0) or (image.max() > 1):
        raise ValueError('Expected a nonempty finite 2-D intensity image in [0,1]')
    return image
