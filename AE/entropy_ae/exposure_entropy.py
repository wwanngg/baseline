import numpy as np
from baseline_utils import rgb_luminance, shannon_entropy

class ExposureEntropy:

    def __init__(self, raw_images=None, srgb_imgs=None, start_index=20, smoothness_number=1):
        self.raw_imgs, self.srgb_imgs = (raw_images, srgb_imgs)
        self.start_index, self.smoothness_number = (start_index, smoothness_number)
        self.r_percent, self.g_percent, self.b_percent = (0.2126, 0.7152, 0.0722)

    def measure(self, rgb_image):
        luminance = rgb_luminance(rgb_image)
        if luminance.ndim != 2:
            raise ValueError('measure expects a single RGB image')
        return {'entropy_bits': shannon_entropy(luminance)}
