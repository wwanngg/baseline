from pathlib import Path
from types import SimpleNamespace
import numpy as np
from source_loader import definitions

UPSTREAM = Path(__file__).resolve().parent / "third_party" / "ae-video-platform"
_entropy = definitions(UPSTREAM / "exposure_entropy.py", ("ExposureEntropy",), {"np": np, "Exposure": object})["ExposureEntropy"]

def luminance(image):
    shape = image.shape[:-1]
    source = SimpleNamespace(srgb_imgs=image.reshape(-1, 1, *image.shape[-3:]), absolute_bit=256, r_percent=0.2126, g_percent=0.7152, b_percent=0.0722)
    return _entropy.input_imgs_processing(source).reshape(shape)
