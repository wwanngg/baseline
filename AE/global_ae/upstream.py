from pathlib import Path
from types import SimpleNamespace
import numpy as np
from source_loader import definitions

UPSTREAM = Path(__file__).resolve().parent / "third_party" / "ae-video-platform"
_global = definitions(UPSTREAM / "exposure_global.py", ("ExposureGlobal",), {"np": np, "HistogramBase": object})["ExposureGlobal"]

def produce_map(algorithm, image):
    return _global.produce_map(algorithm, image)
