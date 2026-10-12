from pathlib import Path
import numpy as np
import torch
from torch import nn
from .source_loader import definitions

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "time-aware-awb"
_namespace = {"np": np, "torch": torch, "nn": nn}
definitions(UPSTREAM / "utils.py", ("vect_norm", "rg_bg_to_illum_rgb"), _namespace)
definitions(UPSTREAM / "model.py", ("HistNet", "IllumEstimator"), _namespace)
HistNet = _namespace["HistNet"]
IllumEstimator = _namespace["IllumEstimator"]
