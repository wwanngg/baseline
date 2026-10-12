import importlib.util
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "camera-exposure-rl"
_path = UPSTREAM / "camera_exposure_rl" / "models.py"
_spec = importlib.util.spec_from_file_location("_drl_ae_original_models", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MLP = _module.MLP
