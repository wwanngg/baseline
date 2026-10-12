import importlib
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import SAC
PROJECT = Path(__file__).resolve().parents[1]
AWB_ROOT = PROJECT.parent
UPSTREAM = PROJECT / 'third_party' / 'RL-AWB'
MODEL_NAMES = ('NCC', 'LEVI', 'Gehler')

def upstream_modules():
    source = UPSTREAM.resolve()
    for filename in ('utils/sacpolicy.py', 'utils/Algorithm.py', 'utils/WBsRGB.py'):
        if not (source / filename).is_file():
            raise FileNotFoundError(f'Missing original RL-AWB source: {source / filename}')
    existing = sys.modules.get('utils')
    if existing is not None:
        path = getattr(existing, '__file__', None)
        if path is None or Path(path).resolve().parent != source / 'utils':
            raise ImportError('Another package named utils is already loaded. Run this adapter in its own Python process.')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        policy = importlib.import_module('utils.sacpolicy')
        algorithm = importlib.import_module('utils.Algorithm')
        histogram = importlib.import_module('utils.WBsRGB')
    finally:
        sys.dont_write_bytecode = previous
    for module in (policy, algorithm, histogram):
        if source not in Path(module.__file__).resolve().parents:
            raise ImportError(f'Unexpected source module: {module.__file__}')
    return (policy, algorithm, histogram)

def resolve_device(requested):
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
    if requested == 'auto':
        return 'cuda' if cuda else 'xpu' if xpu else 'cpu'
    if requested == 'cuda' and (not cuda):
        raise RuntimeError('CUDA is unavailable in this conda env. Use --device cpu or a CUDA-enabled PyTorch env on the 3090 machine.')
    if requested == 'xpu' and (not xpu):
        raise RuntimeError('XPU is unavailable in this conda env.')
    return requested

def load_model(path, device='cpu', seed=531):
    upstream_modules()
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}. Supply the weight yourself; no download is attempted.')
    model = SAC.load(str(path), device=resolve_device(device), seed=seed, buffer_size=1)
    if model.observation_space.shape != (10811,):
        raise ValueError(f'Unexpected observation shape in {path}: {model.observation_space.shape}')
    if not (np.allclose(model.action_space.low, [-0.6, -4]) and np.allclose(model.action_space.high, [0.6, 4])):
        raise ValueError(f'Unexpected action space in {path}')
    model.policy.set_training_mode(False)
    return model
