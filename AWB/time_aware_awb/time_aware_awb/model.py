from .upstream import IllumEstimator
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
MODEL_NAMES = ('awb', 'awb-w-noise', 'awb-w-snr', 'awb-w-noise-snr')

def device_name(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA unavailable: install CUDA-enabled PyTorch in env on the GPU machine')
    if requested == 'xpu' and (not (hasattr(torch, 'xpu') and torch.xpu.is_available())):
        raise ValueError('XPU 不可用')
    if requested not in ('cpu', 'cuda', 'xpu'):
        raise ValueError('未知设备')
    return requested

class Predictor:

    def __init__(self, weights_dir: Path, model_name: str, target: str, device='cpu'):
        if model_name not in MODEL_NAMES or target not in ('neutral', 'preference'):
            raise ValueError('Invalid model/target')
        self.experiment = model_name + ('-pref' if target == 'preference' else '')
        self.config_path = Path(weights_dir) / f'config-{self.experiment}.json'
        self.weights_path = Path(weights_dir) / f'model-{self.experiment}.pt'
        for path in (self.config_path, self.weights_path):
            if not path.is_file():
                raise FileNotFoundError(f'缺少预训练文件：{path}。请自行下载发布的权重及配置。')
        self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        if bool(self.config['user_preference']) != (target == 'preference'):
            raise ValueError('Checkpoint config ground-truth target does not match --target')
        size = self.config['target_size']
        if len(size) != 2 or any((not isinstance(n, int) or n < 1 for n in size)):
            raise ValueError('Invalid pretrained target_size')
        self.device = torch.device(device_name(device))
        dimensions = len(self.config['norm_values']['capture-data-min'])
        self.network = IllumEstimator(dimensions, 4).to(self.device)
        weights = torch.load(self.weights_path, map_location='cpu', weights_only=True)
        self.network.load_state_dict(weights, strict=True)
        self.network.eval()
        self.parameters = sum((p.numel() for p in self.network.parameters()))

    @torch.inference_mode()
    def predict(self, histogram: np.ndarray, capture: np.ndarray) -> np.ndarray:
        hist = torch.from_numpy(np.ascontiguousarray(histogram, dtype=np.float32)).unsqueeze(0).to(self.device)
        features = torch.from_numpy(np.ascontiguousarray(capture, dtype=np.float32)).unsqueeze(0).to(self.device)
        result = self.network(hist, features)[0].cpu().numpy()
        if not np.isfinite(result).all() or np.linalg.norm(result) <= 0:
            raise ValueError('Nonfinite/zero illuminant prediction')
        return result
