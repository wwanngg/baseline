from pathlib import Path
import json
import random
import numpy as np
import torch
import yaml
from .dataset import PROJECT, protect_output, scenes
from .environment import CurriculumConfig, EnvConfig, ExposureEnv
from .dataset import SceneGrid
from .rewards import RewardConfig
from .state import StateConfig
from .sac import SACConfig
DEFAULT_CONFIG = PROJECT / 'configs' / 'config.yaml'

def load_config(path: Path=DEFAULT_CONFIG) -> dict:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    for section in ('dataset', 'state', 'reward', 'env', 'curriculum', 'sac', 'train'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f'Missing configuration section: {section}')
    for section, key in (('dataset', 'root'), ('train', 'output')):
        original = Path(cfg[section][key])
        cfg[section][key] = str((original if original.is_absolute() else path.parent / original).resolve())
    cfg['seed'] = int(cfg['seed'])
    if cfg['seed'] < 0:
        raise ValueError('seed must be nonnegative')
    if cfg.get('implementation_version', 2) != 2:
        raise ValueError('当前仅支持最终 SAC 配置')
    StateConfig(**cfg['state'])
    RewardConfig(**cfg['reward'])
    EnvConfig(**cfg['env'])
    CurriculumConfig(**cfg['curriculum'])
    SACConfig(**cfg['sac'])
    for key in ('steps', 'log_every', 'evaluate_every', 'validation_episodes_per_scene'):
        if int(cfg['train'][key]) < 1:
            raise ValueError(f'train.{key} must be positive')
    if cfg['train'].get('checkpoint_metric', 'reward') not in ('reward', 'success_then_brightness'):
        raise ValueError('Unknown checkpoint selection metric')
    for key in ('early_stopping_patience', 'early_stopping_min_steps'):
        value = cfg['train'].get(key, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f'train.{key} must be a nonnegative integer')
    protect_output(Path(cfg['train']['output']), Path(cfg['dataset']['root']))
    return cfg

def split(cfg: dict) -> tuple[list[Path], list[Path], list[Path]]:
    paths = scenes(Path(cfg['dataset']['root']))
    names = cfg['dataset'].get('eval_scenes', [])
    validation_names = cfg['dataset'].get('validation_scenes', [])
    if set(names) & set(validation_names):
        raise ValueError('Validation and evaluation scenes must be disjoint')
    unknown = (set(names) | set(validation_names)) - {p.name for p in paths}
    if unknown:
        raise ValueError(f'Unknown held-out scenes: {sorted(unknown)}')
    training = [p for p in paths if p.name not in names and p.name not in validation_names]
    validation = [p for p in paths if p.name in validation_names]
    evaluation = [p for p in paths if p.name in names]
    if not training or not validation or (not evaluation):
        raise ValueError('Require nonempty, disjoint train, validation and held-out evaluation scenes')
    return (training, validation, evaluation)

def make_env(path: Path, cfg: dict) -> ExposureEnv:
    return ExposureEnv(SceneGrid(path, cfg['dataset']['side']), StateConfig(**cfg['state']), RewardConfig(**cfg['reward']), EnvConfig(**cfg['env']), CurriculumConfig(**cfg['curriculum']))

def device_name(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA is unavailable; choose cpu or xpu')
    if requested == 'xpu' and (not (hasattr(torch, 'xpu') and torch.xpu.is_available())):
        raise ValueError('XPU is unavailable; choose cpu')
    if requested not in ('cpu', 'cuda', 'xpu'):
        raise ValueError('device must be cpu, cuda, xpu or auto')
    return requested

def seed_all(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
