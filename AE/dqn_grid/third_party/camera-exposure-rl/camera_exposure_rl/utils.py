from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)

def resolve_device(requested: str) -> torch.device:
    if requested == 'cuda' and (not torch.cuda.is_available()):
        return torch.device('cpu')
    return torch.device(requested)

@dataclass
class RunPaths:
    root: Path

    @property
    def log_file(self) -> Path:
        return self.root / 'train.jsonl'

    @property
    def checkpoint(self) -> Path:
        return self.root / 'policy.pt'

    @property
    def checkpoint_best(self) -> Path:
        return self.root / 'policy_best.pt'

    @property
    def config_dump(self) -> Path:
        return self.root / 'config.yaml'

def make_run_dir(base: Path) -> RunPaths:
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(base) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(run_dir)

class JsonlLogger:

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open('a', encoding='utf-8')

    def log(self, record: dict[str, Any]) -> None:
        record = {'ts': time.time(), **record}
        self._fh.write(json.dumps(record) + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> 'JsonlLogger':
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
