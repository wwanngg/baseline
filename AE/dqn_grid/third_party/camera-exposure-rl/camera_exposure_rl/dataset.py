from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
_FILENAME_RE = re.compile('^Param-ISO-(\\d+)-ExpT-(\\d+)\\.jpg$')

@dataclass(frozen=True)
class GridSpec:
    iso_indices: tuple[int, ...]
    exposure_us: tuple[int, ...]

    @property
    def n_iso(self) -> int:
        return len(self.iso_indices)

    @property
    def n_exp(self) -> int:
        return len(self.exposure_us)

class SceneGrid:

    def __init__(self, scene_dir: Path, side: str='left') -> None:
        if side not in {'left', 'right'}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.scene_dir = Path(scene_dir)
        self.side = side
        self.name = self.scene_dir.name
        files = sorted((self.scene_dir / side).glob('*.jpg'))
        if not files:
            raise FileNotFoundError(f'No images under {self.scene_dir / side}')
        cell_to_path: dict[tuple[int, int], Path] = {}
        iso_set: set[int] = set()
        exp_set: set[int] = set()
        for f in files:
            m = _FILENAME_RE.match(f.name)
            if m is None:
                continue
            iso_idx, exp_us = (int(m.group(1)), int(m.group(2)))
            cell_to_path[iso_idx, exp_us] = f
            iso_set.add(iso_idx)
            exp_set.add(exp_us)
        self.spec = GridSpec(iso_indices=tuple(sorted(iso_set)), exposure_us=tuple(sorted(exp_set)))
        self._cell_to_path = cell_to_path

    @property
    def lux(self) -> int | None:
        m = re.search('_(\\d+)Lux$', self.name)
        return int(m.group(1)) if m else None

    def path(self, iso_idx: int, exp_idx: int) -> Path:
        iso = self.spec.iso_indices[iso_idx]
        exp = self.spec.exposure_us[exp_idx]
        try:
            return self._cell_to_path[iso, exp]
        except KeyError as e:
            raise KeyError(f'Missing grid cell (iso={iso}, exp_us={exp}) in scene {self.name}') from e

    def load(self, iso_idx: int, exp_idx: int) -> np.ndarray:
        return _imread_cached(str(self.path(iso_idx, exp_idx)))

    def __repr__(self) -> str:
        return f'SceneGrid(name={self.name!r}, side={self.side!r}, iso={self.spec.n_iso}, exp={self.spec.n_exp})'

@lru_cache(maxsize=512)
def _imread_cached(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cv2.imread returned None for {path}')
    return img

def discover_scenes(root: Path) -> list[Path]:
    root = Path(root)
    scenes = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / 'left').exists() and (p / 'right').exists()]
    if not scenes:
        raise FileNotFoundError(f'No scene folders found under {root}')
    return scenes

def split_scenes(scenes: list[Path], eval_substrings: list[str] | None) -> tuple[list[Path], list[Path]]:
    eval_substrings = eval_substrings or []
    eval_set = {s for s in scenes if any((sub in s.name for sub in eval_substrings))}
    train = [s for s in scenes if s not in eval_set]
    eval_ = [s for s in scenes if s in eval_set]
    return (train, eval_)
