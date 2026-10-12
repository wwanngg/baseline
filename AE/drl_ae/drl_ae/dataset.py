from functools import lru_cache
from pathlib import Path
import re
import cv2
import numpy as np
PROJECT = Path(__file__).resolve().parents[1]
AE_ROOT = PROJECT.parent
DEFAULT_DATASET = PROJECT.parent / 'DataSet_AE'
PATTERN = re.compile('^Param-ISO-(\\d+(?:\\.\\d+)?)-ExpT-(\\d+(?:\\.\\d+)?)\\.jpg$', re.I)

def scenes(root: Path, names: list[str] | None=None) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset does not exist: {root}')
    available = {p.name: p for p in sorted(root.iterdir()) if (p / 'left').is_dir() and (p / 'right').is_dir()}
    if names is not None:
        missing = set(names) - available.keys()
        if missing:
            raise ValueError(f'Unknown scenes: {sorted(missing)}')
        result = [available[name] for name in dict.fromkeys(names)]
    else:
        result = list(available.values())
    if not result:
        raise ValueError('No scenes selected')
    return result

@lru_cache(maxsize=8)
def _read_cached(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    image.setflags(write=False)
    return image

def read_image(path: Path) -> np.ndarray:
    return _read_cached(str(path.resolve()))

class SceneGrid:

    def __init__(self, directory: Path, side: str='left'):
        self.directory = Path(directory).resolve()
        self.name = self.directory.name
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.side = side
        self.paths = {}
        for path in (self.directory / side).iterdir():
            match = PATTERN.fullmatch(path.name)
            if match and path.is_file():
                setting = tuple(map(float, match.groups()))
                if setting in self.paths:
                    raise ValueError(f'Duplicate grid cell: {path}')
                self.paths[setting] = path
        if not self.paths:
            raise ValueError(f'Empty image grid: {self.directory / side}')
        self.gains = np.unique([setting[0] for setting in self.paths])
        self.exposures = np.unique([setting[1] for setting in self.paths])
        if min(len(self.gains), len(self.exposures)) < 2:
            raise ValueError('Need at least two settings along each axis')
        if len(self.paths) != len(self.gains) * len(self.exposures):
            raise ValueError(f'Incomplete rectangular grid: {self.name}')
        self.low = np.array([self.exposures[0], self.gains[0]], dtype=np.float64)
        self.high = np.array([self.exposures[-1], self.gains[-1]], dtype=np.float64)

    def nearest(self, exposure_us: float, gain_db: float) -> tuple[float, float, Path]:
        gain = float(self.gains[np.argmin(np.abs(self.gains - gain_db))])
        exposure = float(self.exposures[np.argmin(np.abs(self.exposures - exposure_us))])
        return (exposure, gain, self.paths[gain, exposure])

    def random_setting(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.choice(self.exposures), rng.choice(self.gains)], dtype=np.float64)

    def builtin_ae(self) -> Path | None:
        prefix = 'Param2-' if self.side == 'left' else 'Param1-'
        folder = self.directory / 'AE'
        candidates = sorted((p for p in folder.glob('*.jpg') if p.name.startswith(prefix)))
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous built-in AE reference: {self.name}')
        return candidates[0] if candidates else None

def protect_output(output: Path, dataset: Path) -> Path:
    output, dataset = (output.resolve(), dataset.resolve())
    if output.is_relative_to(dataset):
        raise ValueError('Output must be outside DataSet_AE')
    return output
