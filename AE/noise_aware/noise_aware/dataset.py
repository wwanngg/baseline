from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
DEFAULT_DATASET = Path(__file__).resolve().parents[2] / 'DataSet_AE'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'results'
_PARAMETER = re.compile('^Param-ISO-(?P<gain>\\d+(?:\\.\\d+)?)-ExpT-(?P<exposure>\\d+(?:\\.\\d+)?)\\.jpg$', re.IGNORECASE)
_BUILTIN = re.compile('^Param(?P<camera>[12])-ExpT-(?P<exposure>\\d+(?:\\.\\d+)?)-ISO-(?P<gain>\\d+(?:\\.\\d+)?)\\.jpg$', re.IGNORECASE)

@dataclass(frozen=True)
class Capture:
    gain_db: float
    exposure_us: float
    path: Path

def discover_scenes(root: str | Path=DEFAULT_DATASET, names: list[str] | None=None) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset does not exist: {root}')
    scenes = sorted((p for p in root.iterdir() if p.is_dir() and (p / 'left').is_dir() and (p / 'right').is_dir()))
    if names:
        available = {p.name: p for p in scenes}
        missing = sorted(set(names) - available.keys())
        if missing:
            raise ValueError(f"Unknown scenes: {', '.join(missing)}")
        scenes = [available[name] for name in dict.fromkeys(names)]
    if not scenes:
        raise ValueError(f'No stereo scenes found in {root}')
    return scenes

def list_captures(scene: Path, camera: str='left') -> list[Capture]:
    if camera not in ('left', 'right'):
        raise ValueError('camera must be left or right')
    captures = []
    for path in (scene / camera).iterdir():
        match = _PARAMETER.fullmatch(path.name)
        if path.is_file() and match:
            captures.append(Capture(float(match['gain']), float(match['exposure']), path))
    captures.sort(key=lambda c: (c.gain_db, c.exposure_us, c.path.name))
    if not captures:
        raise ValueError(f'No Param-ISO-*-ExpT-*.jpg images found in {scene / camera}')
    settings = {(c.gain_db, c.exposure_us) for c in captures}
    if len(settings) != len(captures):
        raise ValueError(f'Duplicate exposure settings in {scene / camera}')
    return captures

def nearest_capture(captures: list[Capture], point: np.ndarray | tuple[float, float]) -> Capture:
    gains = np.unique([c.gain_db for c in captures])
    exposures = np.unique([c.exposure_us for c in captures])
    point = np.asarray(point, dtype=float)
    gain = gains[np.argmin(np.abs(gains - point[0]))]
    exposure = exposures[np.argmin(np.abs(exposures - point[1]))]
    for capture in captures:
        if capture.gain_db == gain and capture.exposure_us == exposure:
            return capture
    span = np.maximum([np.ptp(gains), np.ptp(exposures)], 1)
    return min(captures, key=lambda c: float(np.linalg.norm((np.array([c.gain_db, c.exposure_us]) - point) / span)))

def builtin_ae_images(scene: Path) -> list[tuple[str, Capture]]:
    images = []
    folder = scene / 'AE'
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            match = _BUILTIN.fullmatch(path.name)
            if match and path.is_file():
                camera = 'left' if match['camera'] == '2' else 'right'
                images.append((camera, Capture(float(match['gain']), float(match['exposure']), path)))
    return images
