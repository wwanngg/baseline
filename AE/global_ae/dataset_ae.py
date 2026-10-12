from dataclasses import dataclass
from pathlib import Path
import re
import cv2
import numpy as np
CODE_ROOT = Path(__file__).resolve().parent
AE_ROOT = CODE_ROOT.parent
DEFAULT_DATASET = CODE_ROOT.parent / 'DataSet_AE'
DEFAULT_OUTPUT = CODE_ROOT / 'results'
PATTERN = re.compile('^Param-ISO-(\\d+(?:\\.\\d+)?)-ExpT-(\\d+(?:\\.\\d+)?)\\.jpg$', re.I)
AE_PATTERN = re.compile('^Param[12]-ExpT-(\\d+(?:\\.\\d+)?)-ISO-(\\d+(?:\\.\\d+)?)\\.jpg$', re.I)

@dataclass(frozen=True)
class Capture:
    gain_db: float
    exposure_us: float
    path: Path

def discover_scenes(root=DEFAULT_DATASET, names=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset does not exist: {root}')
    available = {p.name: p for p in sorted(root.iterdir()) if (p / 'left').is_dir() and (p / 'right').is_dir()}
    if names:
        missing = set(names) - available.keys()
        if missing:
            raise ValueError(f'Unknown scenes: {sorted(missing)}')
        result = [available[name] for name in dict.fromkeys(names)]
    else:
        result = list(available.values())
    if not result:
        raise ValueError('No stereo scenes found')
    return result

def list_captures(scene, side='left', search='exposure', gain_db=0.0):
    if side not in ('left', 'right') or search not in ('exposure', 'joint'):
        raise ValueError('Invalid camera side or search mode')
    captures, settings = ([], set())
    for path in (Path(scene) / side).iterdir():
        match = PATTERN.fullmatch(path.name)
        if match and path.is_file():
            gain, exposure = map(float, match.groups())
            if (gain, exposure) in settings:
                raise ValueError(f'Duplicate setting: {path.name}')
            captures.append(Capture(gain, exposure, path))
            settings.add((gain, exposure))
    if not captures:
        raise ValueError(f'No grid images in {scene}/{side}')
    if len(settings) != len({c.gain_db for c in captures}) * len({c.exposure_us for c in captures}):
        raise ValueError(f'Incomplete exposure grid: {scene}')
    if search == 'exposure':
        if not np.isfinite(gain_db):
            raise ValueError('gain_db must be finite')
        captures = [c for c in captures if c.gain_db == gain_db]
        if not captures:
            raise ValueError(f'Gain {gain_db:g} dB is absent from {scene}')
    return sorted(captures, key=lambda capture: (capture.gain_db, -capture.exposure_us, capture.path.name))

def read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Cannot decode image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def global_plane(rgb, mode='linear-blend'):
    if mode not in ('linear-blend', 'srgb-blend', 'linear-green', 'srgb-green'):
        raise ValueError('Unknown global input mode')
    normalized = rgb.astype(np.float64) / 255.0
    if mode.startswith('linear-'):
        normalized = np.where(normalized <= 0.04045, normalized / 12.92, ((normalized + 0.055) / 1.055) ** 2.4)
    if mode.endswith('green'):
        return normalized[:, :, 1]
    return np.sum(normalized * np.array([0.25, 0.5, 0.25]), axis=-1)

def output_directory(output, dataset):
    output, dataset = (Path(output).resolve(), Path(dataset).resolve())
    if output.is_relative_to(dataset):
        raise ValueError('Output must be outside DataSet_AE')
    return output
