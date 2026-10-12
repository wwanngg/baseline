from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat, whosmat
PROJECT = Path(__file__).resolve().parents[1]
AF_ROOT = PROJECT.parent
DEFAULT_DATASET = PROJECT.parent / 'DataSet_AF'

@dataclass(frozen=True)
class StackInfo:
    path: str
    key: str
    shape: tuple[int, int, int]
    dtype: str

    def to_dict(self):
        return asdict(self)

def describe(path: Path, key: str | None=None) -> StackInfo:
    path = Path(path).resolve()
    entries = whosmat(str(path))
    candidates = [(name, shape, dtype) for name, shape, dtype in entries if len(shape) == 3 and (key is None or name == key)]
    if len(candidates) != 1:
        raise ValueError(f'{path.name}: expected one 3-D image stack; use --key for an ambiguous MAT file')
    name, shape, dtype = candidates[0]
    if min(shape) < 1:
        raise ValueError(f'{path.name}: empty image stack')
    return StackInfo(str(path), name, tuple(shape), dtype)

def scene_paths(dataset: Path=DEFAULT_DATASET, scene: str | None=None) -> list[Path]:
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError(f'Dataset directory does not exist: {dataset}')
    paths = sorted(dataset.glob('*.mat'))
    if scene is not None:
        paths = [p for p in paths if p.stem == scene]
    if not paths:
        raise ValueError(f'No MAT stacks found in {dataset}' + (f' for scene {scene}' if scene else ''))
    return paths

def load_stack(path: Path, key: str | None=None, frame_limit: int | None=None, crop_size: int | None=None) -> tuple[np.ndarray, dict]:
    info = describe(path, key)
    stack = np.asarray(loadmat(info.path, variable_names=[info.key])[info.key])
    if not np.issubdtype(stack.dtype, np.number) or not np.isrealobj(stack):
        raise ValueError('Image stack must contain real numeric grayscale values')
    if not np.isfinite(stack).all():
        raise ValueError('Image stack contains NaN/Inf')
    rows, cols, frames = info.shape
    used_frames = frames
    if frame_limit is not None:
        if not 1 <= frame_limit <= frames:
            raise ValueError(f'--max-frames must be between 1 and {frames}')
        used_frames = frame_limit
        stack = stack[:, :, :frame_limit]
    crop = [0, 0, rows, cols]
    if crop_size is not None:
        if not 1 <= crop_size <= min(rows, cols):
            raise ValueError(f'--crop-size must be between 1 and {min(rows, cols)}')
        top, left = ((rows - crop_size) // 2, (cols - crop_size) // 2)
        crop = [top, left, crop_size, crop_size]
        stack = stack[top:top + crop_size, left:left + crop_size, :]
    metadata = {**info.to_dict(), 'used_shape': list(stack.shape), 'crop_top_left_height_width': crop, 'partial_stack': used_frames != frames or stack.shape[:2] != (rows, cols), 'ground_truth_available': False, 'calibrated_stage_positions_available': False, 'index_units': 'zero-based frame index; not physical depth'}
    return (stack, metadata)

def protect_output(output: Path, dataset: Path) -> Path:
    output, dataset = (Path(output).resolve(), Path(dataset).resolve())
    for protected in (dataset, (AF_ROOT / 'third_party').resolve()):
        if output == protected or protected in output.parents:
            raise ValueError(f'Output must be outside the dataset and third_party: {output}')
    if output.exists() and (not output.is_dir()):
        raise ValueError(f'Output is not a directory: {output}')
    return output
