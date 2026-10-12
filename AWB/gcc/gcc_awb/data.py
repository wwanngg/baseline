from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .upstream import AWB_ROOT, UPSTREAM, modules
VARIANT_DIRS = {'raw': 'raw_images', 'denoised': 'denoised_raw_images'}
DEFAULT_DATASET = AWB_ROOT / 'S24_val'
DEFAULT_CHART = UPSTREAM / 'color_chart.jpg'

@dataclass(frozen=True)
class Sample:
    name: str
    root: Path

    def image_path(self, variant):
        return self.root / VARIANT_DIRS[variant] / f'{self.name}.png'

    def metadata(self):
        return json.loads((self.root / 'data' / f'{self.name}.json').read_text(encoding='utf-8'))

def samples(dataset=DEFAULT_DATASET, names=None):
    root = Path(dataset).resolve()
    ids = sorted((path.stem for path in (root / 'data').glob('*.json')))
    if not ids:
        raise ValueError(f"No S24 metadata in {root / 'data'}")
    for directory in VARIANT_DIRS.values():
        image_ids = {path.stem for path in (root / directory).glob('*.png')}
        if image_ids != set(ids):
            raise ValueError(f'Unpaired IDs in {root / directory}')
    if names:
        if set(names) - set(ids):
            raise ValueError(f'Unknown image IDs: {sorted(set(names) - set(ids))}')
        ids = sorted(set(names))
    return [Sample(name, root) for name in ids]

def read_png(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f'Cannot decode image: {path}')
    return image

def load_chart(path=DEFAULT_CHART):
    chart = read_png(Path(path))
    if chart.dtype != np.uint8 or chart.ndim != 3 or chart.shape[2] != 3:
        raise ValueError('Expected the original uint8 BGR color checker JPEG')
    return cv2.resize(chart, (180, 135), interpolation=cv2.INTER_LINEAR)

def prepare_image(sample, variant, chart, use_mask=False):
    _, evaluation, _ = modules()
    bgr = read_png(sample.image_path(variant))
    if bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected a demosaiced, black-level-normalized linear RGB uint16 PNG')
    shape = list(bgr.shape)
    working_bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR) if bgr.shape[:2] != (512, 512) else bgr.copy()
    if use_mask:
        mask_path = sample.root / 'masks' / f'{sample.name}.png'
        excluded = read_png(mask_path)
        if excluded.dtype != np.uint8 or excluded.ndim != 2 or excluded.shape != tuple(shape[:2]):
            raise ValueError(f'Expected an aligned uint8 S24 exclusion mask: {mask_path}')
        excluded = cv2.resize(excluded, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        working_bgr[excluded] = 0
    gamma_bgr = evaluation.gamma_correction(working_bgr)
    mask_pos, checker_pos = evaluation.get_safe_position((256, 256), 180, 135, (512, 512), padding=3)
    inpaint = np.zeros((512, 512), dtype=np.uint8)
    x1, y1, x2, y2 = mask_pos
    inpaint[y1:y2, x1:x2] = 255
    x1, y1, x2, y2 = checker_pos
    gamma_bgr[y1:y2, x1:x2] = chart
    return (Image.fromarray(gamma_bgr[..., ::-1].copy()), Image.fromarray(inpaint), checker_pos, working_bgr[..., ::-1].copy(), shape)

def estimate_illuminant(generated, checker_pos):
    _, evaluation, _ = modules()
    linear = (evaluation.gamma_correction(generated, 1 / 2.2) / 257).astype(np.float32)
    estimate = evaluation.estimate_illuminant(linear, checker_pos).astype(np.float64)
    if estimate.shape != (3,) or not np.isfinite(estimate).all() or np.any(estimate <= 0):
        raise ValueError('GCC gray patch produced no valid positive illuminant')
    return (estimate / np.linalg.norm(estimate), estimate)

def protect_output(output, dataset):
    output = Path(output).resolve()
    for protected in (Path(dataset).resolve(), DEFAULT_DATASET.resolve(), UPSTREAM.parent.resolve(), (UPSTREAM.parent.parent / 'weights').resolve()):
        if output == protected or protected in output.parents:
            raise ValueError(f'Output must be outside datasets, weights and third_party: {output}')
    if output.exists() and (not output.is_dir()):
        raise ValueError(f'Output is not a directory: {output}')
    return output
