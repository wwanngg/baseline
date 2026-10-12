from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
from .upstream import AWB_ROOT, upstream_modules
VARIANT_DIRS = {'raw': 'raw_images', 'denoised': 'denoised_raw_images'}

@dataclass(frozen=True)
class Sample:
    name: str
    root: Path

    def image_path(self, variant):
        return self.root / VARIANT_DIRS[variant] / f'{self.name}.png'

    @property
    def mask_path(self):
        return self.root / 'masks' / f'{self.name}.png'

    def metadata(self):
        return json.loads((self.root / 'data' / f'{self.name}.json').read_text(encoding='utf-8'))

def samples(dataset, names=None):
    dataset = Path(dataset).resolve()
    ids = sorted((path.stem for path in (dataset / 'data').glob('*.json')))
    if not ids:
        raise ValueError(f"No metadata JSON in {dataset / 'data'}")
    for directory in VARIANT_DIRS.values():
        image_ids = {path.stem for path in (dataset / directory).glob('*.png')}
        if image_ids != set(ids):
            missing, extra = (set(ids) - image_ids, image_ids - set(ids))
            raise ValueError(f'Unpaired {directory}: missing={sorted(missing)}, extra={sorted(extra)}')
    if names:
        unknown = set(names) - set(ids)
        if unknown:
            raise ValueError(f'Unknown image IDs: {sorted(unknown)}')
        ids = sorted(set(names))
    return [Sample(name, dataset) for name in ids]

def read_png(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f'Cannot decode PNG: {path}')
    return image

def prepare_image(sample, variant, scale=0.125, bits=14, use_mask=False):
    _, algorithm, _ = upstream_modules()
    bgr = read_png(sample.image_path(variant))
    if bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f'Expected demosaiced linear uint16 RGB PNG: {sample.image_path(variant)}')
    shape = list(bgr.shape)
    if min(np.round(np.array(shape[:2]) * scale)) < 8:
        raise ValueError('Working image must be at least 8 pixels per dimension')
    resized = algorithm.imresize_nearest(bgr, scale)
    image = resized[..., ::-1].astype(np.float64) / 65535 * (2 ** bits - 1)
    if use_mask:
        mask = read_png(sample.mask_path)
        if mask.ndim != 2 or mask.dtype != np.uint8 or mask.shape != tuple(shape[:2]):
            raise ValueError(f'Expected aligned grayscale uint8 exclusion mask: {sample.mask_path}')
        invalid = algorithm.imresize_nearest(mask, scale) > 0
    else:
        invalid = np.zeros(image.shape[:2], dtype=bool)
    return (image, invalid, shape)

def protect_output(output, dataset, weights_dir=None):
    output = Path(output).resolve()
    protected = [Path(dataset).resolve(), (AWB_ROOT / 'S24_val').resolve(), (AWB_ROOT / 'third_party').resolve()]
    for root in protected:
        if output == root or root in output.parents:
            raise ValueError(f'Output must be outside dataset and third_party: {output}')
    if output.exists() and (not output.is_dir()):
        raise ValueError(f'Output is not a directory: {output}')
    return output
