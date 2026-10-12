from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
PROJECT = Path(__file__).resolve().parents[1]
AWB_ROOT = PROJECT.parent
DEFAULT_DATASET = PROJECT.parent / 'S24_val'
UPSTREAM = PROJECT / 'third_party' / 'time-aware-awb'
DEFAULT_WEIGHTS = PROJECT / 'weights'
VARIANT_DIRS = {'raw': 'raw_images', 'denoised': 'denoised_raw_images'}

@dataclass(frozen=True)
class Sample:
    name: str
    root: Path

    @property
    def metadata_path(self):
        return self.root / 'data' / f'{self.name}.json'

    def image_path(self, variant):
        return self.root / VARIANT_DIRS[variant] / f'{self.name}.png'

    @property
    def mask_path(self):
        return self.root / 'masks' / f'{self.name}.png'

    def metadata(self):
        return json.loads(self.metadata_path.read_text(encoding='utf-8'))

def samples(dataset: Path=DEFAULT_DATASET, names=None) -> list[Sample]:
    dataset = Path(dataset).resolve()
    paths = sorted((dataset / 'data').glob('*.json'))
    if not paths:
        raise ValueError(f"No metadata JSON files in {dataset / 'data'}")
    by_name = {path.stem: Sample(path.stem, dataset) for path in paths}
    if names:
        unknown = set(names) - by_name.keys()
        if unknown:
            raise ValueError(f"Unknown image IDs: {', '.join(sorted(unknown))}")
        return [by_name[name] for name in sorted(set(names))]
    return list(by_name.values())

def read_png(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f'Cannot decode PNG: {path}')
    return image

def encode_png16_roundtrip(image: np.ndarray) -> np.ndarray:
    quantized = (np.clip(image, 0, 1).astype(np.float32) * 65535).astype(np.uint16)
    return quantized.astype(np.float64) / 65535

def prepare_image(sample: Sample, variant: str, target_size, use_mask: bool):
    bgr = read_png(sample.image_path(variant))
    if bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f'Expected demosaiced linear RGB uint16 PNG: {sample.image_path(variant)}')
    original_shape = list(bgr.shape)
    rgb = bgr[..., ::-1].astype(np.float64) / 65535
    height, width = target_size
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    display_rgb = encode_png16_roundtrip(rgb)
    if use_mask:
        mask = read_png(sample.mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.dtype != np.uint8 or mask.shape != tuple(original_shape[:2]):
            raise ValueError(f'Expected aligned uint8 mask: {sample.mask_path}')
        mask = mask.astype(np.float64) / 255
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        rgb = rgb * (1 - mask[:, :, None])
    return (encode_png16_roundtrip(rgb), display_rgb, original_shape)

def protect_output(output: Path, dataset: Path) -> Path:
    output, dataset = (Path(output).resolve(), Path(dataset).resolve())
    for protected in (dataset, (AWB_ROOT / 'third_party').resolve()):
        if output == protected or protected in output.parents:
            raise ValueError(f'Output must be outside S24_val and third_party: {output}')
    if output.exists() and (not output.is_dir()):
        raise ValueError(f'Output is not a directory: {output}')
    return output
