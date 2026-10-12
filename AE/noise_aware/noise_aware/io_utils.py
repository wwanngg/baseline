from pathlib import Path
import json
import cv2
import numpy as np

def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image

def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode image: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def output_directory(output: str | Path, dataset: str | Path) -> Path:
    output, dataset = (Path(output).resolve(), Path(dataset).resolve())
    if output.is_relative_to(dataset):
        raise ValueError('Output must be outside DataSet_AE; the dataset is read-only.')
    return output
