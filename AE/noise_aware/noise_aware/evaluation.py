from dataclasses import asdict
from pathlib import Path
import json
import shutil
import numpy as np
from .dataset import list_captures
from .io_utils import output_directory, write_json
from .metric import MetricConfig, assess_image

def load_workspace(path):
    if not Path(path).is_file():
        raise FileNotFoundError("缺少搜索工作区，请先运行 prepare")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    result["metadata"] = json.loads(str(result["metadata_json"].item()))
    return result

def metric_grid(gains, exposures, values):
    gain_axis, exposure_axis = np.unique(gains), np.unique(exposures)
    grid = np.full((len(exposure_axis), len(gain_axis)), np.nan)
    grid[np.searchsorted(exposure_axis, exposures), np.searchsorted(gain_axis, gains)] = values
    return gain_axis, exposure_axis, grid

def copy_stereo_pair(scene, filename, destination, label, gain, exposure):
    selected = {"gain_db": float(gain), "exposure_us": float(exposure), "source_filename": filename}
    for side in ("left", "right"):
        source = Path(scene) / side / filename
        target = Path(destination) / side / f"{label}_{filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected[side] = target.relative_to(destination).as_posix()
    return selected

def evaluate_scene(scene, output, config=None, camera="left", max_images=None, plots=False, progress=print):
    config = config or MetricConfig()
    output = output_directory(output, scene.parent)
    captures = list_captures(scene, camera)
    total = len(captures)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("图像数量须为正整数")
        captures = captures[:max_images]
    destination = output / "Result_graph" / scene.name
    destination.mkdir(parents=True, exist_ok=True)
    scores = []
    for index, capture in enumerate(captures, 1):
        scores.append(assess_image(capture.path, config).score)
        if progress and (index == 1 or index % 25 == 0 or index == len(captures)):
            progress(f"{scene.name}：搜索准备 {index}/{len(captures)}")
    metadata = {"scene": scene.name, "camera": camera, "config": asdict(config), "complete": len(captures) == total}
    workspace = destination / "workspace_result.npz"
    np.savez_compressed(workspace, filenames=np.asarray([c.path.name for c in captures]), gains=np.asarray([c.gain_db for c in captures]), exposures=np.asarray([c.exposure_us for c in captures]), metrics=np.asarray(scores).reshape(-1, 1), metadata_json=np.asarray(json.dumps(metadata)))
    write_json(destination / "metadata.json", metadata)
    return workspace
