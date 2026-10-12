from pathlib import Path
import sys
from dataset_ae import list_captures, output_directory, read_rgb, global_plane
from exposure_entropy import ExposureEntropy

from exposure_error import export_selection

def infer_scene(scene, output, gain_db=0, stride=8, target=0.13, low=0, high=0.9, limit=None):
    if stride < 1 or limit is not None and limit < 1:
        raise ValueError("采样步长与图像数量须为正整数")
    output = output_directory(output, scene.parent)
    captures = list_captures(scene, search="exposure", gain_db=gain_db)
    if limit is not None:
        captures = captures[:limit]
    algorithm = ExposureEntropy()
    rows = []
    for capture in captures:
        rgb = read_rgb(capture.path)[::stride, ::stride]
        value = algorithm.measure(rgb)["entropy_bits"]
        rows.append((capture, value))
    if not rows:
        raise ValueError("没有有效的候选曝光图像")
    selected, _ = max(rows, key=lambda row: row[1])
    export_selection(scene, output, "Entropy", selected.path.name, selected.gain_db, selected.exposure_us)
    print(f"已推理 {scene.name}，候选图像 {len(captures)} 张")
