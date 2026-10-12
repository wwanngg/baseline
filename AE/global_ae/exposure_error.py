import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

BUILTIN_LEFT = re.compile(r"^Param2-ExpT-(\d+(?:\.\d+)?)-ISO-(\d+(?:\.\d+)?)\.jpg$", re.I)
CAPTURE = re.compile(r"^Param-ISO-(\d+(?:\.\d+)?)-ExpT-(\d+(?:\.\d+)?)\.jpg$", re.I)
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "DataSet_AE"
COMMON_SCENES = {"indoor_lounage1_200Lux", "indoor_meeting_room2_191Lux", "outdoor_car5_5290Lux", "outdoor_weed1_3350Lux"}


def signed_exposure_error(exposure_us, gain_db, reference_us, reference_gain_db):
    if not all(math.isfinite(x) for x in (exposure_us, gain_db, reference_us, reference_gain_db)) or min(exposure_us, reference_us) <= 0:
        raise ValueError("曝光时间须为正值，曝光参数须为有限数值")
    return math.log2(exposure_us / reference_us) + (gain_db - reference_gain_db) / (20 * math.log10(2))


def read_reference(scene):
    candidates = [(p, BUILTIN_LEFT.fullmatch(p.name)) for p in (Path(scene) / "AE").glob("*.jpg")]
    candidates = [(p, match) for p, match in candidates if match]
    if len(candidates) != 1:
        raise ValueError("须有唯一的左相机内置 AE 参考")
    return tuple(map(float, candidates[0][1].groups()))


def export_selection(scene, output, method, filename, gain_db, exposure_us, seed=0):
    folder = Path(output) / "Result_Exp" / Path(scene).name
    selected = {"gain_db": float(gain_db), "exposure_us": float(exposure_us), "source_filename": filename, "episode_seed": int(seed)}
    for side in ("left", "right"):
        source = Path(scene) / side / filename
        destination = folder / side / f"{method}_{filename}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        selected[side] = destination.relative_to(folder).as_posix()
    (folder / "selected_images.json").write_text(json.dumps({method: selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_error_metrics(errors, label):
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        raise ValueError("没有有效的曝光误差样本")
    std = f"{values.std(ddof=1):.6f}" if len(values) > 1 else "N/A"
    print(f"{label}：有效样本 {len(values)}，Exposure error={values.mean():.6f} ± {std} EV")


def evaluate_exposure(results, method, dataset=DEFAULT_DATASET, names=None):
    results, dataset = Path(results), Path(dataset)
    scenes = sorted(p for p in dataset.iterdir() if (p / "left").is_dir() and (p / "right").is_dir() and (names is None or p.name in names))
    rows = []
    failed = 0
    for scene in scenes:
        folder = results / "Result_Exp" / scene.name
        manifest = folder / "selected_images.json"
        if not manifest.is_file():
            failed += 1
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if method not in record:
            failed += 1
            continue
        try:
            selection = record[method]
            parsed = CAPTURE.fullmatch(selection["source_filename"])
            if parsed is None:
                raise ValueError("选图文件名无法解析")
            gain, exposure = map(float, parsed.groups())
            if not math.isclose(gain, selection["gain_db"], abs_tol=1e-9) or not math.isclose(exposure, selection["exposure_us"], abs_tol=1e-9):
                raise ValueError("选图参数与文件名不一致")
            source = scene / "left" / selection["source_filename"]
            exported = (folder / selection["left"]).resolve()
            if folder.resolve() not in exported.parents:
                raise ValueError("选图路径越出结果目录")
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(exported.read_bytes()).digest():
                raise ValueError("选图与原图字节不一致")
            candidates = [(p, BUILTIN_LEFT.fullmatch(p.name)) for p in (scene / "AE").glob("*.jpg")]
            candidates = [(p, match) for p, match in candidates if match]
            if len(candidates) != 1:
                raise ValueError("须有唯一的左相机内置 AE 参考")
            reference_us, reference_gain = map(float, candidates[0][1].groups())
            delta = signed_exposure_error(exposure, gain, reference_us, reference_gain)
            rows.append({"scene": scene.name, "signed_delta_ev": delta, "exposure_error_ev": abs(delta)})
        except (OSError, ValueError, KeyError) as error:
            failed += 1
            print(f"{scene.name}：指标计算失败，{error}")
    if not rows:
        raise ValueError("没有可评价的终态选图，请先完成该方法的推理")
    print(f"\n{method}：左相机内置 AE 参考，曝光时间×10^(增益dB/20)，单位 EV，越低越接近参考")
    print("范围                 有效场景       Exposure error：Mean ± Std，EV")
    for label, chosen in (("共同4场景", [r for r in rows if r["scene"] in COMMON_SCENES]), ("本次全部场景", rows)):
        if not chosen:
            continue
        values = np.asarray([r["exposure_error_ev"] for r in chosen], dtype=np.float64)
        std = f"{values.std(ddof=1):.6f}" if len(values) > 1 else "N/A"
        print(f"{label:<14} {len(chosen):>6} {values.mean():>12.6f} ± {std}")
    print(f"无效选图：{failed}；相机内置 AE 参考不等同于人工理想曝光真值。")
    return rows


def main():
    parser = argparse.ArgumentParser(description="评价已保存终态选图的 Exposure error，只在终端输出指标")
    parser.add_argument("--results", type=Path, required=True, help="推理结果目录")
    parser.add_argument("--method", choices=("NM", "Global", "Entropy", "DRL-AE", "DQN-grid"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="原位置的数据集")
    args = parser.parse_args()
    evaluate_exposure(args.results, args.method, args.dataset)


if __name__ == "__main__":
    main()
