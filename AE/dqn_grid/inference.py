import csv
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "third_party/camera-exposure-rl"))

from camera_exposure_rl.agents import DQNAgent
from camera_exposure_rl.dataset import SceneGrid, _imread_cached, discover_scenes, split_scenes
from camera_exposure_rl.env import ExposureEnv, EnvConfig
from camera_exposure_rl.rewards import RewardConfig
from camera_exposure_rl.state import StateConfig
from camera_exposure_rl.utils import load_config, seed_everything
from exposure_error import evaluate_exposure, export_selection, print_error_metrics, read_reference, signed_exposure_error


def build_env(cfg, path):
    return ExposureEnv(SceneGrid(path, side="left"), StateConfig(roi_height=cfg["state"]["roi"]["height"], roi_width=cfg["state"]["roi"]["width"], resize_width=cfg["state"]["resize_width"], history=cfg["state"]["history"]), RewardConfig(brightness_target=cfg["reward"]["brightness_target"], brightness_p=cfg["reward"]["brightness_p"], w_mean=cfg["reward"]["weights"]["mean"], w_flicker=cfg["reward"]["weights"]["flicker"], w_sharpness=cfg["reward"]["weights"]["sharpness"]), EnvConfig(**cfg["env"]))


def infer(checkpoint, output, device="xpu", seeds=(0, 1, 2, 3, 4), names=None, selected_split="all"):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    cfg = load_config(checkpoint.parent / "config.yaml")
    dataset = Path(cfg["dataset"]["root"])
    dataset = (dataset if dataset.is_absolute() else PROJECT.parents[1] / dataset).resolve()
    if output == dataset or dataset in output.parents or (PROJECT / "third_party").resolve() in output.parents:
        raise ValueError("输出目录须位于数据集及第三方源码之外")
    if cfg["dataset"]["side"] != "left":
        raise ValueError("当前曝光误差使用左相机决策与参考")
    if device == "xpu" and not torch.xpu.is_available() or device == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"当前环境中 {device} 不可用")
    scenes = discover_scenes(dataset)
    train, validation = split_scenes(scenes, cfg["dataset"].get("eval_scenes"))
    paths = {"all": scenes, "eval": validation, "train": train}[selected_split]
    if names:
        missing = set(names) - {p.name for p in scenes}
        if missing:
            raise ValueError(f"场景不存在：{sorted(missing)}")
        paths = [p for p in scenes if p.name in names]
    if not paths or not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("须有有效场景及不重复的非负 seed")
    agent = DQNAgent.load(checkpoint, torch.device(device), seed_everything(cfg["seed"]))
    agent.q_net.eval()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for path in paths:
            env = build_env(cfg, path)
            reference_us, reference_gain = read_reference(path)
            for seed in seeds:
                state, _ = env.reset(seed=seed)
                for _ in range(env.env_config.max_episode_steps):
                    state, _, terminated, truncated, info = env.step(agent.act(state, greedy=True))
                    if terminated or truncated:
                        break
                gain = float(env.scene.spec.iso_indices[info["iso_idx"]])
                exposure = float(env.scene.spec.exposure_us[info["exp_idx"]])
                filename = env.scene.path(info["iso_idx"], info["exp_idx"]).name
                delta = signed_exposure_error(exposure, gain, reference_us, reference_gain)
                rows.append({"scene": path.name, "episode_seed": seed, "source_filename": filename, "gain_db": gain, "exposure_us": exposure, "signed_delta_ev": delta, "exposure_error_ev": abs(delta)})
                if seed == seeds[0]:
                    export_selection(path, output, "DQN-grid", filename, gain, exposure, seed)
            _imread_cached.cache_clear()
            print(f"已推理 {path.name}，{len(seeds)} 个初始 seed", flush=True)
    with (output / "exposure_error.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_error_metrics([r["exposure_error_ev"] for r in rows], "全部预声明回合的 Exposure error")
    evaluate_exposure(output, "DQN-grid", dataset, [p.name for p in paths])
    print("DQN 的 eval 场景参与过模型选择，不能作为独立测试成绩。")
    return output
