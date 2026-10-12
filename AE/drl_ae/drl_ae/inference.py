import csv
from pathlib import Path
import sys

from .config import device_name, make_env, split
from .dataset import PROJECT, protect_output, scenes
from .sac import SACAgent

from exposure_error import evaluate_exposure, export_selection, print_error_metrics, read_reference, signed_exposure_error


def infer(checkpoint, output, dataset=None, names=None, selected_split="all", episodes_per_scene=5, device="cpu"):
    if episodes_per_scene < 1:
        raise ValueError("每场景回合数须为正整数")
    agent, saved = SACAgent.load(Path(checkpoint), device_name(device))
    if saved["training_steps"] < 1 or agent.updates < 1:
        raise ValueError("检查点没有完成训练更新")
    cfg = saved["run_config"]
    if cfg["dataset"]["side"] != "left":
        raise ValueError("当前曝光误差使用左相机决策与内置 AE 参考")
    cfg["dataset"]["root"] = str(Path(dataset or PROJECT.parent / "DataSet_AE").resolve())
    train, validation, held_out = split(cfg)
    paths = scenes(Path(cfg["dataset"]["root"]), names) if names else {"all": train + validation + held_out, "train": train, "validation": validation, "eval": held_out}[selected_split]
    output = protect_output(Path(output), Path(cfg["dataset"]["root"]))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        env = make_env(path, cfg)
        if env.state_dim != agent.state_dim:
            raise ValueError("检查点与输入状态维数不匹配")
        reference_us, reference_gain = read_reference(path)
        for episode in range(episodes_per_scene):
            seed = cfg["seed"] + episode
            observation, _ = env.reset(seed=seed)
            for _ in range(env.config.max_steps):
                observation, _, terminated, truncated, info = env.step(agent.act(observation, deterministic=True))
                if info["success"] or terminated or truncated:
                    break
            gain, exposure = info["measured_gain_db"], info["measured_exposure_us"]
            delta = signed_exposure_error(exposure, gain, reference_us, reference_gain)
            rows.append({"scene": path.name, "episode_seed": seed, "gain_db": gain, "exposure_us": exposure, "source_filename": info["source_filename"], "signed_delta_ev": delta, "exposure_error_ev": abs(delta)})
            if episode == 0:
                export_selection(path, output, "DRL-AE", info["source_filename"], gain, exposure, seed)
        print(f"已推理 {path.name}，{episodes_per_scene} 个预声明回合", flush=True)
    with (output / "exposure_error.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_error_metrics([row["exposure_error_ev"] for row in rows], "全部预声明回合的 Exposure error")
    evaluate_exposure(output, "DRL-AE", cfg["dataset"]["root"], [p.name for p in paths])
    return output
