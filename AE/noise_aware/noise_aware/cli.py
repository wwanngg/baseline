import argparse
from pathlib import Path
import sys
from .dataset import DEFAULT_DATASET, DEFAULT_OUTPUT, discover_scenes
from .metric import MetricConfig

def main(argv=None):
    parser = argparse.ArgumentParser(description="Noise-Aware AE：离线搜索准备、推理和 Exposure error 评价")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (("prepare", "计算算法搜索所需的评分曲面"), ("infer", "运行 Nelder–Mead 曝光控制"), ("evaluate", "评价已有终态选图，只输出 Exposure error")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        sub.add_argument("--scene", action="append", help="指定场景，可重复")
        if command == "evaluate":
            sub.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
        else:
            sub.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if command == "prepare":
            sub.add_argument("--max-images", type=int, help="仅用于最小检查，部分曲面不能用于正式控制")
            sub.add_argument("--alpha", type=float, default=0.4)
            sub.add_argument("--beta", type=float, default=0.4)
            sub.add_argument("--resize", type=float, default=1.0)
        elif command == "infer":
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--max-evaluations", type=int, default=50)
            sub.add_argument("--initial", nargs=2, type=float, metavar=("GAIN_DB", "EXPOSURE_US"))
    args = parser.parse_args(argv)
    from exposure_error import evaluate_exposure
    if args.command == "evaluate":
        evaluate_exposure(args.results, "NM", args.dataset, args.scene)
        return 0
    scenes = discover_scenes(args.dataset, args.scene)
    for scene in scenes:
        if args.command == "prepare":
            from .evaluation import evaluate_scene
            evaluate_scene(scene, args.output, MetricConfig(alpha=args.alpha, beta=args.beta, resize_factor=args.resize), max_images=args.max_images)
        else:
            from .control import ControlConfig, control_scene
            control_scene(scene, args.output, ControlConfig(seed=args.seed, max_evaluations=args.max_evaluations), tuple(args.initial) if args.initial else None, plots=False)
            print(f"已完成 {scene.name} 的曝光控制")
    if args.command == "infer":
        evaluate_exposure(args.output, "NM", args.dataset, [p.name for p in scenes])
    return 0
