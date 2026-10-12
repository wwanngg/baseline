import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT.parent / "DataSet_AE"

def main():
    parser = argparse.ArgumentParser(description="Entropy AE：推理及 Exposure error 评价")
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("infer", help="选择曝光图像并输出曝光误差")
    inference.add_argument("--dataset", type=Path, default=DATASET)
    inference.add_argument("--output", type=Path, default=PROJECT / "results")
    inference.add_argument("--scene", action="append")
    inference.add_argument("--gain-db", type=float, default=0)
    inference.add_argument("--sample-stride", type=int, default=8)
    inference.add_argument("--target", type=float, default=0.13)
    inference.add_argument("--high-threshold", type=float, default=0.9)
    inference.add_argument("--low-threshold", type=float, default=0)
    inference.add_argument("--max-images", type=int, help="只用于最小检查")
    evaluation = commands.add_parser("evaluate", help="评价已有终态选图，只输出 Exposure error")
    evaluation.add_argument("--dataset", type=Path, default=DATASET)
    evaluation.add_argument("--results", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    from exposure_error import evaluate_exposure
    if args.command == "evaluate":
        evaluate_exposure(args.results, "Entropy", args.dataset)
        return
    from dataset_ae import discover_scenes
    from evaluation import infer_scene
    scenes = discover_scenes(args.dataset, args.scene)
    for scene in scenes:
        infer_scene(scene, args.output, args.gain_db, args.sample_stride, args.target, args.low_threshold, args.high_threshold, args.max_images)
    evaluate_exposure(args.output, "Entropy", args.dataset, [p.name for p in scenes])

if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        print(f"conda env 缺少依赖包：{error.name}。请自行安装后继续。", file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
