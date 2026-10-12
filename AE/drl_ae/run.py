import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT.parent / "DataSet_AE"


def main():
    parser = argparse.ArgumentParser(description="DRL-AE：稳定 SAC 训练、推理与 Exposure error 评价")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="从零训练 SAC")
    training.add_argument("--config", type=Path, default=PROJECT / "configs/config.yaml")
    training.add_argument("--steps", type=int, help="训练步数预算")
    training.add_argument("--device", choices=("cpu", "xpu", "cuda", "auto"), default="auto")
    inference = commands.add_parser("infer", help="使用检查点推理并评价曝光误差")
    inference.add_argument("--checkpoint", type=Path, default=PROJECT / "weights/policy_best.pt")
    inference.add_argument("--dataset", type=Path, default=DATASET)
    inference.add_argument("--output", type=Path, default=PROJECT / "results")
    inference.add_argument("--split", choices=("all", "eval", "validation", "train"), default="all")
    inference.add_argument("--scene", action="append")
    inference.add_argument("--episodes-per-scene", type=int, default=5)
    inference.add_argument("--device", choices=("cpu", "xpu", "cuda", "auto"), default="auto")
    evaluation = commands.add_parser("evaluate", help="评价已有选图，只输出 Exposure error")
    evaluation.add_argument("--dataset", type=Path, default=DATASET)
    evaluation.add_argument("--results", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    if args.command == "train":
        from drl_ae.config import load_config
        from drl_ae.workflows import train
        cfg = load_config(args.config)
        cfg["device"] = args.device
        if args.steps is not None:
            cfg["train"]["steps"] = args.steps
        train(cfg)
    elif args.command == "infer":
        from drl_ae.inference import infer
        infer(args.checkpoint, args.output, args.dataset, args.scene, args.split, args.episodes_per_scene, args.device)
    else:
        from exposure_error import evaluate_exposure
        evaluate_exposure(args.results, "DRL-AE", args.dataset)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        print(f"conda env 缺少依赖包：{error.name}。请自行安装后继续。", file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
