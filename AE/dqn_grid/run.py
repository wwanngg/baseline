import argparse
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "third_party/camera-exposure-rl"


def main():
    parser = argparse.ArgumentParser(description="第三方 DQN：训练、推理与 Exposure error 评价")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="调用本目录内的第三方训练代码")
    training.add_argument("--config", type=Path, default=PROJECT / "configs/original_xpu.yaml")
    inference = commands.add_parser("infer", help="使用检查点推理，并输出 Exposure error")
    inference.add_argument("--checkpoint", type=Path, default=PROJECT / "weights/policy_best.pt")
    inference.add_argument("--output", type=Path, default=PROJECT / "results")
    inference.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    inference.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    inference.add_argument("--scene", action="append", help="指定场景，可重复")
    inference.add_argument("--split", choices=("all", "eval", "train"), default="all")
    evaluation = commands.add_parser("evaluate", help="评价已有终态选图，只输出 Exposure error")
    evaluation.add_argument("--results", type=Path, default=PROJECT / "results")
    evaluation.add_argument("--dataset", type=Path, default=PROJECT.parent / "DataSet_AE")
    args = parser.parse_args()
    if args.command == "train":
        sys.argv = [str(SOURCE / "scripts/train.py"), "--config", str(args.config)]
        runpy.run_path(str(SOURCE / "scripts/train.py"), run_name="__main__")
    elif args.command == "infer":
        from inference import infer
        infer(args.checkpoint, args.output, args.device, args.seeds, args.scene, args.split)
    else:
        from exposure_error import evaluate_exposure
        evaluate_exposure(args.results, "DQN-grid", args.dataset)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        print(f"conda env 缺少依赖包：{error.name}。请自行安装后继续。", file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
