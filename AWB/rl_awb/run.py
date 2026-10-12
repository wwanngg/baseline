import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='使用 RL-AWB 本地权重推理 S24_val 配对图像')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='检查依赖、配对图像与权重文件')
    inspect.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    inspect.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    infer = commands.add_parser('infer', help='默认运行三个权重及原 RAW、去噪 RAW')
    infer.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    infer.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    infer.add_argument('--model', choices=('all', 'NCC', 'LEVI', 'Gehler'), default='all')
    infer.add_argument('--variant', choices=('both', 'raw', 'denoised'), default='both')
    infer.add_argument('--mask', choices=('without', 'with'), default='without')
    infer.add_argument('--scale', type=float, default=0.125, help='所有权重统一使用的最近邻缩放比例')
    infer.add_argument('--working-bits', type=int, choices=(12, 14, 16), default=14)
    infer.add_argument('--max-steps', type=int, default=100)
    infer.add_argument('--seed', type=int, default=531)
    infer.add_argument('--device', choices=('cpu', 'cuda', 'xpu', 'auto'), default='cpu')
    infer.add_argument('--threads', type=int, default=1)
    infer.add_argument('--image', action='append', help='指定图像 ID，例如 img_0000；可重复指定')
    infer.add_argument('--limit', type=int, help='小规模运行，报告标记为不完整')
    infer.add_argument('--output', type=Path, help='默认输出到本基线的 results/without_mask')
    infer.add_argument('--save-images', action='store_true', help='保存工作尺寸的校正线性 PNG 与 sRGB 预览')
    args = parser.parse_args()
    from rl_awb import workflows
    if args.command == 'inspect':
        workflows.inspect(args.dataset, args.weights_dir)
    if args.command == 'infer':
        output = args.output or PROJECT / 'results' / (args.mask + '_mask')
        print(workflows.infer(args, output))
if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as error:
        packages = {'numpy': 'numpy', 'cv2': 'opencv-python', 'scipy': 'scipy', 'torch': 'torch', 'gymnasium': 'gymnasium', 'stable_baselines3': 'stable-baselines3'}
        if error.name not in packages:
            raise
        print(f'conda env 缺少依赖包：{packages[error.name]}。请先自行安装，再继续运行。', file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('推理已中断，完成的逐图结果保留在输出目录。', file=sys.stderr)
        sys.exit(130)
