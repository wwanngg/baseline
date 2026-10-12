import argparse
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='使用已发布 Time-Aware AWB 权重推理 S24_val 原 RAW 与去噪 RAW')
    commands = parser.add_subparsers(dest='command', required=True)
    view = commands.add_parser('inspect', help='检查配对图像与本地权重，不执行完整推理')
    view.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    view.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    infer = commands.add_parser('infer', help='使用本地已发布权重进行推理')
    infer.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    infer.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    infer.add_argument('--model', choices=('awb', 'awb-w-noise', 'awb-w-snr', 'awb-w-noise-snr'), default='awb-w-noise-snr')
    infer.add_argument('--variant', choices=('raw', 'denoised', 'both'), default='both')
    infer.add_argument('--target', choices=('neutral', 'preference', 'both'), default='both')
    infer.add_argument('--mask', choices=('without', 'with'), default='without')
    infer.add_argument('--device', choices=('cpu', 'cuda', 'xpu', 'auto'), default='cpu')
    infer.add_argument('--output', type=Path, help='默认输出到本基线的 results/without_mask 或 results/with_mask')
    infer.add_argument('--image', action='append', help='指定图像 ID，例如 img_0000；可重复指定')
    infer.add_argument('--limit', type=int, help='小规模检查：只运行前 N 张，结果标记为不完整')
    infer.add_argument('--threads', type=int, default=1)
    infer.add_argument('--save-images', action='store_true', help='保存缩小的白平衡 RAW 和 sRGB 预览图')
    args = parser.parse_args()
    from time_aware_awb import workflows
    if args.command == 'inspect':
        workflows.inspect(args.dataset, args.weights_dir)
    if args.command == 'infer':
        output = args.output or PROJECT / 'results' / (args.mask + '_mask')
        print(workflows.infer(args.dataset, output, args.weights_dir, args.model, args.variant, args.target, args.mask, args.device, args.image, args.limit, args.save_images, args.threads))
if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as error:
        packages = {'numpy': 'numpy', 'cv2': 'opencv-python', 'torch': 'torch'}
        if error.name not in packages:
            raise
        print(f'conda env 缺少依赖包：{packages[error.name]}。请先自行安装，再继续运行。', file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
