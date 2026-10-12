import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='使用 GCC 本地权重推理 S24_val，默认设备 XPU')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('inspect', help='检查图像配对、本地权重和 XPU，不执行推理')
    check.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    check.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    check.add_argument('--verify-hashes', action='store_true', help='按本地下载记录核对六个权重文件的摘要')
    inference = commands.add_parser('infer', help='默认运行两个权重及原 RAW、去噪 RAW')
    inference.add_argument('--dataset', type=Path, default=PROJECT.parent / 'S24_val')
    inference.add_argument('--weights-dir', type=Path, default=PROJECT / 'weights')
    inference.add_argument('--model', choices=('both', 'nus8', 'gehler'), default='both')
    inference.add_argument('--variant', choices=('both', 'raw', 'denoised'), default='both')
    inference.add_argument('--device', choices=('xpu', 'cpu', 'cuda', 'auto'), default='xpu')
    inference.add_argument('--dtype', choices=('auto', 'float16', 'float32', 'bfloat16'), default='auto')
    inference.add_argument('--mask', choices=('without', 'with'), default='without')
    inference.add_argument('--batch-size', type=int, default=1)
    inference.add_argument('--threads', type=int, default=1)
    inference.add_argument('--seed', type=int, default=42)
    inference.add_argument('--image', action='append', help='指定图像 ID，例如 img_0000；可重复指定')
    inference.add_argument('--limit', type=int, help='小规模运行，结果明确标记为不完整')
    inference.add_argument('--output', type=Path)
    inference.add_argument('--chart', type=Path, default=PROJECT / 'third_party/GCC-official/color_chart.jpg')
    inference.add_argument('--save-images', action='store_true', help='保存 512 像素色卡生成图、校正线性 RGB 和 CCM 预览')
    args = parser.parse_args()
    from gcc_awb import workflows
    if args.command == 'inspect':
        workflows.inspect(args.dataset, args.weights_dir, args.verify_hashes)
    if args.command == 'infer':
        output = args.output or PROJECT / 'results' / (args.mask + '_mask')
        print(workflows.infer(args, output))
if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as error:
        packages = {'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy', 'cv2': 'opencv-python', 'diffusers': 'diffusers', 'transformers': 'transformers', 'accelerate': 'accelerate', 'huggingface_hub': 'huggingface_hub', 'safetensors': 'safetensors', 'PIL': 'Pillow'}
        if error.name not in packages:
            raise
        print(f'conda env 缺少依赖包：{packages[error.name]}。请先自行安装，再继续运行。', file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('推理已中断，完成的记录保留在输出目录。', file=sys.stderr)
        sys.exit(130)
