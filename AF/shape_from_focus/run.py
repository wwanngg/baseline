import argparse
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT.parent / 'DataSet_AF'

def main():
    parser = argparse.ArgumentParser(description='在原位置的图像栈上运行 SML 焦点评分与焦点重建')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='检查数据集信息，不执行重建')
    inspect.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    inspect.add_argument('--scene', help='MAT 文件名，例如 xsin 或 xsim；默认检查全部场景')
    inspect.add_argument('--key', help='指定 MATLAB 图像栈变量')
    compute = commands.add_parser('run', help='重建图像栈')
    compute.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    compute.add_argument('--scene', help='MAT 文件名；默认运行全部场景')
    compute.add_argument('--key', help='指定 MATLAB 图像栈变量，需同时指定单个场景')
    compute.add_argument('--mode', choices=('paper', 'upstream'), default='paper')
    compute.add_argument('--window', type=int, help='SML 奇数窗口大小，默认 5')
    compute.add_argument('--threshold', type=float, help='原始像素强度下的 ML 阈值，默认 7')
    compute.add_argument('--step', type=int, help='空间差分间隔，默认 1')
    compute.add_argument('--interpolation', choices=('gaussian', 'none'), help='paper 模式默认高斯插值，upstream 模式不插值')
    compute.add_argument('--output', type=Path, help='默认输出到本基线的 results/<mode>')
    compute.add_argument('--max-frames', type=int, help='小规模检查：只运行前 N 帧，结果标记为不完整')
    compute.add_argument('--crop-size', type=int, help='小规模检查：中心正方形裁剪，结果标记为不完整')
    compute.add_argument('--no-plots', action='store_true', help='只保存数值结果，不绘制图表')
    args = parser.parse_args()
    from shape_from_focus.dataset import describe, scene_paths
    if args.command == 'inspect':
        for path in scene_paths(args.dataset, args.scene):
            info = describe(path, args.key)
            print(f'{path.stem}: key={info.key}；高×宽×帧数={info.shape}；数据类型={info.dtype}；路径={info.path}')
        print('原 MAT 文件不包含真实深度和台面位移标定。')
    else:
        from shape_from_focus.focus import FocusConfig
        from shape_from_focus.workflows import run
        config = FocusConfig.from_options(args.mode, args.window, args.threshold, args.step, args.interpolation)
        output = args.output or PROJECT / 'results' / config.mode
        print(run(args.dataset, output, config, args.scene, args.key, args.max_frames, args.crop_size, not args.no_plots))
if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as error:
        packages = {'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib'}
        if error.name not in packages:
            raise
        print(f'conda env 缺少依赖包：{packages[error.name]}。请先自行安装，再继续运行。', file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, KeyError) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
