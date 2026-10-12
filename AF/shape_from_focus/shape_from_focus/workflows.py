import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .dataset import describe, load_stack, protect_output, scene_paths
from .focus import FocusConfig, reconstruct

def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Cannot write an empty table')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def save_plots(folder: Path, result: dict, scene: str, frames: int):
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, key, title in zip(axes[:2], ('best_frame_index', 'refined_frame_index'), ('Discrete focus index', 'Refined focus index')):
        data = np.ma.masked_where(~result['valid_mask'], result[key])
        plot = axis.imshow(data, cmap='viridis', vmin=0, vmax=max(frames - 1, 1))
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(plot, ax=axis, label='Frame index (0-based)', shrink=0.8)
    image = result['all_in_focus']
    axes[2].imshow(image, cmap='gray', vmin=0, vmax=max(float(image.max()), 1))
    axes[2].set_title('All-in-focus (discrete selection)')
    axes[2].set_axis_off()
    figure.suptitle(f'{scene}: index units, not calibrated depth')
    figure.savefig(folder / 'focus_maps.png', dpi=130)
    plt.close(figure)
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection='3d')
    stride = max(1, int(np.ceil(max(result['valid_mask'].shape) / 120)))
    z = result['refined_frame_index'][::stride, ::stride]
    yy, xx = np.meshgrid(np.arange(result['valid_mask'].shape[0])[::stride], np.arange(result['valid_mask'].shape[1])[::stride], indexing='ij')
    if np.isfinite(z).any():
        axis.plot_surface(xx, yy, z, cmap='viridis', linewidth=0)
    axis.set_xlabel('Column')
    axis.set_ylabel('Row')
    axis.set_zlabel('Focus frame index')
    axis.set_title(scene)
    figure.tight_layout()
    figure.savefig(folder / 'focus_surface.png', dpi=130)
    plt.close(figure)

def run(dataset: Path, output: Path, config: FocusConfig, scene: str | None=None, key: str | None=None, frame_limit: int | None=None, crop_size: int | None=None, plots: bool=True) -> Path:
    output = protect_output(output, dataset)
    paths = scene_paths(dataset, scene)
    if key is not None and len(paths) > 1:
        raise ValueError('Use --scene when overriding --key')
    if plots:
        import matplotlib
    infos = [describe(path, key) for path in paths]
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for info in infos:
        path = Path(info.path)
        stack, metadata = load_stack(path, key, frame_limit, crop_size)
        print(f"{path.stem}: key={metadata['key']}; shape={stack.shape}; mode={config.mode}; interpolation={config.interpolation}; partial={metadata['partial_stack']}", flush=True)

        def progress(stage, current, total):
            if current == 1 or current == total or current % 10 == 0:
                print(f'  {stage}: {current}/{total}', flush=True)
        started = perf_counter()
        result = reconstruct(stack, config, progress)
        computation_seconds = perf_counter() - started
        folder = protect_output(output / path.stem, dataset)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(folder / 'focus_result.npz', **arrays)
        write_csv(folder / 'focus_curve.csv', result['frame_scores'])
        valid = result['valid_mask']
        valid_count, eligible = (int(valid.sum()), int(result['spatial_mask'].sum()))
        indices = result['best_frame_index'][valid]
        row = {'scene': path.stem, 'mode': config.mode, 'interpolation': config.interpolation, 'height': stack.shape[0], 'width': stack.shape[1], 'frames': stack.shape[2], 'partial_stack': metadata['partial_stack'], 'spatially_eligible_pixels': eligible, 'valid_pixels': valid_count, 'valid_fraction_of_eligible': valid_count / eligible, 'interpolated_pixels': int(result['interpolated_mask'].sum()), 'interpolated_fraction_of_valid': float(result['interpolated_mask'].sum()) / valid_count if valid_count else 0.0, 'min_focus_index': int(indices.min()) if valid_count else None, 'max_focus_index': int(indices.max()) if valid_count else None, 'mean_peak_sml': float(result['focus_peak'][valid].mean()) if valid_count else None, 'global_focus_frame_index': int(np.argmax([r['mean_sml'] for r in result['frame_scores']])), 'computation_seconds': computation_seconds}
        write_json(folder / 'result.json', {**row, 'dataset': metadata, 'config': asdict(config), 'accuracy_metrics_available': False, 'accuracy_note': 'Input MAT files contain only synthetic image stacks, without ground-truth depths or stage calibration. Coverage and focus scores are diagnostics, not depth accuracy or autofocus success metrics.', 'invalid_pixel_encoding': 'best_frame_index=-1; refined_frame_index=NaN; consult valid_mask', 'all_in_focus_note': 'Select the discrete best frame per pixel; Gaussian interpolation refines indices only.', 'gaussian_note': 'Assumes equally spaced frames. Falls back to the discrete index at endpoints, plateaus, nonpositive neighbours and ill-conditioned log fits.'})
        if plots:
            save_plots(folder, result, path.stem, stack.shape[2])
        summaries.append(row)
        print(f'  有效覆盖率={valid_count / eligible:.2%}（{valid_count}/{eligible}）；纯计算耗时={computation_seconds:.3f}s；MAE=N/A；RMSE=N/A', flush=True)
    write_csv(output / 'summary.csv', summaries)
    write_json(output / 'summary.json', {'dataset': str(Path(dataset).resolve()), 'config': asdict(config), 'scenes': summaries, 'partial_stack': any((r['partial_stack'] for r in summaries)), 'index_units': 'zero-based frame index; not physical depth', 'accuracy_metrics_available': False})
    return output
