import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile
import cv2
import gymnasium
import numpy as np
import scipy
import stable_baselines3
import torch
from .controller import rollout
from .data import prepare_image, protect_output, samples, VARIANT_DIRS
from .estimator import Estimator, INITIAL
from .metrics import angular_error, error_summary, METRIC_NAMES, release_summary, reproduction_error
from .upstream import AWB_ROOT, load_model, MODEL_NAMES, resolve_device, UPSTREAM, upstream_modules
TARGETS = {'neutral': 'gt_illum', 'preference_cross': 'pref_illum'}

def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def versions():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'gymnasium': gymnasium.__version__, 'stable_baselines3': stable_baselines3.__version__}

def checkpoint_info(weights_dir, model_name, with_hash=False):
    path = (Path(weights_dir) / f'{model_name}_model.zip').resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing weight: {path}. Provide it yourself; no download is attempted.')
    with zipfile.ZipFile(path) as archive:
        for required in ('data', 'policy.pth', '_stable_baselines3_version'):
            if required not in archive.namelist():
                raise ValueError(f'Incomplete SB3 checkpoint {path}: no {required}')
        saved_version = archive.read('_stable_baselines3_version').decode().strip()
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sb3_archive_version': saved_version}
    if with_hash:
        result['sha256'] = sha256(path)
    return result

def inspect(dataset, weights_dir):
    upstream_modules()
    selected = samples(dataset)
    missing_masks = [sample.name for sample in selected if not sample.mask_path.is_file()]
    invalid_metadata = []
    for sample in selected:
        metadata = sample.metadata()
        for key in TARGETS.values():
            value = np.asarray(metadata.get(key), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
                invalid_metadata.append(f'{sample.name}:{key}')
    if invalid_metadata:
        raise ValueError(f'Invalid S24 references: {invalid_metadata}')
    print(json.dumps({'dataset': str(Path(dataset).resolve()), 'paired_samples': len(selected), 'cases': [f'{model}/{variant}' for model in MODEL_NAMES for variant in VARIANT_DIRS], 'weights': {model: checkpoint_info(weights_dir, model) for model in MODEL_NAMES}, 'missing_masks': missing_masks, 'versions': versions(), 'cuda_available': torch.cuda.is_available(), 'xpu_available': hasattr(torch, 'xpu') and torch.xpu.is_available(), 'training': False, 'downloads': False}, indent=2))

def score_prediction(prediction, initial, metadata):
    result = {}
    for target, key in TARGETS.items():
        reference = np.asarray(metadata[key], dtype=np.float64)
        if reference.shape != (3,) or not np.isfinite(reference).all() or np.any(reference <= 0):
            raise ValueError(f'Invalid reference: {key}')
        result[target] = {'reference_key': key, 'reference': reference.tolist(), 'recovery_deg': angular_error(prediction, reference), 'reproduction_deg': reproduction_error(prediction, reference), 'initial_recovery_deg': angular_error(initial, reference), 'initial_reproduction_deg': reproduction_error(initial, reference)}
    return result

def save_images(case_dir, name, image, prediction, metadata, bits):
    destination = case_dir / 'images'
    destination.mkdir(exist_ok=True)
    linear = image / (2 ** bits - 1)
    corrected = linear * (prediction[1] / prediction)[None, None, :]
    ccm = np.asarray(metadata['ccm'], dtype=np.float64)
    if ccm.shape != (3, 3) or not np.isfinite(ccm).all():
        raise ValueError('Invalid CCM for illustrative preview')
    preview = np.clip(corrected @ ccm.T, 0, 1) ** (1 / 2.2)
    for suffix, rgb, maximum, dtype in (('wb_linear.png', corrected, 65535, np.uint16), ('preview.png', preview, 255, np.uint8)):
        codes = np.round(np.clip(rgb[..., ::-1], 0, 1) * maximum).astype(dtype)
        ok, encoded = cv2.imencode('.png', codes)
        if not ok:
            raise OSError(f'Cannot encode preview for {name}')
        encoded.tofile(destination / f'{name}_{suffix}')

def evaluate_case(rows, model_name, variant):
    valid = [row for row in rows if row['status'] == 'ok']
    summary = {'model': model_name, 'variant': variant, 'attempted': len(rows), 'valid': len(valid), 'failed': len(rows) - len(valid), 'initial_gp_fallback': sum((row['stop_reason'] == 'initial_gp_fallback' for row in valid)), 'converged': sum((row['stop_reason'] == 'converged' for row in valid)), 'max_steps': sum((row['stop_reason'] == 'max_steps' for row in valid)), 'mean_steps': float(np.mean([row['steps'] for row in valid])) if valid else None, 'mean_runtime_s': float(np.mean([row['runtime_s'] for row in rows])) if rows else None, 'metrics': {}}
    for target in TARGETS:
        scores = [row['scores'][target] for row in valid]
        errors = [score['recovery_deg'] for score in scores]
        initial = [score['initial_recovery_deg'] for score in scores]
        summary['metrics'][target] = {'recovery': error_summary(errors), 'reproduction': error_summary([score['reproduction_deg'] for score in scores]), 'initial_recovery': error_summary(initial), 'initial_reproduction': error_summary([score['initial_reproduction_deg'] for score in scores]), 'release_aggregation_on_unified_recovery': release_summary(errors), 'improved': sum((final < start - 1e-06 for final, start in zip(errors, initial))), 'worsened': sum((final > start + 1e-06 for final, start in zip(errors, initial))), 'unchanged': sum((abs(final - start) <= 1e-06 for final, start in zip(errors, initial)))}
    return summary

def infer(args, output):
    if not 0 < args.scale <= 1 or args.max_steps < 1 or args.threads < 1:
        raise ValueError('Require 0 < scale <= 1, positive max_steps and threads')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    population = samples(args.dataset)
    selected = samples(args.dataset, args.image)
    if args.limit is not None:
        selected = selected[:args.limit]
    models = MODEL_NAMES if args.model == 'all' else (args.model,)
    variants = tuple(VARIANT_DIRS) if args.variant == 'both' else (args.variant,)
    weights = {name: checkpoint_info(args.weights_dir, name, with_hash=True) for name in models}
    output = protect_output(output, args.dataset, args.weights_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}. Choose a new --output directory to keep runs separate.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'format_version': 1, 'method': 'RL-AWB', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'dataset': str(Path(args.dataset).resolve()), 'dataset_role': 'official_validation', 'population': len(population), 'selected_ids': [sample.name for sample in selected], 'partial': len(selected) != len(population), 'models': list(models), 'variants': list(variants), 'expected_cases': [f'{name}/{variant}' for name in models for variant in variants], 'completed_cases': [], 'device': device, 'threads': args.threads, 'seed': args.seed, 'mask': args.mask, 'scale': args.scale, 'working_bits': args.working_bits, 'max_steps': args.max_steps, 'stop_threshold': 0.05, 'stop_consecutive': 3, 'resampling': 'original MATLAB-style nearest neighbor', 'code_conversion': f'uint16/65535*(2**{args.working_bits}-1), no black subtraction or gamma', 'input': 'demosaiced black-level-normalized linear camera RGB PNG', 'prediction_target': 'neutral', 'preference_evaluation': 'cross-evaluation of the same neutral predictions', 'initial_presets': {name: INITIAL[name] for name in models}, 'weights': weights, 'versions': versions(), 'save_images': args.save_images, 'source_hashes': {str(path.relative_to(AWB_ROOT)): sha256(path) for path in (UPSTREAM / 'env.py', UPSTREAM / 'utils/Algorithm.py', UPSTREAM / 'utils/WBsRGB.py', UPSTREAM / 'utils/sacpolicy.py')}, 'adaptations': ['array-only input without gt.mat or inference rewards', 'reset clock before observation to remove previous-image dependence', 'same S24 14-bit-equivalent preprocessing for all checkpoints by default', 'finite/positive prediction checks and explicit failures', 'shared percentile summaries, separate original aggregation fields']}
    write_json(output / 'manifest.json', manifest)
    try:
        for model_name in models:
            print(f"Loading {model_name}: {weights[model_name]['path']} on {device}", flush=True)
            model = load_model(weights[model_name]['path'], device, args.seed)
            with torch.inference_mode():
                for variant in variants:
                    case_dir = output / model_name / variant
                    case_dir.mkdir(parents=True, exist_ok=True)
                    rows = []
                    with (case_dir / 'predictions.jsonl').open('w', encoding='utf-8') as predictions, (case_dir / 'trace.jsonl').open('w', encoding='utf-8') as traces:
                        for position, sample in enumerate(selected, 1):
                            started = time.perf_counter()
                            row = {'image': sample.name, 'model': model_name, 'variant': variant, 'source': str(sample.image_path(variant)), 'status': 'failed'}
                            try:
                                image, invalid, original_shape = prepare_image(sample, variant, args.scale, args.working_bits, args.mask == 'with')
                                estimator = Estimator(image, invalid, model_name, args.working_bits)
                                result = rollout(model, estimator, args.max_steps)
                                metadata = sample.metadata()
                                scores = score_prediction(result.illuminant, result.initial_illuminant, metadata)
                                row.update(status='ok', prediction=result.illuminant.tolist(), initial_prediction=result.initial_illuminant.tolist(), scores=scores, original_shape=original_shape, working_shape=list(image.shape), steps=result.steps, stop_reason=result.stop_reason, fallback_steps=sum((not step['for_rl'] for step in result.trace)), gray_pixel_percent=result.trace[-1]['gray_pixel_percent'], minkowski_p=result.trace[-1]['minkowski_p'])
                                for step in result.trace:
                                    traces.write(json.dumps({'image': sample.name, **step}, allow_nan=False) + '\n')
                                if args.save_images:
                                    try:
                                        save_images(case_dir, sample.name, image, result.illuminant, metadata, args.working_bits)
                                    except (OSError, ValueError, KeyError) as error:
                                        row['preview_error'] = str(error)
                            except (OSError, ValueError, KeyError) as error:
                                row['error'] = str(error)
                            row['runtime_s'] = time.perf_counter() - started
                            predictions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            predictions.flush()
                            traces.flush()
                            rows.append(row)
                            detail = f"{row['stop_reason']}, steps={row['steps']}, neutral={scores['neutral']['recovery_deg']:.3f}deg" if row['status'] == 'ok' else row['error']
                            print(f'[{model_name}/{variant} {position}/{len(selected)}] {sample.name}: {detail}', flush=True)
                    metrics = evaluate_case(rows, model_name, variant)
                    write_json(case_dir / 'evaluation.json', metrics)
                    print_case_metrics(metrics)
                    manifest['completed_cases'].append(f'{model_name}/{variant}')
                    write_json(output / 'manifest.json', manifest)
            del model
            gc.collect()
        manifest['status'] = 'complete'
        write_json(output / 'manifest.json', manifest)
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(error) or type(error).__name__
        write_json(output / 'manifest.json', manifest)
        raise
    return output


def print_case_metrics(result):
    variant = "原 RAW" if result["variant"] == "raw" else "去噪 RAW"
    if "metrics" in result:
        cases = [(target, entry["recovery"]) for target, entry in result["metrics"].items()]
        valid, total = result["valid"], result["attempted"]
    else:
        cases = [(result["target"], result)]
        valid, total = result["valid_predictions"], result["selected_samples"]
    for target, metrics in cases:
        label = "中性目标" if target == "neutral" else "偏好目标"
        print(f"\n{result['model']} / {variant} / {label}：有效 {valid}/{total}，角误差单位：度")
        for key, name in (("mean_deg", "Mean"), ("median_deg", "Median"), ("best25_deg", "Best25%"), ("worst25_deg", "Worst25%"), ("worst5_deg", "Worst5%"), ("trimean_deg", "Tri-mean"), ("max_deg", "Max")):
            value = metrics[key]
            print(f"  {name}：{value:.6f}" if value is not None else f"  {name}：N/A")
