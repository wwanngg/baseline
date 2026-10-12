import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from .data import AWB_ROOT, DEFAULT_WEIGHTS, VARIANT_DIRS, prepare_image, protect_output, samples
from .features import capture_features, histogram_features
from .metrics import angular_error, angular_error_upstream, error_summary
from .model import MODEL_NAMES, Predictor, device_name

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def write_csv(path: Path, rows):
    if not rows:
        raise ValueError('No rows to save')
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def file_hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(dataset, weights_dir=DEFAULT_WEIGHTS):
    indexed = samples(dataset)
    missing = {variant: [s.name for s in indexed if not s.image_path(variant).is_file()] for variant in VARIANT_DIRS}
    result = {'dataset': str(Path(dataset).resolve()), 'official_split': 'validation', 'metadata_samples': len(indexed), 'variant_pairs': {key: len(indexed) - len(names) for key, names in missing.items()}, 'missing_images': missing, 'missing_masks': [s.name for s in indexed if not s.mask_path.is_file()], 'pretrained_files': {name + suffix: all(((Path(weights_dir) / f'{prefix}-{name}{suffix}.{extension}').is_file() for prefix, extension in (('config', 'json'), ('model', 'pt')))) for name in MODEL_NAMES for suffix in ('', '-pref')}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def save_preview(folder, name, rgb, prediction, metadata):
    illum = np.asarray(prediction)
    if np.any(illum <= 0):
        return 'skipped_nonpositive_illuminant'
    wb = np.clip(rgb * (illum[1] / illum), 0, 1)
    raw16 = (wb * 65535).astype(np.uint16)
    matrix = np.asarray(metadata['ccm'], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid CCM for preview')
    linear_srgb = np.clip(rgb * (illum[1] / illum) @ matrix.T, 0, 1)
    preview = (linear_srgb ** (1 / 2.2) * 255).astype(np.uint8)
    for kind, extension, image in (('wb_raw16', '.png', raw16), ('preview', '.jpg', preview)):
        target = folder / kind / (name + extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(extension, image[:, :, ::-1])
        if not ok:
            raise OSError(f'Cannot encode output: {target}')
        encoded.tofile(target)
    return 'saved_downsampled_preview'

def evaluate_case(rows, case, total, partial):
    valid = [row for row in rows if row['status'] == 'ok']
    primary = error_summary([row['angular_error_deg'] for row in valid])
    compatible = error_summary([row['angular_error_deg_upstream'] for row in valid])
    return {'method': 'Time-Aware AWB', 'dataset': 'S24_val_' + case['variant'], 'variant': case['variant'], 'target': case['target'], 'model': case['model'], 'mask': case['mask'], 'official_split': 'validation', 'dataset_samples': total, 'selected_samples': len(rows), 'valid_predictions': len(valid), 'failed_predictions': len(rows) - len(valid), 'partial_set': partial, 'parameters': case['parameters'], **primary, **{'upstream_' + key: value for key, value in compatible.items()}, 'mean_model_ms': float(np.mean([r['model_ms'] for r in valid])) if valid else None}

def infer(dataset: Path, output: Path, weights_dir: Path, model_name='awb-w-noise-snr', variant='both', target='both', mask='without', device='cpu', names=None, limit=None, save_images=False, threads=1):
    if mask not in ('with', 'without') or threads < 1 or (limit is not None and limit < 1):
        raise ValueError('Invalid mask, CPU thread count or sample limit')
    all_samples = samples(dataset)
    selected = samples(dataset, names)
    if limit is not None:
        selected = selected[:limit]
    variants = tuple(VARIANT_DIRS) if variant == 'both' else (variant,)
    targets = ('neutral', 'preference') if target == 'both' else (target,)
    if any((v not in VARIANT_DIRS for v in variants)) or any((t not in ('neutral', 'preference') for t in targets)):
        raise ValueError('Invalid dataset variant or ground-truth target')
    for sample in selected:
        for v in variants:
            if not sample.image_path(v).is_file():
                raise FileNotFoundError(sample.image_path(v))
        if mask == 'with' and (not sample.mask_path.is_file()):
            raise FileNotFoundError(sample.mask_path)
    output = protect_output(output, dataset)
    if (output / 'manifest.json').is_file():
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if previous['mask'] != mask or previous['dataset_root'] != str(Path(dataset).resolve()):
            raise ValueError('Use a different --output for a different mask policy or dataset')
    torch.set_num_threads(threads)
    device = device_name(device)
    predictors = {t: Predictor(weights_dir, model_name, t, device) for t in targets}
    partial = len(selected) != len(all_samples)
    manifest = {'schema': 1, 'dataset_root': str(Path(dataset).resolve()), 'official_split': 'validation', 'dataset_samples': len(all_samples), 'selected_names': [s.name for s in selected], 'partial_set': partial, 'mask': mask, 'device': device, 'cpu_threads': threads, 'model_name': model_name, 'status': 'running', 'cases': [], 'capture_metadata_policy': 'Same supplied JSON metadata for raw and denoised; noise/SNR stats are not recomputed', 'primary_angular_error': 'float64 cosine, acos and exact pi, degrees', 'upstream_angular_error': 'released float32 + norm epsilon, PI=22/7, shrink=False', 'preprocessing': 'RGB float64; bicubic resize; optional aligned nearest mask; source float32/uint16 roundtrip', 'torch_version': torch.__version__}
    write_json(output / 'manifest.json', manifest)
    summaries = []
    for v in variants:
        for t, predictor in predictors.items():
            folder = protect_output(output / v / t / model_name, dataset)
            case = {'variant': v, 'target': t, 'model': model_name, 'mask': mask, 'folder': str(folder.relative_to(output)), 'parameters': predictor.parameters, 'weights_file': str(predictor.weights_path.resolve()), 'weights_sha256': file_hash(predictor.weights_path), 'config_file': str(predictor.config_path.resolve()), 'config_sha256': file_hash(predictor.config_path), 'capture_features': predictor.config['capture_data'], 'target_size': predictor.config['target_size'], 'hist_bins': predictor.config['hist_bins']}
            print(f'{v}/{t}: model={predictor.experiment}; params={predictor.parameters}; selected={len(selected)}/{len(all_samples)}; device={device}', flush=True)
            rows = []
            for i, sample in enumerate(selected, 1):
                row = {'image_id': sample.name, 'variant': v, 'target': t, 'model': model_name, 'status': 'failed', 'angular_error_deg': None, 'angular_error_deg_upstream': None}
                try:
                    metadata = sample.metadata()
                    started = perf_counter()
                    image, display, shape = prepare_image(sample, v, predictor.config['target_size'], mask == 'with')
                    histogram = histogram_features(image, predictor.config)
                    capture = capture_features(metadata, predictor.config)
                    feature_ms = (perf_counter() - started) * 1000
                    started = perf_counter()
                    estimate = predictor.predict(histogram, capture)
                    model_ms = (perf_counter() - started) * 1000
                    ground_truth = metadata['gt_illum' if t == 'neutral' else 'pref_illum']
                    row.update(status='ok', pred_r=float(estimate[0]), pred_g=float(estimate[1]), pred_b=float(estimate[2]), gt_r=float(ground_truth[0]), gt_g=float(ground_truth[1]), gt_b=float(ground_truth[2]), angular_error_deg=angular_error(estimate, ground_truth), angular_error_deg_upstream=angular_error_upstream(estimate, ground_truth), neutral_error_deg=angular_error(estimate, metadata['gt_illum']), preference_error_deg=angular_error(estimate, metadata['pref_illum']), feature_ms=feature_ms, model_ms=model_ms, scene_class=metadata.get('scene_class'), light_class=metadata.get('light_class'))
                    prediction = {**row, 'est_illum': estimate.tolist(), 'ground_truth': ground_truth, 'source_image': str(sample.image_path(v)), 'metadata_file': str(sample.metadata_path), 'original_shape': shape, 'inference_shape': list(image.shape), 'capture_metadata_policy': manifest['capture_metadata_policy']}
                    if save_images:
                        try:
                            prediction['preview_status'] = save_preview(folder, sample.name, display, estimate, metadata)
                        except (OSError, ValueError, KeyError) as error:
                            prediction['preview_status'] = f'failed: {error}'
                    write_json(folder / 'predictions' / f'{sample.name}.json', prediction)
                except (OSError, ValueError, KeyError) as error:
                    row.update(status='failed', error=str(error))
                    write_json(folder / 'predictions' / f'{sample.name}.json', row)
                rows.append(row)
                if i == 1 or i == len(selected) or i % 20 == 0:
                    print(f"  {i}/{len(selected)}: {sample.name}; status={row['status']}", flush=True)
            write_csv(folder / 'per_image.csv', rows)
            summary = evaluate_case(rows, case, len(all_samples), partial)
            write_json(folder / 'evaluation.json', summary)
            print_case_metrics(summary)
            manifest['cases'].append(case)
            summaries.append(summary)
            write_json(output / 'manifest.json', manifest)
    manifest['status'] = 'completed'
    write_json(output / 'manifest.json', manifest)
    write_csv(output / 'evaluation.csv', summaries)
    write_json(output / 'evaluation.json', {'manifest': str(output / 'manifest.json'), 'partial_set': partial, 'cases': summaries})
    return output / 'evaluation.json'


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
