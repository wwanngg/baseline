import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from .data import estimate_illuminant, load_chart, prepare_image, protect_output, samples, VARIANT_DIRS
from .metrics import angular_error, error_summary, METRIC_NAMES, release_summary
from .model import Predictor, PROMPT, resolve_device
from .upstream import AWB_ROOT, MODEL_NAMES, modules, sha256, UPSTREAM, WEIGHT_FILES, weights_path
TARGETS = {'neutral': 'gt_illum', 'preference_cross': 'pref_illum'}

def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'huggingface_hub', 'safetensors', 'numpy', 'opencv-python', 'Pillow')}

def weights_info(weights_dir, model, verify=False):
    root = weights_path(weights_dir, model)
    files = {}
    for relative, expected_size in WEIGHT_FILES.items():
        metadata = root / '.cache/huggingface/download' / (relative + '.metadata')
        record = {'bytes': expected_size}
        if metadata.is_file():
            lines = metadata.read_text().splitlines()
            record.update(revision=lines[0], download_sha256=lines[1])
        if verify:
            record['sha256'] = sha256(root / relative)
            if 'download_sha256' in record and record['sha256'] != record['download_sha256']:
                raise ValueError(f'Weight SHA256 mismatch: {root / relative}')
        files[relative] = record
    return {'path': str(root), 'model_index_sha256': sha256(root / 'model_index.json'), 'large_files': files}

def inspect(dataset, weights_dir, verify=False):
    indexed = samples(dataset)
    weights = {name: weights_info(weights_dir, name, verify) for name in MODEL_NAMES}
    report = {'paired_samples': len(indexed), 'dataset': str(Path(dataset).resolve()), 'weights': weights, 'versions': versions(), 'xpu_available': hasattr(torch, 'xpu') and torch.xpu.is_available(), 'cuda_available': torch.cuda.is_available(), 'training': False, 'downloads': False}
    if report['xpu_available']:
        report['xpu_device'] = str(torch.xpu.get_device_properties(0))
    print(json.dumps(report, indent=2))
    return report

def score_prediction(prediction, source_estimate, metadata):
    _, original, _ = modules()
    scores = {}
    for target, key in TARGETS.items():
        reference = np.asarray(metadata[key], dtype=np.float64)
        if reference.shape != (3,) or not np.isfinite(reference).all() or np.any(reference <= 0):
            raise ValueError(f'Invalid S24 reference: {key}')
        scores[target] = {'reference_key': key, 'reference': reference.tolist(), 'recovery_deg': angular_error(prediction, reference), 'source_recovery_deg': float(original.calculate_angular_error(reference, source_estimate))}
    return scores

def save_images(destination, name, generated, linear_codes, prediction, metadata):
    destination.mkdir(exist_ok=True)
    generated.save(destination / f'{name}_generated_checker.png')
    corrected = linear_codes.astype(np.float64) / 65535 * (prediction[1] / prediction)[None, None, :]
    ccm = np.asarray(metadata['ccm'], dtype=np.float64)
    if ccm.shape != (3, 3) or not np.isfinite(ccm).all():
        raise ValueError('Invalid S24 CCM for preview')
    preview = np.clip(corrected @ ccm.T, 0, 1) ** (1 / 2.2)
    for suffix, array, maximum, dtype in (('wb_linear.png', corrected, 65535, np.uint16), ('preview.png', preview, 255, np.uint8)):
        bgr = np.round(np.clip(array[..., ::-1], 0, 1) * maximum).astype(dtype)
        ok, encoded = cv2.imencode('.png', bgr)
        if not ok:
            raise OSError('Cannot encode GCC result preview')
        encoded.tofile(destination / f'{name}_{suffix}')

def evaluate_case(rows, model, variant):
    valid = [row for row in rows if row['status'] == 'ok']
    return {'model': model, 'variant': variant, 'attempted': len(rows), 'valid': len(valid), 'failed': len(rows) - len(valid), 'metrics': {target: {'recovery': error_summary([row['scores'][target]['recovery_deg'] for row in valid]), 'source_recovery': release_summary([row['scores'][target]['source_recovery_deg'] for row in valid])} for target in TARGETS}}

def infer(args, output):
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Batch size, threads and limit must be positive')
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    population = samples(args.dataset)
    selected = samples(args.dataset, args.image)
    if args.limit is not None:
        selected = selected[:args.limit]
    models = MODEL_NAMES if args.model == 'both' else (args.model,)
    variants = tuple(VARIANT_DIRS) if args.variant == 'both' else (args.variant,)
    weights = {name: weights_info(args.weights_dir, name) for name in models}
    chart = load_chart(args.chart)
    output = protect_output(output, args.dataset)
    weights_root = Path(args.weights_dir).resolve()
    if output == weights_root or weights_root in output.parents:
        raise ValueError('Output must be outside the checkpoint directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}. Choose a new --output directory.')
    output.mkdir(parents=True, exist_ok=True)
    dtype = ('float32' if device == 'cpu' else 'float16') if args.dtype == 'auto' else args.dtype
    manifest = {'format_version': 1, 'method': 'GCC', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'dataset': str(Path(args.dataset).resolve()), 'dataset_role': 'official_validation', 'population': len(population), 'selected_ids': [sample.name for sample in selected], 'partial': len(selected) != len(population), 'models': list(models), 'variants': list(variants), 'expected_cases': [f'{name}/{variant}' for name in models for variant in variants], 'completed_cases': [], 'weights': weights, 'device': device, 'dtype': dtype, 'batch_size': args.batch_size, 'threads': args.threads, 'seed': args.seed, 'mask': args.mask, 'working_size': [512, 512], 'input': 'existing black-level-normalized demosaiced linear uint16 camera RGB', 'preprocessing': 'uint16 INTER_LINEAR resize, gamma 1/2.2 uint8, fixed central neutral checker', 'chart_file': str(Path(args.chart).resolve()), 'chart_sha256': sha256(args.chart), 'chart_size': [180, 135], 'mask_padding': 3, 'gray_patch_index': 19, 'inverse_gamma': 'source uint8 inverse gamma, divide by257, float32 swatch extraction', 'prompt': PROMPT, 'prediction_target': 'neutral', 'preference_evaluation': 'same neutral prediction cross-evaluated against pref_illum', 'steps': 1, 'timestep': 999, 'guidance_scale': 0, 'strength': 0.999, 'noise': 'zeros', 'latent_encode': 'VAE mean', 'laplacian_levels': 2, 'output_latent': 'DDIM pred_original_sample', 'attention': 'PyTorch AttnProcessor2_0, no xformers', 'local_files_only': True, 'versions': versions(), 'source_hashes': {str(path.relative_to(AWB_ROOT)): sha256(path) for path in (UPSTREAM / 'pipelines/pipeline.py', UPSTREAM / 'pipelines/inpaint_pipeline.py', UPSTREAM / 'utils/evaluation_utils.py', UPSTREAM / 'utils/pipeline_utils.py')}, 'save_images': args.save_images}
    write_json(output / 'manifest.json', manifest)
    predictor = None
    try:
        for model in models:
            print(f'正在加载 GCC/{model}，本地推理设备：{device}, dtype={dtype}', flush=True)
            predictor = Predictor(model, args.weights_dir, device, args.dtype)
            for variant in variants:
                destination = output / model / variant
                destination.mkdir(parents=True, exist_ok=True)
                rows = []
                with (destination / 'predictions.jsonl').open('w', encoding='utf-8') as handle:

                    def record(row):
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        handle.flush()
                        detail = f"neutral={row['scores']['neutral']['recovery_deg']:.3f}deg" if row['status'] == 'ok' else row['error']
                        print(f"[{model}/{variant} {len(rows)}/{len(selected)}] {row['image']}: {detail}", flush=True)
                    for start in range(0, len(selected), args.batch_size):
                        prepared = []
                        for sample in selected[start:start + args.batch_size]:
                            began = time.perf_counter()
                            try:
                                image, mask, position, linear, shape = prepare_image(sample, variant, chart, args.mask == 'with')
                                prepared.append((sample, image, mask, position, linear, shape, time.perf_counter() - began))
                            except (OSError, ValueError) as error:
                                record({'image': sample.name, 'model': model, 'variant': variant, 'status': 'failed', 'error': str(error)})
                        if not prepared:
                            continue
                        generated, model_seconds = predictor.predict([row[1] for row in prepared], [row[2] for row in prepared], args.seed)
                        for generated_image, item in zip(generated, prepared):
                            sample, _, _, position, linear, shape, preparation_seconds = item
                            row = {'image': sample.name, 'model': model, 'variant': variant, 'status': 'failed', 'source': str(sample.image_path(variant)), 'original_shape': shape, 'working_shape': [512, 512, 3], 'checker_position': list(position), 'preparation_seconds': preparation_seconds, 'batch_model_seconds': model_seconds, 'batch_images': len(prepared)}
                            try:
                                prediction, source_estimate = estimate_illuminant(generated_image, position)
                                metadata = sample.metadata()
                                scores = score_prediction(prediction, source_estimate, metadata)
                                row.update(status='ok', prediction=prediction.tolist(), source_patch_estimate=source_estimate.tolist(), scores=scores)
                                if args.save_images:
                                    try:
                                        save_images(destination / 'images', sample.name, generated_image, linear, prediction, metadata)
                                    except (OSError, ValueError, KeyError) as error:
                                        row['preview_error'] = str(error)
                            except (OSError, ValueError, KeyError) as error:
                                row['error'] = str(error)
                            record(row)
                metrics = evaluate_case(rows, model, variant)
                write_json(destination / 'evaluation.json', metrics)
                print_case_metrics(metrics)
                manifest['completed_cases'].append(f'{model}/{variant}')
                write_json(output / 'manifest.json', manifest)
            predictor.close()
            predictor = None
        manifest['status'] = 'complete'
        write_json(output / 'manifest.json', manifest)
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(error) or type(error).__name__
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        if predictor is not None:
            predictor.close()
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
