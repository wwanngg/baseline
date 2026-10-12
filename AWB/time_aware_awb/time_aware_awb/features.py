import itertools
import cv2
import numpy as np
EPS = 1e-08
TIME_EVENTS = ('sunrise', 'sunset', 'dusk', 'dawn', 'noon', 'midnight')

def capture_features(metadata: dict, config: dict) -> np.ndarray:
    chunks = []
    for key in config['capture_data']:
        if key == 'noise_stats':
            value = metadata[key]['rgb_mean'] + metadata[key]['rgb_std']
        elif key == 'snr_stats':
            value = metadata[key]['mean'] + metadata[key]['std']
        elif key == 'time':
            probabilities = np.array([metadata['capture_metadata'][f'prob_{e}'] for e in TIME_EVENTS])
            if np.any(probabilities < 0):
                raise ValueError('Negative time probabilities')
            flags = [metadata['capture_metadata'][f'is_before_{e}'] for e in TIME_EVENTS]
            value = np.concatenate([np.sqrt(probabilities), flags])
        elif key == 'flash':
            value = [metadata['capture_metadata'][key]]
        elif key in ('iso', 'shutter_speed'):
            number = metadata['capture_metadata'][key]
            if number <= 0:
                raise ValueError(f'{key} must be positive before logarithm')
            value = [np.log(number)]
        else:
            raise ValueError(f'Unsupported capture feature: {key}')
        chunks.append(np.asarray(value).flatten())
    raw = np.concatenate(chunks).astype(np.float32)
    lo = np.array(config['norm_values']['capture-data-min'])
    hi = np.array(config['norm_values']['capture-data-max'])
    if raw.shape != lo.shape or lo.shape != hi.shape or (not np.isfinite([lo, hi]).all()) or np.any(hi < lo):
        raise ValueError('Capture feature size/normalization does not match the pretrained config')
    result = ((raw - lo) / (hi - lo + EPS)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite capture features')
    return result

def rgb_to_rgbg(rgb):
    return np.stack([rgb[..., 0] / (rgb[..., 1] + EPS), rgb[..., 2] / (rgb[..., 1] + EPS)], axis=-1)

def compute_edges(image):
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    edges = np.zeros(image.shape)
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if dy or dx:
            edges += np.abs(image - padded[1 + dy:image.shape[0] + 1 + dy, 1 + dx:image.shape[1] + 1 + dx])
    return edges / 8

def histogram_features(image: np.ndarray, config: dict) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or (not np.isfinite(image).all()):
        raise ValueError('Histogram input must be finite linear RGB')
    u_min, v_min, u_max, v_max = config['hist_boundaries']
    bins = int(config['hist_bins'])
    if bins < 1 or not np.isfinite([u_min, v_min, u_max, v_max]).all() or u_min >= u_max or (v_min >= v_max):
        raise ValueError('Invalid pretrained histogram boundaries/bins')
    pixels = image.reshape(-1, 3)
    brightness = np.linalg.norm(pixels, axis=1)
    chroma = rgb_to_rgbg(pixels)
    ranges = ((u_min, u_max), (v_min, v_max))
    color, _, _ = np.histogram2d(chroma[:, 0], chroma[:, 1], bins=bins, range=ranges, weights=brightness)
    color = np.sqrt(color / (color.sum() + EPS))
    xx, yy = np.meshgrid(np.arange(bins), np.arange(bins), indexing='ij')
    edge_chroma = rgb_to_rgbg(compute_edges(image)).reshape(-1, 2)
    edge, _, _ = np.histogram2d(edge_chroma[:, 0], edge_chroma[:, 1], bins=bins, range=ranges, weights=brightness)
    edge = np.sqrt(edge / (edge.sum() + EPS))
    return np.stack([color, xx / bins, yy / bins, edge], axis=0).astype(np.float32)
