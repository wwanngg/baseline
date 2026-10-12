import numpy as np
METRIC_NAMES = ('mean_deg', 'median_deg', 'best25_deg', 'worst25_deg', 'worst5_deg', 'trimean_deg', 'max_deg')

def angular_error(prediction, reference):
    predicted, gt = (np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    if predicted.shape != (3,) or gt.shape != (3,) or (not np.isfinite([predicted, gt]).all()):
        raise ValueError('Angular error requires finite RGB illuminant vectors')
    norms = np.linalg.norm(predicted) * np.linalg.norm(gt)
    if norms <= 0:
        raise ValueError('Zero illuminant vector')
    return float(np.degrees(np.arccos(np.clip(np.dot(predicted, gt) / norms, -1, 1))))

def reproduction_error(prediction, reference):
    prediction, reference = (np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    if prediction.shape != (3,) or reference.shape != (3,) or np.any(prediction <= 0) or np.any(reference <= 0):
        raise ValueError('Reproduction error requires positive RGB vectors')
    return angular_error(reference / prediction, np.ones(3))

def error_summary(errors):
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return dict.fromkeys(METRIC_NAMES)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Error summary requires finite 1-D values')
    q25, median, q75, q95 = np.percentile(values, [25, 50, 75, 95])
    return {'mean_deg': float(values.mean()), 'median_deg': float(median), 'best25_deg': float(values[values <= q25].mean()), 'worst25_deg': float(values[values >= q75].mean()), 'worst5_deg': float(values[values >= q95].mean()), 'trimean_deg': float((q25 + 2 * median + q75) / 4), 'max_deg': float(values.max())}

def release_summary(errors):
    values = np.sort(np.asarray(errors, dtype=np.float64))
    n = len(values)
    if n < 4:
        return {'available': False, 'reason': 'Release quartile slicing needs at least four samples'}
    f50 = values[int(np.floor(0.5 * n)) - 1]
    f25 = values[int(np.floor(0.25 * n)) - 1]
    f75 = values[int(np.floor(0.75 * n)) - 1]
    return {'available': True, 'mean_deg': float(values.mean()), 'median_deg': float(np.median(values)), 'trimean_deg': float(0.25 * (f25 + 2 * f50 + f75)), 'best25_deg': float(values[:int(np.floor(0.25 * n))].mean()), 'worst25_deg': float(values[int(np.floor(0.75 * n) - 1):].mean())}
