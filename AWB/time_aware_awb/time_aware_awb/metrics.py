import numpy as np
import torch

def angular_error(prediction, reference) -> float:
    predicted, gt = (np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    if predicted.shape != (3,) or gt.shape != (3,) or (not np.isfinite([predicted, gt]).all()):
        raise ValueError('Angular error requires finite RGB illuminant vectors')
    norms = np.linalg.norm(predicted) * np.linalg.norm(gt)
    if norms <= 0:
        raise ValueError('Zero illuminant vector')
    return float(np.degrees(np.arccos(np.clip(np.dot(predicted, gt) / norms, -1, 1))))

def angular_error_upstream(prediction, reference) -> float:
    predicted, gt = (torch.tensor(prediction, dtype=torch.float32), torch.tensor(reference, dtype=torch.float32))
    cosine = torch.clamp((predicted * gt).sum() / (predicted.norm() * gt.norm() + 1e-08), -1, 1)
    return float(torch.acos(cosine) * (180.0 / (22.0 / 7)))

def error_summary(errors) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ('mean_deg', 'median_deg', 'best25_deg', 'worst25_deg', 'worst5_deg', 'trimean_deg', 'max_deg')}
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Angular error summary requires finite 1-D samples')
    q25, median, q75, q95 = np.percentile(values, [25, 50, 75, 95])
    return {'mean_deg': float(values.mean()), 'median_deg': float(median), 'best25_deg': float(values[values <= q25].mean()), 'worst25_deg': float(values[values >= q75].mean()), 'worst5_deg': float(values[values >= q95].mean()), 'trimean_deg': float((q25 + 2 * median + q75) / 4), 'max_deg': float(values.max())}
