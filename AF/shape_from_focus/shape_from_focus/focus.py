from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve
from .upstream import ML, SML

@dataclass(frozen=True)
class FocusConfig:
    mode: str = 'paper'
    window: int = 5
    threshold: float = 7.0
    step: int = 1
    interpolation: str = 'gaussian'

    def __post_init__(self):
        if self.mode not in ('paper', 'upstream') or self.interpolation not in ('none', 'gaussian'):
            raise ValueError('Invalid focus mode/interpolation')
        if self.step < 1 or not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('step must be positive and threshold finite/nonnegative')
        if self.mode == 'paper' and (self.window < 1 or self.window % 2 != 1):
            raise ValueError('Paper SML requires a positive odd window size')
        if self.mode == 'upstream' and (self.window, self.threshold, self.step, self.interpolation) != (2, 0, 1, 'none'):
            raise ValueError('Upstream preset uses a fixed 2x2 window, no threshold, step=1 and no interpolation')

    @classmethod
    def from_options(cls, mode='paper', window=None, threshold=None, step=None, interpolation=None):
        if mode == 'upstream':
            if any((v is not None for v in (window, threshold, step))) or interpolation not in (None, 'none'):
                raise ValueError('--mode upstream has fixed parameters; omit window/threshold/step/Gaussian overrides')
            return cls(mode='upstream', window=2, threshold=0, step=1, interpolation='none')
        return cls(mode=mode, window=5 if window is None else window, threshold=7.0 if threshold is None else threshold, step=1 if step is None else step, interpolation='gaussian' if interpolation is None else interpolation)

def spatial_mask(shape: tuple[int, int], config: FocusConfig) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    if config.mode == 'upstream':
        if min(shape) < 8:
            raise ValueError('Upstream preset needs image dimensions >= 8')
        mask[4:rows - 3, 4:cols - 3] = True
    else:
        margin = config.step + config.window // 2
        if min(shape) <= 2 * margin:
            raise ValueError(f'Image dimensions must exceed {2 * margin} for this SML step/window')
        mask[margin:rows - margin, margin:cols - margin] = True
    return mask

def modified_laplacian(image: np.ndarray, step: int=1) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or step < 1 or min(image.shape) <= 2 * step:
        raise ValueError('Modified Laplacian requires a sufficiently large 2-D grayscale image')
    if not np.isfinite(image).all():
        raise ValueError('Image contains NaN/Inf')
    centre = image[step:-step, step:-step]
    ml = np.zeros_like(image)
    ml[step:-step, step:-step] = np.abs(2 * centre - image[:-2 * step, step:-step] - image[2 * step:, step:-step]) + np.abs(2 * centre - image[step:-step, :-2 * step] - image[step:-step, 2 * step:])
    return ml

def focus_measure(image: np.ndarray, config: FocusConfig) -> np.ndarray:
    mask = spatial_mask(image.shape, config)
    ml = modified_laplacian(image, config.step)
    if config.mode == 'upstream':
        result = SML(ML(np.asarray(image, dtype=np.float64)))
    else:
        selected = np.where(ml >= config.threshold, ml, 0)
        result = convolve(selected, np.ones((config.window, config.window)), mode='constant', cval=0)
        result[~mask] = 0
    return result

def gaussian_peak(index: np.ndarray, centre: np.ndarray, left: np.ndarray, right: np.ndarray, frames: int, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    refined = index.astype(np.float64)
    candidates = valid & (index > 0) & (index < frames - 1) & (left > 0) & (right > 0) & (centre > left) & (centre > right)
    use = np.zeros(index.shape, dtype=bool)
    if np.any(candidates):
        log_left = np.log(left[candidates]) - np.log(centre[candidates])
        log_right = np.log(right[candidates]) - np.log(centre[candidates])
        denominator = log_left + log_right
        good = np.isfinite(denominator) & (denominator < -1e-12)
        offset = np.zeros_like(denominator)
        np.divide(0.5 * (log_left - log_right), denominator, out=offset, where=good)
        good &= np.isfinite(offset) & (np.abs(offset) <= 0.5 + 1e-12)
        coordinates = np.where(candidates)
        use[coordinates] = good
        corrected = index[candidates].astype(float) + offset
        refined[coordinates] = np.where(good, corrected, index[candidates])
    refined[~valid] = np.nan
    return (refined, use)

def reconstruct(stack: np.ndarray, config: FocusConfig | None=None, progress=None) -> dict:
    config = config or FocusConfig()
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[2] < 1 or (not np.issubdtype(stack.dtype, np.number)) or (not np.isrealobj(stack)):
        raise ValueError('Expected a real numeric grayscale stack with shape (height, width, frames)')
    rows, cols, frames = stack.shape
    mask = spatial_mask((rows, cols), config)
    peak = np.full((rows, cols), -np.inf)
    index = np.zeros((rows, cols), dtype=np.int32)
    frame_scores = []
    for frame in range(frames):
        score = focus_measure(stack[:, :, frame], config)
        improved = score > peak
        peak[improved], index[improved] = (score[improved], frame)
        values = score[mask]
        frame_scores.append({'frame_index': frame, 'mean_sml': float(values.mean()), 'sum_sml': float(values.sum()), 'positive_pixels': int(np.count_nonzero(values > 0))})
        if progress:
            progress('SML', frame + 1, frames)
    valid = mask & (peak > 0)
    interpolated = np.zeros((rows, cols), dtype=bool)
    refined = index.astype(float)
    if config.interpolation == 'gaussian' and frames >= 3:
        left, right = (np.zeros_like(peak), np.zeros_like(peak))
        interior = valid & (index > 0) & (index < frames - 1)
        for frame in range(frames):
            is_left, is_right = (interior & (index == frame + 1), interior & (index == frame - 1))
            if not np.any(is_left | is_right):
                continue
            score = focus_measure(stack[:, :, frame], config)
            left[is_left], right[is_right] = (score[is_left], score[is_right])
            if progress:
                progress('Gaussian neighbours', frame + 1, frames)
        refined, interpolated = gaussian_peak(index, peak, left, right, frames, valid)
    refined[~valid] = np.nan
    all_in_focus = np.take_along_axis(stack, index[:, :, None], axis=2)[:, :, 0].copy()
    all_in_focus[~valid] = 0
    index[~valid] = -1
    return {'best_frame_index': index, 'refined_frame_index': refined, 'focus_peak': peak, 'spatial_mask': mask, 'valid_mask': valid, 'interpolated_mask': interpolated, 'all_in_focus': all_in_focus, 'frame_scores': frame_scores}
