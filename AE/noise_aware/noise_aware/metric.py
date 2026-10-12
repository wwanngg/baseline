from dataclasses import dataclass
from pathlib import Path
import math
import cv2
import numpy as np
from .io_utils import read_image
from .upstream import PARAMETERS

@dataclass(frozen=True)
class MetricConfig:
    alpha: float = PARAMETERS['Alpha']
    beta: float = PARAMETERS['Beta']
    gradient_scale: float = PARAMETERS['Kg']
    entropy_scale: float = PARAMETERS['Ke']
    gradient_lambda: float = PARAMETERS['Lambda']
    gradient_threshold: float = PARAMETERS['Gamma']
    grid_size: int = 10
    homogeneous_fraction: float = PARAMETERS['p']
    lower_intensity: int = 15
    upper_intensity: int = 235
    resize_factor: float = PARAMETERS['Resize_factor']
    green_weighted_noise: bool = False

    def __post_init__(self) -> None:
        numbers = (self.alpha, self.beta, self.gradient_scale, self.entropy_scale, self.gradient_lambda, self.gradient_threshold, self.homogeneous_fraction, self.resize_factor)
        if not all((math.isfinite(v) for v in numbers)):
            raise ValueError('Metric parameters must be finite.')
        if not 0 <= self.alpha <= 1 or self.beta < 0:
            raise ValueError('Require 0 <= alpha <= 1 and beta >= 0.')
        if self.gradient_scale < 0 or self.entropy_scale < 0:
            raise ValueError('Metric scaling factors must be nonnegative.')
        if self.gradient_lambda <= 0 or not 0 <= self.gradient_threshold < 1:
            raise ValueError('Invalid gradient mapping parameters.')
        if not isinstance(self.grid_size, int) or self.grid_size < 2:
            raise ValueError('grid_size must be an integer >= 2.')
        if not 0 < self.homogeneous_fraction <= 1 or self.resize_factor <= 0:
            raise ValueError('Invalid homogeneous fraction or resize factor.')
        if not 0 <= self.lower_intensity <= self.upper_intensity <= 255:
            raise ValueError('Invalid unsaturated intensity range.')

@dataclass(frozen=True)
class MetricResult:
    score: float
    gradient: float
    entropy: float
    noise: float
    intensity: float

def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    weights = np.array([0.114020904255103, 0.587043074451121, 0.298936021293775])
    return np.floor(image.astype(np.float64) @ weights + 0.5).clip(0, 255).astype(np.uint8)

def _sobel_squared(channel: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return dx * dx + dy * dy

def _noise(channel: np.ndarray, config: MetricConfig) -> float:
    h, w = channel.shape
    gradient = _sobel_squared(channel)
    rank = max(0, min(gradient.size - 1, int(gradient.size * config.homogeneous_fraction + 0.5) - 1))
    threshold = np.partition(gradient.ravel(), rank)[rank]
    mask = (gradient <= threshold) & (channel >= config.lower_intensity) & (channel <= config.upper_intensity)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    laplacian = cv2.filter2D(channel.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    count = int(mask.sum())
    if count < channel.size * 0.0001:
        return float(math.sqrt(math.pi / 2) * np.abs(laplacian).sum() / (6 * (w - 2) * (h - 2)))
    return float(math.sqrt(math.pi / 2) * np.abs(laplacian[mask]).sum() / (6 * count))

def assess_image(image: str | Path | np.ndarray, config: MetricConfig | None=None) -> MetricResult:
    config = config or MetricConfig()
    if isinstance(image, (str, Path)):
        image = read_image(image)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected an 8-bit image (uint8).')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Expected grayscale or three-channel BGR.')
    if config.resize_factor != 1:
        h, w = image.shape[:2]
        size = (int(w * config.resize_factor + 0.5), int(h * config.resize_factor + 0.5))
        if min(size) < 3:
            raise ValueError('Resized image is too small.')
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    if min(h, w) < max(3, config.grid_size):
        raise ValueError('Image dimensions must be at least grid_size and at least 3.')
    channels = [image] if image.ndim == 2 else cv2.split(image)
    noise_values = [_noise(channel, config) for channel in channels]
    if config.green_weighted_noise and len(channels) == 3:
        noise = float(np.dot(noise_values, [0.25, 0.5, 0.25]))
    else:
        noise = float(np.mean(noise_values))
    gray = grayscale(image)
    normalized = np.sqrt(_sobel_squared(gray)) / math.sqrt(32 * 255 ** 2)
    mapped = np.log1p(config.gradient_lambda * np.maximum(normalized - config.gradient_threshold, 0))
    mapped /= math.log1p(config.gradient_lambda * (1 - config.gradient_threshold))
    grid_means = np.array([block.mean() for rows in np.array_split(mapped, config.grid_size, axis=0) for block in np.array_split(rows, config.grid_size, axis=1)])
    mean, std = (float(grid_means.mean()), float(grid_means.std(ddof=1)))
    gradient = mean / std if std > 1e-12 else mean
    histogram = np.bincount(gray.ravel(), minlength=256)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = float(-np.dot(probabilities, np.log2(probabilities)) * config.entropy_scale)
    score = config.alpha * config.gradient_scale * gradient + (1 - config.alpha) * entropy - config.beta * noise
    return MetricResult(float(score), gradient, entropy, noise, float(gray.mean()))
