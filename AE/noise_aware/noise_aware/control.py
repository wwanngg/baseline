from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from .dataset import Capture, list_captures, nearest_capture
from .evaluation import copy_stereo_pair, load_workspace, metric_grid
from .io_utils import output_directory, read_image, write_json
from .metric import grayscale

@dataclass(frozen=True)
class ControlConfig:
    seed: int = 0
    max_evaluations: int = 50
    tolerance: float = 1e-05
    simplex_scale: float = 1.7
    gain_resolution: float = 0.1
    exposure_resolution_us: float = 10.0

    def __post_init__(self) -> None:
        if self.max_evaluations < 3:
            raise ValueError('max_evaluations must be at least 3')
        values = (self.tolerance, self.simplex_scale, self.gain_resolution, self.exposure_resolution_us)
        if not all((math.isfinite(v) and v > 0 for v in values)):
            raise ValueError('Control tolerances, scale and resolutions must be finite and positive')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')

class MetricSurface:

    def __init__(self, gains: np.ndarray, exposures: np.ndarray, scores: np.ndarray, config: ControlConfig | None=None):
        self.config = config or ControlConfig()
        self.gains, self.exposures, self.grid = metric_grid(gains, exposures, scores)
        if min(len(self.gains), len(self.exposures)) < 2 or not np.isfinite(self.grid).all():
            raise ValueError('Control requires a complete rectangular grid with at least two settings per axis')
        self.spline = RectBivariateSpline(self.exposures, self.gains, self.grid, kx=min(3, len(self.exposures) - 1), ky=min(3, len(self.gains) - 1), s=0)
        self.history: list[tuple[float, float, float, float, float]] = []

    def objective(self, point: np.ndarray) -> float:
        gain, exposure = map(float, point)
        gain_q = self.gains[0] + np.floor((gain - self.gains[0]) / self.config.gain_resolution + 0.5) * self.config.gain_resolution
        exposure_q = self.exposures[0] + np.floor((exposure - self.exposures[0]) / self.config.exposure_resolution_us + 0.5) * self.config.exposure_resolution_us
        value = 10000000000.0
        if np.isfinite(point).all() and self.gains[0] <= gain <= self.gains[-1] and (self.exposures[0] <= exposure <= self.exposures[-1]) and (gain_q <= self.gains[-1] + 1e-08) and (exposure_q <= self.exposures[-1] + 1e-08):
            value = -float(self.spline.ev(exposure_q, gain_q))
        self.history.append((gain, exposure, float(gain_q), float(exposure_q), value))
        return value

def initial_simplex(captures: list[Capture], config: ControlConfig, initial: tuple[float, float] | None=None) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    capture = captures[int(rng.integers(len(captures)))] if initial is None else nearest_capture(captures, initial)
    start = np.array([capture.gain_db, capture.exposure_us], dtype=float)
    intensity = float(grayscale(read_image(capture.path)).mean())
    h = -intensity / (255 * config.simplex_scale) if intensity >= 128 else config.simplex_scale * (1 - intensity / 255)
    vertices = [start]
    for axis in range(2):
        target = start.copy()
        target[axis] *= 1 + h
        snapped = nearest_capture(captures, target)
        vertex = np.array([snapped.gain_db, snapped.exposure_us])
        if vertex[axis] == start[axis]:
            options = np.unique([c.gain_db if axis == 0 else c.exposure_us for c in captures])
            alternatives = options[options != start[axis]]
            if not len(alternatives):
                raise ValueError('Cannot initialize control with a single setting per axis')
            vertex = start.copy()
            vertex[axis] = alternatives[np.argmin(np.abs(alternatives - start[axis]))]
        vertices.append(vertex)
    return np.asarray(vertices, dtype=float)

def control_scene(scene: Path, output: Path, config: ControlConfig | None=None, initial: tuple[float, float] | None=None, plots: bool=True) -> dict:
    config = config or ControlConfig()
    if initial is not None and (not np.isfinite(initial).all()):
        raise ValueError('Initial gain and exposure must be finite')
    output = output_directory(output, scene.parent)
    workspace = load_workspace(output / 'Result_graph' / scene.name / 'workspace_result.npz')
    if not workspace['metadata']['complete']:
        raise ValueError('Workspace was produced with only part of the scene. Re-run evaluate without --max-images.')
    captures = list_captures(scene, workspace['metadata']['camera'])
    if set(workspace['filenames'].tolist()) != {c.path.name for c in captures}:
        raise ValueError('Workspace image list does not match the current scene. Re-run evaluate.')
    surface = MetricSurface(workspace['gains'], workspace['exposures'], workspace['metrics'][:, 0], config)
    simplex = initial_simplex(captures, config, initial)
    result = minimize(surface.objective, simplex[0], method='Nelder-Mead', options={'initial_simplex': simplex, 'maxfev': config.max_evaluations, 'fatol': config.tolerance, 'xatol': float('inf'), 'adaptive': False})
    selected = nearest_capture(captures, result.x)
    destination = output / 'Result_NM' / scene.name
    destination.mkdir(parents=True, exist_ok=True)
    report = {'scene': scene.name, 'seed': config.seed, 'selected_gain_db': selected.gain_db, 'selected_exposure_us': selected.exposure_us, 'selected_filename': selected.path.name}
    write_json(destination / 'control_result.json', report)
    selected_dir = output / 'Result_Exp' / scene.name
    manifest_path = selected_dir / 'selected_images.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    manifest['NM'] = copy_stereo_pair(scene, selected.path.name, selected_dir, 'NM', selected.gain_db, selected.exposure_us)
    write_json(manifest_path, manifest)
    return report
