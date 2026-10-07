"""Utilities for storing and consuming per-sample regression uncertainty."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Literal, NotRequired, TypedDict

import numpy as np

from lib.types import PartKey

PARTS: tuple[PartKey, ...] = ('train', 'val', 'test')


class SigmaSourceConfig(TypedDict):
    root: str
    value_kind: Literal['sigma', 'variance']
    target_units: NotRequired[Literal['raw', 'standardized']]
    sigma_scale: NotRequired[float]


@dataclass(frozen=True)
class LoadedSigma:
    values: dict[PartKey, np.ndarray]
    artifact: Path


def array_fingerprint(values: np.ndarray) -> str:
    """Return a stable fingerprint that includes row order, shape and dtype."""
    values = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def central_quantile_sigma(
    quantile_predictions: np.ndarray,
    levels: np.ndarray,
    *,
    central_mass: float = 0.9,
    floor: float = 0.0,
) -> np.ndarray:
    """Estimate a Gaussian-equivalent sigma from a trimmed quantile function."""
    predictions, levels = _validate_quantiles(quantile_predictions, levels)
    if not 0.0 < central_mass < 1.0:
        raise ValueError('central_mass must be strictly between zero and one')
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError('floor must be a finite non-negative number')

    alpha = (1.0 - central_mass) / 2.0
    mask = (levels >= alpha) & (levels <= 1.0 - alpha)
    if mask.sum() < 2:
        raise ValueError('The quantile grid has fewer than two central levels')

    central = np.sort(predictions, axis=-1)[:, mask]
    center = central.mean(axis=-1, keepdims=True)
    conditional_variance = np.mean(np.square(central - center), axis=-1)

    sigma = np.sqrt(
        np.maximum(
            conditional_variance * central_quantile_variance_correction(central_mass),
            floor**2,
        )
    )
    return sigma.astype(np.float32)


def central_quantile_variance_correction(central_mass: float) -> float:
    """Correct a central conditional variance to the variance of a Normal."""
    if not 0.0 < central_mass < 1.0:
        raise ValueError('central_mass must be strictly between zero and one')
    alpha = (1.0 - central_mass) / 2.0
    z = NormalDist().inv_cdf(1.0 - alpha)
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    conditional_variance = 1.0 - 2.0 * z * density / central_mass
    if conditional_variance <= 0.0:
        raise RuntimeError('Invalid Gaussian truncation correction')
    return 1.0 / conditional_variance


def full_quantile_moments(
    quantile_predictions: np.ndarray,
    levels: np.ndarray,
    *,
    n_tail_levels: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a quantile function with exponential extrapolation in both tails."""
    predictions, levels = _validate_quantiles(quantile_predictions, levels)
    values = np.sort(predictions, axis=-1)
    n_tail_levels = min(n_tail_levels, values.shape[-1] // 2)
    if n_tail_levels < 2:
        raise ValueError('At least four quantile levels are required')

    x_left = np.log(levels[:n_tail_levels])
    x_left -= x_left.mean()
    left = values[:, :n_tail_levels]
    beta_left = np.sum((left - left.mean(-1, keepdims=True)) * x_left, -1)
    beta_left /= np.square(x_left).sum()

    x_right = np.log1p(-levels[-n_tail_levels:])
    x_right -= x_right.mean()
    right = values[:, -n_tail_levels:]
    beta_right = -np.sum((right - right.mean(-1, keepdims=True)) * x_right, -1)
    beta_right /= np.square(x_right).sum()

    scale_floor = np.finfo(np.float64).tiny
    beta_left = np.maximum(beta_left, scale_floor)
    beta_right = np.maximum(beta_right, scale_floor)

    delta = np.diff(levels)
    q_left = values[:, :-1]
    q_right = values[:, 1:]
    moment1 = (
        np.sum(delta * (q_left + q_right) / 2.0, axis=-1)
        + levels[0] * (values[:, 0] - beta_left)
        + (1.0 - levels[-1]) * (values[:, -1] + beta_right)
    )
    moment2 = (
        np.sum(
            delta * (np.square(q_left) + q_left * q_right + np.square(q_right)) / 3.0,
            axis=-1,
        )
        + levels[0]
        * (
            np.square(values[:, 0])
            - 2.0 * beta_left * values[:, 0]
            + 2.0 * np.square(beta_left)
        )
        + (1.0 - levels[-1])
        * (
            np.square(values[:, -1])
            + 2.0 * beta_right * values[:, -1]
            + 2.0 * np.square(beta_right)
        )
    )
    variance = np.maximum(moment2 - np.square(moment1), 0.0)
    return moment1.astype(np.float32), variance.astype(np.float32)


def load_sigma_source(
    config: SigmaSourceConfig,
    *,
    dataset_name: str,
    expected_targets: dict[PartKey, np.ndarray],
    label_std: float,
) -> LoadedSigma:
    """Load sigma and convert it to standardized target units."""
    root = Path(config['root'])
    directory_artifact = root / dataset_name / 'predictions.npz'
    flat_artifact = root / f'{dataset_name}.npz'
    if directory_artifact.exists():
        artifact = directory_artifact
        metadata_path = directory_artifact.with_name('metadata.json')
        if not metadata_path.exists():
            raise FileNotFoundError(
                f'Missing metadata for directory sigma artifact: {metadata_path}'
            )
        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(metadata, dataset_name, expected_targets, metadata_path)
    elif flat_artifact.exists():
        artifact = flat_artifact
    else:
        raise FileNotFoundError(
            f'No sigma artifact for {dataset_name!r}; checked {directory_artifact} '
            f'and {flat_artifact}'
        )

    value_kind = config['value_kind']
    if value_kind not in ('sigma', 'variance'):
        raise ValueError(f'Unknown sigma source value_kind: {value_kind!r}')
    target_units = config.get('target_units', 'raw')
    if target_units not in ('raw', 'standardized'):
        raise ValueError(f'Unknown sigma source target_units: {target_units!r}')
    sigma_scale = float(config.get('sigma_scale', 1.0))
    if not math.isfinite(sigma_scale) or sigma_scale <= 0.0:
        raise ValueError('sigma_source.sigma_scale must be finite and positive')
    if not math.isfinite(label_std) or label_std <= 0.0:
        raise ValueError('The training target standard deviation must be positive')

    result: dict[PartKey, np.ndarray] = {}
    with np.load(artifact) as stored:
        for part in PARTS:
            if part not in stored:
                raise ValueError(f'{artifact} does not contain the {part!r} split')
            values = np.asarray(stored[part], dtype=np.float64)
            expected_size = len(expected_targets[part])
            if values.ndim != 1 or len(values) != expected_size:
                raise ValueError(
                    f'{artifact}:{part} must have shape ({expected_size},), '
                    f'got {values.shape}'
                )
            if not np.isfinite(values).all() or not (values > 0.0).all():
                raise ValueError(f'{artifact}:{part} contains invalid uncertainty')
            if value_kind == 'variance':
                values = np.sqrt(values)
            if target_units == 'raw':
                values /= label_std
            result[part] = (values * sigma_scale).astype(np.float32)
    return LoadedSigma(result, artifact)


def _validate_quantiles(
    quantile_predictions: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(quantile_predictions, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if predictions.ndim != 2:
        raise ValueError('quantile_predictions must be a two-dimensional array')
    if levels.ndim != 1 or predictions.shape[1] != len(levels):
        raise ValueError('levels must match the quantile prediction dimension')
    if len(levels) < 2 or not np.all(np.diff(levels) > 0.0):
        raise ValueError('levels must be strictly increasing')
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError('levels must lie strictly between zero and one')
    if not np.isfinite(predictions).all():
        raise ValueError('quantile_predictions contains non-finite values')
    return predictions, levels


def _validate_metadata(
    metadata: dict,
    dataset_name: str,
    expected_targets: dict[PartKey, np.ndarray],
    path: Path,
) -> None:
    if metadata.get('dataset') != dataset_name:
        raise ValueError(
            f'{path} describes dataset {metadata.get("dataset")!r}, '
            f'expected {dataset_name!r}'
        )
    sizes = metadata.get('split_sizes')
    fingerprints = metadata.get('target_fingerprints')
    if not isinstance(sizes, dict) or not isinstance(fingerprints, dict):
        raise ValueError(f'{path} has no split sizes or target fingerprints')
    for part in PARTS:
        target = np.asarray(expected_targets[part])
        if sizes.get(part) != len(target):
            raise ValueError(f'{path} has the wrong {part!r} split size')
        if fingerprints.get(part) != array_fingerprint(target):
            raise ValueError(f'{path} has a mismatched {part!r} row order')
