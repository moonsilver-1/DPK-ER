from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from data_loader import SEED


DEFAULT_DP_SEEDS = [3407, 42, 2026]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def clip_by_l2_norm(matrix: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    scale = np.where(norms > max_norm, max_norm / safe_norms, 1.0)
    return matrix * scale


def gaussian_sigma(epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon


def add_gaussian_noise(
    matrix: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    seed: int = SEED,
    normalize: bool = False,
) -> np.ndarray:
    sigma = gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=matrix.shape)
    noised = matrix + noise
    return l2_normalize(noised) if normalize else noised


def private_embeddings(
    matrix: np.ndarray,
    epsilon: float,
    clip_norm: float = 1.0,
    delta: float = 1e-5,
    seed: int = SEED,
    normalize: bool = True,
) -> np.ndarray:
    clipped = clip_by_l2_norm(matrix, max_norm=clip_norm)
    return add_gaussian_noise(
        clipped,
        epsilon=epsilon,
        delta=delta,
        sensitivity=clip_norm,
        seed=seed,
        normalize=normalize,
    )


def private_embeddings_by_seed(
    matrix: np.ndarray,
    epsilon: float,
    seeds: Iterable[int] = DEFAULT_DP_SEEDS,
    clip_norm: float = 1.0,
    delta: float = 1e-5,
    normalize: bool = True,
) -> dict[int, np.ndarray]:
    return {
        int(seed): private_embeddings(
            matrix,
            epsilon=epsilon,
            clip_norm=clip_norm,
            delta=delta,
            seed=int(seed),
            normalize=normalize,
        )
        for seed in seeds
    }

