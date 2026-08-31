"""Synthetic dataset generators used by the benchmarks.

All generators return float32 numpy arrays and are deterministic given a seed.
"""
from __future__ import annotations

import numpy as np


def gaussian_blobs(
    n_per_class: int,
    k: int,
    d: int,
    center_norm: float = 2.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Well-separated Gaussian clusters.

    Class centers are random unit vectors scaled to ``center_norm``; features
    are drawn as ``center[class] + N(0, sigma^2 I)``. ``sigma`` controls task
    difficulty. Returns (X, y, centers).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, d))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * center_norm
    y = np.repeat(np.arange(k), n_per_class)
    X = centers[y] + rng.normal(0.0, sigma, size=(len(y), d))
    return X.astype(np.float32), y.astype(np.int64), centers


def latent_task(
    n_train: int,
    n_test: int,
    d: int = 128,
    r: int = 4,
    signal: float = 2.0,
    sigma: float = 0.5,
    seed: int = 0,
) -> dict:
    """Latent-manifold task used for robustness experiments.

    The class is determined by a low-dim latent ``z ~ signal * e_class + N(0,I)``
    in R^r which is mapped into R^d by a fixed random linear map ``A``
    (orthonormal columns). Observed features ``x = A z + N(0, sigma^2 I)``.
    Returns dict with train/test and the mixing matrix A.
    """
    rng = np.random.default_rng(seed)
    A, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(d, r)))
    U = np.eye(r)

    def gen(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, r, size=n)
        z = U[y] * signal + rng.normal(0.0, 1.0, size=(n, r))
        x = z @ A.T + rng.normal(0.0, sigma, size=(n, d))
        return x.astype(np.float32), y.astype(np.int64)

    Xtr, ytr = gen(n_train)
    Xte, yte = gen(n_test)
    return {"train": (Xtr, ytr), "test": (Xte, yte), "A": A}


def salient_features(
    n_per_class: int,
    k: int,
    d: int,
    s: int,
    center_norm: float = 2.0,
    sigma: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data with a known, sparse set of ground-truth salient features.

    The first ``s`` features carry the class signal; the remaining ``d - s``
    features are pure noise. Returns (X, y, saliency) where ``saliency`` is a
    binary vector marking the informative coordinates (for interpretability
    benchmarking).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(k, s))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * center_norm
    y = np.repeat(np.arange(k), n_per_class)
    X = np.zeros((len(y), d), dtype=np.float32)
    X[:, :s] = centers[y] + rng.normal(0.0, sigma, size=(len(y), s))
    X[:, s:] = rng.normal(0.0, 1.0, size=(len(y), d - s))
    saliency = np.zeros(d)
    saliency[:s] = 1.0
    return X, y.astype(np.int64), saliency
