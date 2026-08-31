"""Benchmark 1: Accuracy -- learning curves on Gaussian blobs.

Task: K=6 classes, d=128 features, controlled separation (sigma=1.0).
Train-set size per class is swept; test accuracy is measured on a fixed held-out
set. A large pool is generated once and nested prefixes are used so that larger
training sets contain the smaller ones.
"""
from __future__ import annotations

import numpy as np

from hdc_empirical.data import gaussian_blobs
from hdc_empirical.metrics import accuracy
from benchmarks.common import make_methods

K = 6
D = 128
SIGMA = 1.0
SIZES_PER_CLASS = [10, 50, 200, 1000]
TEST_PER_CLASS = 300


def run() -> dict:
    rng = np.random.default_rng(0)
    X, y, _ = gaussian_blobs(
        n_per_class=max(SIZES_PER_CLASS) + TEST_PER_CLASS,
        k=K, d=D, center_norm=2.0, sigma=SIGMA, seed=0,
    )
    y_ = y.reshape(K, -1)
    pool = [X[y == c] for c in range(K)]

    # fixed test set: last TEST_PER_CLASS per class
    Xte = np.vstack([pool[c][-TEST_PER_CLASS:] for c in range(K)])
    yte = np.repeat(np.arange(K), TEST_PER_CLASS)

    results: dict[str, dict[int, float]] = {}
    for model in make_methods(d=D, k=K):
        row: dict[int, float] = {}
        for s in SIZES_PER_CLASS:
            Xtr = np.vstack([pool[c][:s] for c in range(K)])
            ytr = np.repeat(np.arange(K), s)
            model.fit(Xtr, ytr)
            row[int(s)] = round(accuracy(yte, model.predict(Xte)), 4)
        results[model.name] = row
    return {"task": "gaussian_blobs_d=128_K=6_sigma=1.0", "test_per_class": TEST_PER_CLASS,
            "sizes_per_class": SIZES_PER_CLASS, "test_accuracy": results}
