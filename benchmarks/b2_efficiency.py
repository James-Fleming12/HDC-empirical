"""Benchmark 2: Efficiency -- inference throughput, online-update throughput,
online learning curves, cold-start (few-shot) training time, and stored model size.

All timings are wall-clock on this machine (single-threaded numpy/torch-CPU).

A single fixed pool of Gaussian blobs (seed=0) is used so that train and test
share the same class centers.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from hdc_empirical.data import gaussian_blobs
from hdc_empirical.metrics import bench_time, accuracy
from hdc_empirical.models import (
    HDC,
    RFLinearBin,
    RFLinear,
    LinearHead,
    MLPHead,
    SVMHead,
    KNNHead,
)

D = 128
K = 6
D_HDC = 10_000
N_INFER = 10_000
N_STREAM = 3200
BATCH = 64
PER_CLASS = 2000  # train and test pools

_train = None
_test = None


def _data():
    """Shared fixed train/test pools (same class centers)."""
    global _train, _test
    if _train is None:
        X, y, _ = gaussian_blobs(n_per_class=2 * PER_CLASS, k=K, d=D,
                                 center_norm=2.0, sigma=1.0, seed=0)
        y_ = y.reshape(K, -1)
        Xtr = np.vstack([X[y == c][:PER_CLASS] for c in range(K)])
        Xte = np.vstack([X[y == c][PER_CLASS:] for c in range(K)])
        _train = (Xtr, np.repeat(np.arange(K), PER_CLASS))
        _test = (Xte, np.repeat(np.arange(K), PER_CLASS))
    return _train, _test


def _inference_throughput(X: np.ndarray) -> dict:
    models = [
        HDC(d=D, D=D_HDC, seed=0),
        RFLinear(d=D, D=D_HDC, seed=0),
        RFLinearBin(d=D, D=D_HDC, seed=0),
        LinearHead(d=D, seed=0),
        MLPHead(d=D, K=K, seed=0),
    ]
    Xtr, ytr = _data()[0]
    for m in models:
        m.fit(Xtr, ytr)

    out = {}
    for m in models:
        ms = bench_time(lambda m=m: m.predict(X), repeat=5)
        out[m.name] = {
            "ms_per_10k": round(ms, 3),
            "pred_per_sec": round(10_000 / (ms / 1000.0), 1),
        }
    return out


def _update_throughput() -> dict:
    X, y = _data()[0]
    idx = np.random.default_rng(3).permutation(len(X))[:N_STREAM]
    Xs, ys = X[idx], y[idx]

    models = {
        "hdc": HDC(d=D, D=D_HDC, seed=0),
        "rf-linear": RFLinear(d=D, D=D_HDC, seed=0),
        "rf-linear-bin": RFLinearBin(d=D, D=D_HDC, seed=0),
        "linear-sgd": SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0),
        "mlp": MLPHead(d=D, K=K, seed=0),
    }

    out = {}
    for name, m in models.items():
        classes = np.arange(K)
        batches = [i for i in range(0, N_STREAM, BATCH)]

        def step(i: int) -> None:
            xb, yb = Xs[i:i + BATCH], ys[i:i + BATCH]
            if name == "linear-sgd":
                m.partial_fit(xb, yb, classes=classes)
            else:
                m.partial_fit(xb, yb)

        def run_stream() -> None:
            for i in batches:
                step(i)

        ms = bench_time(run_stream, repeat=2)
        n_updates = len(batches)
        out[name] = {
            "ms_per_batch(64)": round(ms / n_updates, 3),
            "samples_per_sec": round(N_STREAM / (ms / 1000.0), 1),
        }
    return out


def _online_learning_curve() -> dict:
    """Accuracy after a single pass over an online stream.

    All methods see the exact same sample order exactly once. HDC updates
    prototypes directly; the SGD heads take one optimizer step per batch.
    """
    X, y = _data()[0]
    idx = np.random.default_rng(3).permutation(len(X))[:N_STREAM]
    Xs, ys = X[idx], y[idx]
    Xte, yte = _data()[1]

    models = {
        "hdc": HDC(d=D, D=D_HDC, seed=0),
        "rf-linear": RFLinear(d=D, D=D_HDC, seed=0),
        "rf-linear-bin": RFLinearBin(d=D, D=D_HDC, seed=0),
        "linear-sgd": SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0),
        "mlp": MLPHead(d=D, K=K, seed=0),
    }
    checkpoints = {64, 320, 1600, 3200}
    out = {}
    for name, m in models.items():
        row: dict[str, float] = {}
        for i in range(0, N_STREAM, BATCH):
            xb, yb = Xs[i:i + BATCH], ys[i:i + BATCH]
            if name == "linear-sgd":
                m.partial_fit(xb, yb, classes=np.arange(K))
            else:
                m.partial_fit(xb, yb)
            if i + BATCH in checkpoints:
                row[f"seen_{i + BATCH}"] = round(accuracy(yte, m.predict(Xte)), 4)
        out[name] = row
    return out


def _cold_start() -> dict:
    Xtr, ytr = _data()[0]
    per = 50 // K
    idx = np.concatenate([np.where(ytr == c)[0][:per] for c in range(K)])
    Xsmall, ysmall = Xtr[idx], ytr[idx]
    Xte, yte = _data()[1]

    models = [
        HDC(d=D, D=D_HDC, seed=0),
        RFLinearBin(d=D, D=D_HDC, seed=0),
        LinearHead(d=D, seed=0),
        MLPHead(d=D, K=K, seed=0, epochs=40),
    ]
    out = {}
    for m in models:
        ms = bench_time(lambda m=m: m.fit(Xsmall, ysmall), repeat=2)
        acc = accuracy(yte, m.predict(Xte))
        out[m.name] = {"fit_ms(50_samples)": round(ms, 2), "acc_after_cold_start": round(acc, 4)}
    return out


def _model_sizes() -> dict:
    Xtr, ytr = _data()[0]
    models = [
        HDC(d=D, D=D_HDC, seed=0),
        RFLinear(d=D, D=D_HDC, seed=0),
        RFLinearBin(d=D, D=D_HDC, seed=0),
        LinearHead(d=D, seed=0),
        MLPHead(d=D, K=K, seed=0),
        SVMHead(d=D, seed=0),
        KNNHead(d=D),
    ]
    for m in models:
        m.fit(Xtr, ytr)
    out = {}
    for m in models:
        out[m.name] = {
            "params": int(m.n_params()),
            "storage_bytes": int(m.storage_bytes()),
            "storage_kb": round(m.storage_bytes() / 1024.0, 1),
        }
    return out


def run() -> dict:
    Xtr, _ = _data()[0]
    return {
        "input_dim": D, "n_classes": K, "D_hdc": D_HDC,
        "inference": _inference_throughput(Xtr[:N_INFER]),
        "update": _update_throughput(),
        "online_curve": _online_learning_curve(),
        "cold_start": _cold_start(),
        "model_size": _model_sizes(),
    }
