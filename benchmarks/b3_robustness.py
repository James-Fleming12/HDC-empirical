"""Benchmark 3: Robustness on the latent-manifold task.

Four stress conditions, all evaluated against a model trained on clean data:
  1. Gaussian input noise        -- additive N(0, a^2 I) on test inputs
  2. Salt-and-pepper corruption  -- a fraction of coordinates flipped to +/-6
  3. Domain shift (covariate)    -- correlated nuisance noise along one direction
  4. Label noise during training -- a fraction of train labels randomly flipped
"""
from __future__ import annotations

import numpy as np

from hdc_empirical.data import latent_task
from hdc_empirical.metrics import accuracy
from hdc_empirical.models import (
    HDC,
    KNNHead,
    LinearHead,
    MLPHead,
    RFLinear,
    RFLinearBin,
    SVMHead,
)

D = 128
K = 4
N_TRAIN = 2000
N_TEST = 1000
NOISE_LEVELS = [0.0, 1.0, 2.0, 4.0, 8.0]
SALT_FRACS = [0.05, 0.1, 0.2, 0.4]
SHIFT_TAUS = [0.0, 1.0, 2.0, 4.0]
LABEL_FLIPS = [0.0, 0.2, 0.4]


def _models():
    return [
        HDC(d=D, D=10_000, seed=0),
        RFLinear(d=D, D=10_000, seed=0),
        RFLinearBin(d=D, D=10_000, seed=0),
        LinearHead(d=D, seed=0),
        MLPHead(d=D, K=K, seed=0),
        SVMHead(d=D, seed=0),
        KNNHead(d=D),
    ]


def _evaluate(models, Xte, yte) -> dict:
    return {m.name: round(accuracy(yte, m.predict(Xte)), 4) for m in models}


def _gaussian_noise(models, Xte, yte, rng) -> dict:
    out = {}
    for a in NOISE_LEVELS:
        Xn = Xte + rng.normal(0.0, a, size=Xte.shape).astype(np.float32)
        out[f"a={a}"] = _evaluate(models, Xn, yte)
    return out


def _salt_pepper(models, Xte, yte, rng) -> dict:
    out = {}
    for f in SALT_FRACS:
        Xn = Xte.copy()
        mask = rng.random(Xte.shape) < f
        val = np.where(Xn[mask] >= 0, 6.0, -6.0)
        Xn[mask] = val
        out[f"frac={f}"] = _evaluate(models, Xn, yte)
    return out


def _domain_shift(models, Xte, yte, rng) -> dict:
    """Correlated nuisance noise: x += u * v with u ~ N(0, tau^2) along a
    random unit direction v. Larger tau = stronger covariate shift."""
    v = rng.normal(0.0, 1.0, size=(D,))
    v = v / np.linalg.norm(v)
    out = {}
    for tau in SHIFT_TAUS:
        u = rng.normal(0.0, tau, size=(len(Xte),)).astype(np.float32)
        Xn = Xte + np.outer(u, v).astype(np.float32)
        out[f"tau={tau}"] = _evaluate(models, Xn, yte)
    return out


def _label_noise(models, Xtr, ytr, Xte, yte, rng) -> dict:
    out = {}
    for p in LABEL_FLIPS:
        yl = ytr.copy()
        flips = rng.random(len(yl)) < p
        new = rng.integers(0, K, size=len(yl))
        yl[flips] = new[flips]
        accs = {}
        for m in models:
            m.fit(Xtr, yl)
            accs[m.name] = round(accuracy(yte, m.predict(Xte)), 4)
        out[f"flip={p}"] = accs
    return out


def run() -> dict:
    rng = np.random.default_rng(0)
    data = latent_task(N_TRAIN, N_TEST, d=D, seed=0)
    Xtr, ytr = data["train"]
    Xte, yte = data["test"]

    models = _models()
    for m in models:
        m.fit(Xtr, ytr)

    clean = _evaluate(models, Xte, yte)
    return {
        "task": "latent_manifold_d=128_K=4",
        "clean_test_acc": clean,
        "gaussian_noise": _gaussian_noise(models, Xte, yte, rng),
        "salt_pepper": _salt_pepper(models, Xte, yte, rng),
        "domain_shift": _domain_shift(models, Xte, yte, rng),
        "label_noise": _label_noise(models, Xtr, ytr, Xte, yte, rng),
    }
