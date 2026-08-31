"""Benchmark 4: Interpretability on data with a known salient-feature mask.

The first s features carry the class signal; the rest are pure noise.

Quantitative proxies reported per method:
  * Importance fidelity -- Spearman rho, top-s precision/recall between the
    method's inferred feature importance and the ground-truth saliency.
  * Global linearity -- mean R^2 of a linear regression of the class logits on
    the raw features (1.0 = decision function is a single inspectable linear map).
  * Model size -- parameters / storage.
"""
from __future__ import annotations

import numpy as np

from hdc_empirical.data import salient_features
from hdc_empirical.metrics import (
    global_linearity,
    occlusion_importance,
    spearman_rho,
    top_s_precision,
    top_s_recall,
)
from hdc_empirical.models import HDC, RFLinear, RFLinearBin, LinearHead, MLPHead

D = 128
K = 4
S = 16                       # number of truly salient features
N_TRAIN = 400                # per class
N_TEST = 100                 # per class, used for attribution
DELTA = 0.5                  # occlusion jitter


def _importance(model, X):
    if model.name == "hdc":
        imp = occlusion_importance(model, X, delta=DELTA)
    elif model.name in ("linear", "rf-linear", "rf-linear-bin"):
        imp = model.importance_raw()
    else:
        imp = model.importance_raw(X)
    return imp / (np.abs(imp).sum() + 1e-12)


def run() -> dict:
    X, y, saliency = salient_features(
        n_per_class=N_TRAIN + N_TEST, k=K, d=D, s=S,
        center_norm=2.0, sigma=0.5, seed=0,
    )
    y_ = y.reshape(K, -1)
    pool = [X[y == c] for c in range(K)]
    Xtr = np.vstack([pool[c][:N_TRAIN] for c in range(K)])
    ytr = np.repeat(np.arange(K), N_TRAIN)
    Xte = np.vstack([pool[c][N_TRAIN:N_TRAIN + N_TEST] for c in range(K)])
    yte = np.repeat(np.arange(K), N_TEST)

    models = [
        HDC(d=D, D=10_000, seed=0),
        RFLinear(d=D, D=10_000, seed=0),
        RFLinearBin(d=D, D=10_000, seed=0),
        LinearHead(d=D, seed=0),
        MLPHead(d=D, K=K, seed=0),
    ]

    out = {}
    for m in models:
        m.fit(Xtr, ytr)
        imp = _importance(m, Xte)
        lin = global_linearity(m, Xte)
        out[m.name] = {
            "spearman_rho": round(spearman_rho(imp, saliency), 4),
            "top_s_precision": round(top_s_precision(imp, saliency, S), 4),
            "top_s_recall": round(top_s_recall(imp, saliency, S), 4),
            "global_linearity": round(lin, 4),
            "params": int(m.n_params()),
            "storage_kb": round(m.storage_bytes() / 1024.0, 1),
        }
    return {
        "task": "salient_features_d=128_K=4_s=16",
        "n_salient": S,
        "per_method": out,
    }
