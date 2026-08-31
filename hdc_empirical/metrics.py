"""Evaluation metrics and timing helpers."""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def top_s_precision(importance: np.ndarray, saliency: np.ndarray, s: int) -> float:
    top = np.argsort(importance)[::-1][:s]
    return float(saliency[top].mean())


def top_s_recall(importance: np.ndarray, saliency: np.ndarray, s: int) -> float:
    top = np.argsort(importance)[::-1][:s]
    denom = max(int(saliency.sum()), 1)
    return float(saliency[top].sum() / denom)


def global_linearity(model, X: np.ndarray) -> float:
    """Mean R^2 of a linear regression that predicts each class logit from the
    raw input features. 1.0 means the decision function is globally linear in
    the inputs (easy to inspect); low values mean it is highly nonlinear."""
    L = model.logits(X)
    rs = []
    for c in range(L.shape[1]):
        rs.append(LinearRegression().fit(X, L[:, c]).score(X, L[:, c]))
    return float(np.mean(rs))


def bench_time(fn, repeat: int = 5) -> float:
    """Median wall-clock time of ``fn`` over ``repeat`` runs, in milliseconds."""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def occlusion_importance(model, X: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """Perturbation/occlusion importance for arbitrary models.

    For each input feature j, measure the mean absolute change in the summed
    class evidence when x_j is jittered by ``delta``.
    """
    base = model.logits(X).sum(axis=1)
    imp = np.zeros(model.d)
    for j in range(model.d):
        Xp = X.copy()
        Xp[:, j] += delta
        d = model.logits(Xp).sum(axis=1) - base
        imp[j] = np.abs(d).mean()
    return imp
