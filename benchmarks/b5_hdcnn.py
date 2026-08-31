"""Benchmark 5: HDCnn -- frozen learned feature extractor -> HDC classification.

The standard HDCnn recipe replaces a network's classification head with HDC:
train the backbone normally with cross-entropy, then (after freezing) classify
its penultimate-layer embeddings by encoding them to hypervectors and bundling
class prototypes. The questions here are:

  * Does replacing the head with HDC on top of a *learned* encoder keep the
    accuracy / robustness profile of raw-feature HDC, or does it come to rely
    on the encoder instead?
  * On identical frozen embeddings, does the HDC head match a learned linear
    head / an end-to-end trained network?

Methods added (backbone is MLPHead's architecture, 128-64-64, frozen after CE):
  * emb-hdc          : embeddings -> sign(emb P) -> prototypes          (HDCnn)
  * emb-linear       : embeddings -> linear head (LR)         [classifier head]
  * emb-rf-linear-bin: embeddings -> sign(emb P) -> linear head [learned HD head]
  * hdc-raw / mlp    : raw-feature HDC / end-to-end network from benchmarks 3/4
"""
from __future__ import annotations

import numpy as np

from hdc_empirical.data import gaussian_blobs, latent_task
from hdc_empirical.metrics import accuracy
from hdc_empirical.models import (
    HDC,
    LinearHead,
    MLPHead,
    NNEncoder,
    RFLinearBin,
)

D = 128
K_ACC = 6
K_ROB = 4
EMB = 64
D_HDC = 10_000

# same task hyperparameters as b1 / b3 for direct comparison
ACC_SIZES = [10, 50, 200, 1000]
ACC_TEST = 300
ROB_TRAIN = 2000
ROB_TEST = 1000
NOISE_LEVELS = [0.0, 1.0, 2.0, 4.0, 8.0]
SALT_FRACS = [0.05, 0.1, 0.2, 0.4]
SHIFT_TAUS = [0.0, 1.0, 2.0, 4.0]
LABEL_FLIPS = [0.0, 0.2, 0.4]


def _make_heads() -> dict:
    return {
        "emb-hdc": HDC(d=EMB, D=D_HDC, seed=0),
        "emb-linear": LinearHead(d=EMB, seed=0),
        "emb-rf-linear-bin": RFLinearBin(d=EMB, D=D_HDC, seed=0, epochs=40),
    }


# --------------------------------------------------------------------------- #
# 5a. Accuracy -- head-only learning curves on frozen embeddings
# --------------------------------------------------------------------------- #
def _accuracy() -> dict:
    X, y, _ = gaussian_blobs(
        n_per_class=max(ACC_SIZES) + ACC_TEST, k=K_ACC, d=D,
        center_norm=2.0, sigma=1.0, seed=0,
    )
    Xtr_full = np.vstack([X[y == c][:max(ACC_SIZES)] for c in range(K_ACC)])
    ytr_full = np.repeat(np.arange(K_ACC), max(ACC_SIZES))
    Xte = np.vstack([X[y == c][max(ACC_SIZES):] for c in range(K_ACC)])
    yte = np.repeat(np.arange(K_ACC), ACC_TEST)

    out: dict[str, dict[str, float]] = {}
    for s in ACC_SIZES:
        Xs = np.vstack([Xtr_full[c * max(ACC_SIZES): c * max(ACC_SIZES) + s]
                        for c in range(K_ACC)])
        ys = np.repeat(np.arange(K_ACC), s)
        # backbone trained on this subset, then frozen
        bb = NNEncoder(d=D, K=K_ACC, seed=0).fit(Xs, ys)
        Emb_tr, Emb_te = bb.embed(Xs), bb.embed(Xte)
        row: dict[str, float] = {}
        for name, head in _make_heads().items():
            head.fit(Emb_tr, ys)
            row[name] = round(accuracy(yte, head.predict(Emb_te)), 4)
        # end-to-end network (same architecture, trained jointly) as reference
        mlp = MLPHead(d=D, K=K_ACC, seed=0).fit(Xs, ys)
        row["mlp"] = round(accuracy(yte, mlp.predict(Xte)), 4)
        out[f"per_class={s}"] = row
    return out


# --------------------------------------------------------------------------- #
# 5b. Robustness on frozen embeddings
# --------------------------------------------------------------------------- #
def _robustness() -> dict:
    rng = np.random.default_rng(0)
    data = latent_task(ROB_TRAIN, ROB_TEST, d=D, seed=0)
    Xtr, ytr = data["train"]
    Xte, yte = data["test"]

    bb = NNEncoder(d=D, K=K_ROB, seed=0).fit(Xtr, ytr)
    Emb_tr, Emb_te = bb.embed(Xtr), bb.embed(Xte)

    raw_models = {
        "hdc-raw": HDC(d=D, D=D_HDC, seed=0),
        "mlp": MLPHead(d=D, K=K_ROB, seed=0),
    }
    for m in raw_models.values():
        m.fit(Xtr, ytr)
    emb_models = _make_heads()
    for m in emb_models.values():
        m.fit(Emb_tr, ytr)

    def predict(Xn, y):
        accs = {}
        for name, m in raw_models.items():
            accs[name] = round(accuracy(y, m.predict(Xn)), 4)
        En = bb.embed(Xn)
        for name, m in emb_models.items():
            accs[name] = round(accuracy(y, m.predict(En)), 4)
        return accs

    clean = predict(Xte, yte)

    gauss = {}
    for a in NOISE_LEVELS:
        Xn = Xte + rng.normal(0.0, a, size=Xte.shape).astype(np.float32)
        gauss[f"a={a}"] = predict(Xn, yte)

    salt = {}
    for f in SALT_FRACS:
        Xn = Xte.copy()
        mask = rng.random(Xte.shape) < f
        val = np.where(Xn[mask] >= 0, 6.0, -6.0)
        Xn[mask] = val
        salt[f"frac={f}"] = predict(Xn, yte)

    v = rng.normal(0.0, 1.0, size=(D,))
    v = v / np.linalg.norm(v)
    shift = {}
    for tau in SHIFT_TAUS:
        u = rng.normal(0.0, tau, size=(len(Xte),)).astype(np.float32)
        Xn = Xte + np.outer(u, v).astype(np.float32)
        shift[f"tau={tau}"] = predict(Xn, yte)

    lab = {}
    for p in LABEL_FLIPS:
        yl = ytr.copy()
        flips = rng.random(len(yl)) < p
        new = rng.integers(0, K_ROB, size=len(yl))
        yl[flips] = new[flips]
        accs = {}
        for name, m in raw_models.items():
            m.fit(Xtr, yl)
            accs[name] = round(accuracy(yte, m.predict(Xte)), 4)
        for name, m in emb_models.items():
            m.fit(Emb_tr, yl)
            accs[name] = round(accuracy(yte, m.predict(Emb_te)), 4)
        lab[f"flip={p}"] = accs

    return {"clean_test_acc": clean, "gaussian_noise": gauss,
            "salt_pepper": salt, "domain_shift": shift, "label_noise": lab}


def run() -> dict:
    return {
        "task": "hdcnn_backbone=d-h-h=128-64-64_emb=64",
        "accuracy_heads": _accuracy(),
        "robustness": _robustness(),
    }
