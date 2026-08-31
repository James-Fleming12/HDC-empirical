"""Factory for the standard method set."""
from __future__ import annotations

from hdc_empirical.models import (
    HDC,
    KNNHead,
    LinearHead,
    MLPHead,
    RFLinear,
    RFLinearBin,
    SVMHead,
)


def make_methods(d: int, k: int, mlp_epochs: int = 60) -> list:
    return [
        HDC(d=d, D=10_000, seed=0),
        RFLinear(d=d, D=10_000, seed=0),
        RFLinearBin(d=d, D=10_000, seed=0),
        LinearHead(d=d, seed=0),
        MLPHead(d=d, K=k, seed=0, epochs=mlp_epochs),
        SVMHead(d=d, seed=0),
        KNNHead(d=d),
    ]
