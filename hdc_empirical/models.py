"""Model implementations with a common interface.

Every model exposes at minimum:
    fit(X, y)       -> self
    predict(X)      -> np.ndarray of class indices
    logits(X)       -> np.ndarray (n, K) of per-class scores
    n_params()      -> int  (number of stored parameters / coefficients)
and optionally:
    partial_fit(X, y) -> incremental update
    importance_raw(X) -> (d,) mean-abs input-feature importance
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def random_projection(d: int, D: int, seed: int) -> np.ndarray:
    """Gaussian random projection with output variance ~ O(1)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, D)).astype(np.float32)


# --------------------------------------------------------------------------- #
# HDC: random projection + binarization + class-prototype bundling
# --------------------------------------------------------------------------- #
class HDC:
    name = "hdc"
    kind = "prototype"

    def __init__(self, d: int, D: int = 10000, seed: int = 0):
        self.d, self.D = d, D
        self.P = random_projection(d, D, seed)
        self._sum = None      # (K, D) int64 running accumulation
        self.prototypes = None  # (K, D) int8 in {-1, +1}
        self.counts = None
        self.K = None

    # -- encoding ---------------------------------------------------------- #
    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.sign(X.astype(np.float32) @ self.P).astype(np.int8)

    # -- training ---------------------------------------------------------- #
    def _init(self, y: np.ndarray) -> None:
        self.K = int(np.max(y)) + 1
        if self._sum is None:
            self._sum = np.zeros((self.K, self.D), dtype=np.int64)
            self.counts = np.zeros(self.K, dtype=np.int64)

    def _binarize(self) -> None:
        self.prototypes = np.sign(self._sum).astype(np.int8)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HDC":
        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "HDC":
        self._init(y)
        H = self.encode(X)
        for c in range(self.K):
            m = H[y == c]
            if len(m) > 0:
                self._sum[c] += m.sum(axis=0)
                self.counts[c] += len(m)
        self._binarize()
        return self

    # -- inference --------------------------------------------------------- #
    def scores(self, X: np.ndarray) -> np.ndarray:
        H = self.encode(X).astype(np.float32)
        return H @ self.prototypes.astype(np.float32).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X).argmax(axis=1)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X)

    # -- bookkeeping ------------------------------------------------------- #
    def n_params(self) -> int:
        protos = 0 if self.prototypes is None else int(self.prototypes.size)
        return int(self.d * self.D) + protos

    def storage_bytes(self) -> int:
        protos = 0 if self.prototypes is None else int(self.prototypes.nbytes)
        return int(self.P.nbytes) + protos


# --------------------------------------------------------------------------- #
# Classifier head trained on a FIXED random-feature encoding.
# rf-linear  : continuous random features phi(x) = xP
# rf-linear-bin : binarized random features phi(x) = sign(xP)  (same as HDC)
# The head is a single torch Linear layer trained by Adam (i.e., the canonical
# "classifier head of a network", trained on frozen features).
# --------------------------------------------------------------------------- #
class _LinearHead(nn.Module):
    def __init__(self, D: int, K: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.head = nn.Linear(D, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RFLinear:
    """Linear classifier head (torch, Adam) on random features."""

    name = "rf-linear"
    kind = "head-sgd"

    def __init__(self, d: int, D: int = 10000, seed: int = 0, binarize: bool = False,
                 epochs: int = 60, lr: float = 1e-2, batch: int = 64):
        self.d, self.D, self.binarize = d, D, binarize
        self.seed, self.epochs, self.lr, self.batch = seed, epochs, lr, batch
        self.P = random_projection(d, D, seed)
        self.model: _LinearHead | None = None

    def _feat(self, X: np.ndarray) -> np.ndarray:
        F = X.astype(np.float32) @ self.P
        if self.binarize:
            F = np.sign(F)
        return F

    def _init_model(self, K: int) -> None:
        self.model = _LinearHead(self.D, K, self.seed)
        self._opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._loss = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFLinear":
        K = int(np.max(y)) + 1
        self._init_model(K)
        F = torch.tensor(self._feat(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        n = len(yt)
        self.model.train()
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for i in range(0, n, self.batch):
                b = idx[i:i + self.batch]
                self._opt.zero_grad()
                loss = self._loss(self.model(F[b]), yt[b])
                loss.backward()
                self._opt.step()
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "RFLinear":
        """Incremental mode: one SGD step over the batch (online learning)."""
        K = int(np.max(y)) + 1
        if self.model is None:
            self._init_model(K)
        F = torch.tensor(self._feat(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        self.model.train()
        self._opt.zero_grad()
        loss = self._loss(self.model(F), yt)
        loss.backward()
        self._opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(self._feat(X), dtype=torch.float32)).numpy()

    def importance_raw(self) -> np.ndarray:
        """Map head weights back through P to raw-input feature importance."""
        w = self.model.head.weight.detach().numpy().T   # (D, K)
        raw = self.P @ w                                # (d, K)
        return np.abs(raw).mean(axis=1)

    def n_params(self) -> int:
        if self.model is None:
            return int(self.d * self.D)
        return int(self.d * self.D) + sum(p.numel() for p in self.model.parameters())

    def storage_bytes(self) -> int:
        head = 0 if self.model is None else sum(p.numel() for p in self.model.parameters()) * 4
        return int(self.P.nbytes) + head


class RFLinearBin(RFLinear):
    name = "rf-linear-bin"

    def __init__(self, d: int, D: int = 10000, seed: int = 0):
        super().__init__(d, D, seed, binarize=True)


# --------------------------------------------------------------------------- #
# Plain logistic-regression head on raw features
# --------------------------------------------------------------------------- #
class LinearHead:
    name = "linear"
    kind = "head-batch"

    def __init__(self, d: int, C: float = 1.0, seed: int = 0):
        self.d = d
        self.clf = LogisticRegression(solver="lbfgs", max_iter=3000, C=C, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearHead":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(X)

    def importance_raw(self) -> np.ndarray:
        return np.abs(self.clf.coef_).mean(axis=0)

    def n_params(self) -> int:
        return int(self.clf.coef_.size)

    def storage_bytes(self) -> int:
        return int(self.clf.coef_.nbytes)


# --------------------------------------------------------------------------- #
# Small MLP (feature extractor + classifier head) trained end-to-end
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, d: int, h: int, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead:
    name = "mlp"
    kind = "head-sgd"

    def __init__(self, d: int, K: int, h: int = 64, seed: int = 0,
                 epochs: int = 60, lr: float = 1e-2, batch: int = 64):
        self.d, self.K, self.h = d, K, h
        self.seed, self.epochs, self.lr, self.batch = seed, epochs, lr, batch
        self.model: _MLP | None = None

    def _init_model(self) -> None:
        torch.manual_seed(self.seed)
        self.model = _MLP(self.d, self.h, self.K)
        self._opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._loss = nn.CrossEntropyLoss()

    def _step(self, Xt: torch.Tensor, yt: torch.Tensor, idx: torch.Tensor) -> None:
        self._opt.zero_grad()
        out = self.model(Xt[idx])
        loss = self._loss(out, yt[idx])
        loss.backward()
        self._opt.step()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPHead":
        self._init_model()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        n = len(Xt)
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for i in range(0, n, self.batch):
                self._step(Xt, yt, idx[i:i + self.batch])
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "MLPHead":
        """Incremental mode: run one SGD step over the batch (online learning)."""
        if self.model is None:
            self._init_model()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        self.model.train()
        self._step(Xt, yt, torch.arange(len(Xt)))
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def importance_raw(self, X: np.ndarray) -> np.ndarray:
        """Gradient-based attribution: mean |d logit_c / d x_j| over classes."""
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        out = self.model(Xt)
        n, K = out.shape
        grad = torch.zeros(self.d)
        for c in range(K):
            out[:, c].sum().backward(retain_graph=True)
            grad = grad + Xt.grad.detach().abs().mean(dim=0)
            Xt.grad.zero_()
        return grad.numpy()

    def n_params(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def storage_bytes(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters()) * 4


# --------------------------------------------------------------------------- #
# Reference models
# --------------------------------------------------------------------------- #
class SVMHead:
    name = "svm"
    kind = "kernel"

    def __init__(self, d: int, C: float = 1.0, seed: int = 0):
        self.d = d
        self.clf = SVC(kernel="rbf", gamma="scale", C=C, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMHead":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(X)

    def n_params(self) -> int:
        return int(self.clf.support_vectors_.shape[0]) * self.d

    def storage_bytes(self) -> int:
        return int(self.clf.support_vectors_.nbytes)


class KNNHead:
    name = "knn"
    kind = "instance"

    def __init__(self, d: int, k: int = 5):
        self.d = d
        self.clf = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="euclidean")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNHead":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def logits(self, X: np.ndarray) -> np.ndarray:
        probs = self.clf.predict_proba(X)
        return probs

    def n_params(self) -> int:
        return int(self.clf._fit_X.shape[0] * self.d)

    def storage_bytes(self) -> int:
        return int(self.clf._fit_X.nbytes)
