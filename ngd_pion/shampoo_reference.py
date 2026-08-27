"""Spec-faithful reference for `shampoo.py`, in the role `reference.py` plays.

Deliberately naive: dense `scipy.linalg.eigh`, one layer at a time, no
batching, no amortisation of the inverse roots. Nothing here is meant to be
fast. `tests/test_shampoo.py` pins the torch path against it, and where the two
disagree this file is right until proven otherwise.

It is a separate module rather than a section of `reference.py` because it is a
different algorithm, not a different section of `ALGORITHM.md`'s NGD-Pion: it
never forms a Fisher, never takes a covariance, and its preconditioner is built
from the gradients themselves.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

__all__ = ["gram", "inverse_root", "ShampooPionReference"]


def skew(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - M.T)


def generators(W: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`G_in = W^T G - G^T W`, `G_out = G W^T - W G^T`, as in `reference.py`."""
    return W.T @ G - G.T @ W, G @ W.T - W @ G.T


def gram(G: np.ndarray) -> np.ndarray:
    """`G G^T`; for skew `G` this is also `G^T G` and `-G^2`."""
    return G @ G.T


def inverse_root(P: np.ndarray, power: float, eps: float, damping: str) -> np.ndarray:
    """`P^-power`, damped by a relative floor or by the original's shift."""
    w, U = eigh(P)
    w = np.clip(w, 0.0, None)
    if damping == "floor":
        lam_max = max(w.max(), np.finfo(w.dtype).tiny)
        w = np.maximum(w, eps * lam_max)
    elif damping == "shift":
        w = w + eps
    else:
        raise ValueError(f"damping must be 'floor' or 'shift', got {damping!r}")
    return (U * w ** (-power)) @ U.T


def cayley(X: np.ndarray, c: float) -> np.ndarray:
    """`(I + c/2 X)^-1 (I - c/2 X)`, exactly orthogonal for skew `X`."""
    eye = np.eye(X.shape[-1])
    A = 0.5 * c * X
    return np.linalg.solve(eye + A, eye - A)


class ShampooPionReference:
    """One weight matrix, one optimizer. Call `step(G)` with the gradient."""

    def __init__(
        self,
        W: np.ndarray,
        lr: float = 1e-3,
        power: float = 0.25,
        beta: float = 0.0,
        eps: float = 1e-4,
        damping: str = "floor",
        t_fac: int = 25,
    ) -> None:
        self.W = W.astype(np.float64).copy()
        self.lr, self.power, self.beta = lr, power, beta
        self.eps, self.damping, self.t_fac = eps, damping, t_fac
        m, n = self.W.shape
        self.P_in = np.zeros((n, n))
        self.P_out = np.zeros((m, m))
        self.Q_in = self.Q_out = None
        self.t = 0
        self.since_refactor = 0

    def _accumulate(self, P: np.ndarray, G: np.ndarray) -> np.ndarray:
        L = gram(G)
        return P + L if self.beta == 0.0 else self.beta * P + (1.0 - self.beta) * L

    def step(self, G: np.ndarray) -> np.ndarray:
        G_in, G_out = generators(self.W, G.astype(np.float64))
        self.P_in = self._accumulate(self.P_in, G_in)
        self.P_out = self._accumulate(self.P_out, G_out)

        if self.Q_in is None or self.since_refactor >= self.t_fac:
            if self.power > 0.0:
                self.Q_in = inverse_root(self.P_in, self.power, self.eps, self.damping)
                self.Q_out = inverse_root(self.P_out, self.power, self.eps, self.damping)
            else:
                self.Q_in = np.eye(self.P_in.shape[0])
                self.Q_out = np.eye(self.P_out.shape[0])
            self.since_refactor = 0

        X_in = skew(self.Q_in @ G_in @ self.Q_in)
        X_out = skew(self.Q_out @ G_out @ self.Q_out)
        self.W = cayley(X_out, self.lr) @ self.W @ cayley(X_in, self.lr)
        self.t += 1
        self.since_refactor += 1
        return self.W
