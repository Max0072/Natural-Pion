"""ALGORITHM.md §3 -- the input covariance `A = E[x x^T]`.

The only statistic the method keeps. `S = E[delta delta^T]` is taken to be the
identity, so no backward signal is needed and no backward hook has to exist.

**Precision is not negotiable here.** Perturbing `A` at bf16 level produces a
step wrong by three to four orders of magnitude: the small eigenvalues of `A`
are both the least well determined and the most amplified, since the method
divides by their square roots. fp32 is the floor regardless of what precision
the surrounding model trains in.
"""

from __future__ import annotations

import torch

__all__ = ["CovarianceAccumulator"]

_ALLOWED = (torch.float32, torch.float64)


class CovarianceAccumulator:
    """Exponential moving average of `x x^T` over observed activations.

    The first observation initialises the average rather than being blended
    into a zero matrix. Starting from zeros would leave `A` scaled by
    `1 - beta` for the first steps, and since the factorisation divides by its
    eigenvalues that bias is not harmless. Initialising means the estimate is
    a valid covariance from the very first refactor.
    """

    def __init__(self, beta: float = 0.95, dtype: torch.dtype = torch.float32) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must lie in [0, 1), got {beta}")
        if dtype not in _ALLOWED:
            raise ValueError(
                f"covariance must be accumulated in fp32 or fp64, got {dtype}; "
                "lower precision corrupts the small eigenvalues the method divides by"
            )
        self.beta = beta
        self.dtype = dtype
        self._matrix: torch.Tensor | None = None
        self.count = 0

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    @property
    def matrix(self) -> torch.Tensor:
        if self._matrix is None:
            raise RuntimeError("no activations observed yet")
        return self._matrix

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Fold a batch of activations in. `x` is `(..., d)`; leading dims are flattened."""
        flat = x.detach().reshape(-1, x.shape[-1]).to(self.dtype)
        if flat.shape[0] == 0:
            return
        gram = (flat.transpose(0, 1) @ flat) / flat.shape[0]
        if self._matrix is None:
            self._matrix = gram
        else:
            self._matrix.mul_(self.beta).add_(gram, alpha=1.0 - self.beta)
        self.count += flat.shape[0]

    def to(self, device) -> "CovarianceAccumulator":
        """Move the accumulated matrix. Needed after a checkpoint reload."""
        if self._matrix is not None:
            self._matrix = self._matrix.to(device)
        return self

    def state_dict(self) -> dict:
        return {"matrix": self._matrix, "count": self.count, "beta": self.beta}

    def load_state_dict(self, state: dict) -> None:
        self._matrix = state["matrix"]
        self.count = state["count"]
        self.beta = state["beta"]
