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

from .linalg import exact_fp32

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
        # Two separate ways this matmul stops being fp32, and both have to be
        # closed here.
        #
        # `exact_fp32` closes TF32: on an Ampere-or-newer card torch runs fp32
        # matmuls with ten bits of mantissa unless told otherwise.
        #
        # `autocast(enabled=False)` closes the larger one. `observe` is called
        # from a `register_forward_pre_hook`, so it runs *inside* the training
        # loop's autocast block, and autocast dispatches by **operation**: it
        # does not care that `flat` is fp32, it substitutes a bf16 matmul and
        # returns bf16. `self._matrix` then takes that dtype on the first
        # observation and `mul_().add_()` keeps it, so the one statistic this
        # method inverts spends the entire run at eight bits of mantissa.
        #
        # That is not a small loss of accuracy, it is a change of kind.
        # Quantising the *activations* is harmless -- the gram of bf16 vectors
        # is still a gram, so still PSD, and measured it raises `lam_min`
        # because the quantisation noise contributes `E[e e^T] >= 0`. Rounding
        # the *assembled matrix* entrywise has no sign constraint, and on a
        # trained model, whose activations have collapsed onto a
        # low-dimensional subspace, it pushes the near-zero eigenvalues
        # straight through zero. Measured on LLaMA-60M at step 499: 6253
        # negative eigenvalues across 56 layers, `lam_min = -2.128` against
        # `lam_max = 446`, matching the predicted `2 sqrt(512) eps_bf16 max|A|`
        # of 2.1. Downstream, `curv = <X, F(X)>` -- provably non-negative for
        # PSD factors -- came out at -1.336, was clamped to `+tiny`, and pinned
        # `alpha` at its cap for every run this project has produced.
        #
        # Neither existing guard could see it. `_ALLOWED` checks `self.dtype`,
        # which is honestly fp32 throughout; `exact_fp32` governs how an fp32
        # matmul is executed, not whether one happens at all. What was never
        # checked is the only thing that mattered: the dtype that came back.
        with torch.autocast(flat.device.type, enabled=False), exact_fp32():
            gram = (flat.transpose(0, 1) @ flat) / flat.shape[0]
        if gram.dtype != self.dtype:
            raise RuntimeError(
                f"covariance gram came back {gram.dtype}, expected {self.dtype}; "
                "some dispatch mechanism is overriding the accumulation precision"
            )
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
        """Reload, restoring the accumulation dtype.

        The cast is not cosmetic. `mul_().add_()` keeps whatever dtype
        `_matrix` already has, so a checkpoint written before the autocast bug
        above was closed carries a bf16 matrix that would stay bf16 for the
        whole resumed run, silently, however correct `observe` now is. The
        precision it lost is gone either way -- this only stops it spreading.
        """
        matrix = state["matrix"]
        self._matrix = None if matrix is None else matrix.to(self.dtype)
        self.count = state["count"]
        self.beta = state["beta"]
