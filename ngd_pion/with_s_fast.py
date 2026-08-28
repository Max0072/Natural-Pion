"""`NGDPionS` plus the three diagnostics the trust region is read from.

**The name is a leftover and the split has closed.** This class existed because
`with_s.NGDPionS._apply` took `angle` from `torch.linalg.matrix_norm(X, 2)`,
which cost 69.5 s of a 73.5 s step -- cusolver falls back to a slow path on the
1376x1376 skew matrices. The power iteration has since moved into `NGDPionS`
itself, along with `angle_iters` and `angle_warmup`, so **the two now produce
bit-identical trajectories** and this one is not faster than its parent.

What is left of the difference is that this class records `quad`, `curv` and
`pred_drop` in the optimizer state, which `harness.train` uses for the reduction
ratio and `harness.instrument` writes to the per-layer rows. That is the whole
of it, and `tests/test_fast.py` pins both halves: equal weights, and the three
extra keys.

`ngd-pion-s` maps here, so this is the class every result in this repository was
produced by. It is kept rather than folded into `NGDPionS` because the reference
is meant to stay free of instrumentation, not because it costs anything.
"""

from __future__ import annotations

import torch

from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .linalg import cayley, spectral_norm
from .with_s import NGDPionS

__all__ = ["FastNGDPionS"]


class FastNGDPionS(NGDPionS):
    """`NGDPionS`, with `angle` from power iteration.

    Args:
        angle_iters: iterations per step once the cached vector is warm. Two is
            generous; measured relative error from a warm start is `1e-4` after
            one.
        angle_warmup: iterations spent where no usable cached vector exists --
            the first step, and the step after each refactorisation. The two
            places warmth is unavailable are paid for explicitly rather than
            hoped over.
    """

    def __init__(self, params, *, angle_iters: int = 2, angle_warmup: int = 50, **kwargs):
        if angle_iters < 0 or angle_warmup < 0:
            raise ValueError("angle iteration counts must be non-negative")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["angle_iters"] = angle_iters
            group["angle_warmup"] = angle_warmup

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]

        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        D = state["cov_backward"].matrix.to(device=p.device, dtype=dt)
        Wt = W.transpose(-1, -2)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, Wt @ D @ W, X_in)).sum() + (
            X_out * fisher_apply(D, W @ A @ Wt, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)
        state["quad"] = quad
        state["curv"] = curv

        c = group["lr"] * float(alpha)
        # The one place this class departs from `NGDPionS`, and it is a
        # diagnostic: nothing below reads `angle`, and nothing in `_step` does
        # either. Left on the device as a tensor because `harness.instrument`
        # calls `float()` on it every few hundred steps and there is no reason
        # to sync every step.
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        sigma_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        sigma_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        state["angle"] = c * torch.maximum(sigma_in, sigma_out)
        # First-order predicted decrease from this layer's rotation. The sign
        # convention is stated in `direction`: with `X` unsigned and the step
        # `Cayley(-eta alpha X)`, the loss falls by `eta alpha quad`. The
        # factor of a half went with the 2026-08-28 generator convention:
        # `<G, W X>` is now `<G_in, X>` rather than half of it.
        # `harness.train` sums these and divides the measured decrease by the
        # total -- the ratio a trust region is actually meant to watch, and
        # which this method has never once looked at.
        state["pred_drop"] = c * quad

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1
