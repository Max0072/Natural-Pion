"""`NGDPionS` with the angle diagnostic costed properly.

Stands to `with_s.py` exactly as `fast.py` stands to `optimizer.py`: same
algorithm, same trajectory, one quantity computed by a cheap approximation
instead of an exact decomposition.

`with_s.NGDPionS._apply` takes `angle` from `torch.linalg.matrix_norm(X, 2)`.
On LLaMA-60M that call cost **69.5 s of a 73.5 s step** -- cusolver fails to
converge on the 1376x1376 skew matrices and falls back to a slow path, and 24
of the 56 weights have a 1376 side. Power iteration does the same job in
14.4 ms.

It is safe to approximate because `angle` is read by `harness.instrument` and
by nothing inside the step. Nothing here changes what the optimizer does.

Cold, power iteration converges badly on a skew matrix -- its singular values
come in equal pairs and the large ones bunch, so the ratio governing
convergence sits near 1. Measured on a random 512x512 skew: 21% relative error
after one iteration, 6% after five. Warm, from a vector converged on the
previous step's `X`, one iteration gives `1e-4`. So the cached vector is spent
generously exactly twice: on the first step, and again after each
refactorisation, where `X` moves discontinuously because the basis it is built
in has just been rebuilt.
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

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1
