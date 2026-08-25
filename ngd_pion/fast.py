"""NGD-Pion with the cost work in it, and nothing else.

`optimizer.py` is the reference. It is a direct transcription of
`ALGORITHM.md`, deliberately left unoptimised, so that there is something to
check against -- the same role `reference.py` plays for the mathematics, one
level up. Nothing in this module may be back-ported into it.

The rule this subclass lives under: **every difference is either exactly
equivalent or confined to a diagnostic.** Nothing here is allowed to move the
weights. `tests/test_fast.py` pins that by running both optimizers from the
same initial state on the same gradients and comparing trajectories, not
outputs.

What is different so far:

* `skew_ratio` is recorded, which the reference does not compute at all. It
  is a diagnostic and nothing reads it inside the step. See `_apply`.

* `angle` comes from `spectral_norm` -- power iteration -- instead of
  `torch.linalg.matrix_norm(X, 2)`. Measured on one RTX PRO 6000 Blackwell,
  the exact call cost 69.5 s of the 73.5 s optimizer step, because cusolver
  fails to converge on the 1376x1376 matrices and falls back to a slow path;
  24 of the 56 weights of LLaMA-60M have a 1376 side. Power iteration does the
  same job in 14.4 ms. It is safe to approximate precisely because `angle` is
  read by `harness.instrument` and by nothing in the step -- see `_apply`.

Still to come, and deliberately not here yet: the batched `_apply`, the
Newton-Schulz retraction, and the contracted trust-region curvature.
"""

from __future__ import annotations

import torch

from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .linalg import cayley, spectral_norm
from .optimizer import NGDPion

__all__ = ["FastNGDPion"]


class FastNGDPion(NGDPion):
    """`NGDPion`, with the diagnostics costed properly.

    Args:
        angle_iters: power iterations per step once the cached vector is warm.
            Two is generous: measured relative error from a warm start is
            `1e-4` after one.
        angle_warmup: iterations spent when there is no usable cached vector --
            on the first step, and on the step after each refactorisation,
            where `X` moves discontinuously because the basis it is built in
            has just been rebuilt. Cold convergence is genuinely poor (6%
            relative error after five iterations on a random 512x512 skew,
            because a skew matrix has its singular values in equal pairs and
            the large ones bunch), so the two places where warmth is not
            available are paid for explicitly rather than hoped over.
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

        # How much of `W^T G` is antisymmetric, which is the only part the
        # method uses. `G_in = W^T G - G^T W` is a difference of nearly equal
        # quantities, and the forward runs under bf16 autocast, so `G` carries
        # roughly `8e-3` of relative noise however it is stored afterwards.
        # Measured on synthetic `G` pushed through bf16: at a skew-to-sym ratio
        # of `1e-2` the relative error in `G_in` is already 12%, at `1e-3` it is
        # 117% -- the signal is gone. The same ratios in fp32 give `2e-5` and
        # `2e-4`, four orders better, so if this number falls the cure is the
        # precision of the gradient rather than anything in this module.
        #
        # `W^T G` is recomputed here rather than taken from `generators`, which
        # returns only the difference. One extra matmul, about 0.4% of a step,
        # and it keeps `generators` untouched so the trajectory stays
        # bit-identical to the reference.
        M = W.transpose(-1, -2) @ G
        sym = M + M.transpose(-1, -2)
        state["skew_ratio"] = float(
            torch.linalg.matrix_norm(G_in, "fro")
            / torch.linalg.matrix_norm(sym, "fro").clamp_min(torch.finfo(dt).tiny)
        )

        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        eye_out = torch.eye(W.shape[0], dtype=dt, device=W.device)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, W.T @ W, X_in)).sum() + (
            X_out * fisher_apply(eye_out, W @ A @ W.T, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)

        c = group["lr"] * float(alpha)
        # The one place this class departs from the reference, and it is a
        # diagnostic: nothing below reads `angle`, and nothing in `_step` does
        # either. `harness.instrument.layer_diagnostics` reads it every few
        # hundred steps and calls `float()` on it there, which is also why it
        # is left on the device as a tensor rather than synced every step.
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
