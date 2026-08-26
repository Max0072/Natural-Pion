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

* `skew_ratio` and `quad_over_curv` are recorded, neither of which the
  reference computes. Both are diagnostics and nothing reads them inside the
  step. See `_apply`.

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

    def __init__(
        self,
        params,
        *,
        angle_iters: int = 2,
        angle_warmup: int = 50,
        diag_every: int = 0,
        angle_max: float = 0.0,
        **kwargs,
    ):
        if angle_iters < 0 or angle_warmup < 0:
            raise ValueError("angle iteration counts must be non-negative")
        if diag_every < 0:
            raise ValueError(f"diag_every must be non-negative, got {diag_every}")
        if angle_max < 0.0:
            raise ValueError(f"angle_max must be non-negative, got {angle_max}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["angle_iters"] = angle_iters
            group["angle_warmup"] = angle_warmup
            group["diag_every"] = diag_every
            group["angle_max"] = angle_max

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

        # `alpha` is clamped at `alpha_max = 1`, and in every run so far it sits
        # at the cap, so it reports nothing about how far above the cap the
        # ratio actually is. That number is the direct test of whether the
        # curvature is underestimated.
        #
        # The arithmetic that makes it worth logging: with `alpha` at the cap
        # the applied step is just `lr`, and `lr = 3e-3` is what works. Were the
        # curvature right, the trust region's own optimum would be
        # `quad / curv`, so matching `3e-3` needs `curv` about 330x larger than
        # `quad`. Meanwhile `alpha >= 1` says `curv <= quad`. The gap between
        # those two is the size of whatever the local formula is missing --
        # plausibly the Jacobian from this layer's output to the logits, which
        # `<D, H D> = tr(D A D^T)` omits and which is exact only for a
        # quadratic loss at the output layer.
        #
        # Kept as a tensor: `harness.instrument` calls `float()` on it every few
        # hundred steps, so there is no reason to sync every step.
        state["quad_over_curv"] = quad / curv.clamp_min(torch.finfo(dt).tiny)
        # The two of them raw as well. The ratio alone cannot say which side
        # degenerates: `quad = sum g^2 / lamt` is measured against the FLOORED
        # operator, because `natural_gradient` divides by `basis.denominator`,
        # while `curv = sum g^2 lam / lamt^2` is measured against the RAW one,
        # because `fisher_apply` is handed `A` and `W^T W` unmodified. The
        # floor only raises, so `lam <= lamt` termwise and `curv <= quad`
        # identically -- which is why `alpha` sits at the cap by construction
        # rather than because the curvature is underestimated.
        state["quad"] = quad
        state["curv"] = curv

        # How much of `quad` comes from index pairs the floor touched. This is
        # the direct measurement of the mechanism above: if the collapse is the
        # floor, this goes to 1 exactly where `quad_over_curv` blows up. Two
        # extra n^3 matmuls per side, so it runs on the logging cadence rather
        # than every step, and never at all when `diag_every` is 0.
        every = group["diag_every"]
        if every and state["step"] % every == 0:
            state["floor_share_in"] = _floor_share(G_in, basis_in, group["eps"])
            state["floor_share_out"] = _floor_share(G_out, basis_out, group["eps"])

        # `angle` comes from power iteration rather than `matrix_norm(X, 2)`.
        # `harness.instrument.layer_diagnostics` reads it every few hundred
        # steps and calls `float()` there, which is why it stays a device
        # tensor rather than syncing every step.
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        sigma_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        sigma_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        sigma = torch.maximum(sigma_in, sigma_out)

        c = group["lr"] * alpha
        state["angle_requested"] = c * sigma
        # The second trust region, and the one the method never had.
        #
        # `alpha` cannot bound the step. `X = F^-1 G` makes `curv = <X, F(X)> =
        # <X, G> = quad` an identity, true at any radius whatever, so
        # `quad/curv` measures only the drift between the `F` that built `X`
        # and the `F` that judges it -- staleness, and nothing else. On a
        # freshly factorised basis it is exactly 1 by construction, so the step
        # straight after every refactorisation is unbounded. Measured: at
        # `eta = 1.0` that first step is a rotation of 5.66 radians, the model
        # is wrecked inside twenty steps (loss 49.7), and it recurs on the
        # refactorisation cycle.
        #
        # What is missing is a bound on the *domain* of the quadratic model
        # rather than on its self-consistency. The model is a Taylor expansion
        # in the algebra and `Cayley` departs from the exponential at order
        # `||A||^2`, so there is a radius beyond which it means nothing;
        # ALGORITHM.md puts it at 0.1 rad. Capping there costs the working
        # configuration nothing -- its measured `angle_max` is 0.057 -- and cuts
        # that first step by a factor of 57.
        #
        # `c` stays a tensor: `cayley` takes either, and forcing a sync here
        # would cost more than the step it guards (see `spectral_norm`).
        if group["angle_max"]:
            tiny = torch.finfo(dt).tiny
            c = torch.minimum(c, sigma.new_tensor(group["angle_max"]) / sigma.clamp_min(tiny))
        state["angle"] = c * sigma

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1


def _floor_share(G_skew: torch.Tensor, basis, eps: float) -> float:
    """Fraction of one side's `quad` contributed by floored index pairs.

    In the basis, `quad` is `sum_ij Gb_ij^2 / denominator_ij` with
    `Gb = P^T G P`, every term non-negative, so the share is a genuine
    fraction in `[0, 1]`. A pair counts as floored when either of its two
    eigenvalues was raised, since `denominator_ij = 2(lam_i + lam_j)` is a sum
    and one raised index is enough to change it.

    `basis.lam` is already floored, and every value the floor touched is
    exactly `eps * lam_max`; `lam_max` is untouched by flooring, so the raised
    entries are recoverable from the floored spectrum alone.
    """
    lam = basis.lam
    at_floor = lam <= lam.amax(dim=-1, keepdim=True) * eps * (1.0 + 1e-5)
    P = basis.P
    Gb = P.transpose(-1, -2) @ G_skew @ P
    contrib = Gb * Gb / basis.denominator
    pair = at_floor.unsqueeze(-1) | at_floor.unsqueeze(-2)
    total = contrib.sum()
    return float((contrib * pair).sum() / total.clamp_min(torch.finfo(lam.dtype).tiny))
