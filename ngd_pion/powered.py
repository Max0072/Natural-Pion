"""Preconditioning by `F^-p` rather than by `F^-1`, which is what Adam does.

Natural gradient divides by the curvature; Adam divides by the *square root* of
the second moment. That one difference is why Adam does not suffer the failure
measured here on `ngd-pion-s`.

    natural gradient    step = F^-1 g          ~ 1 / (||W|| ||delta|| ||x||)
    Adam                step = g / sqrt(v)     scale-invariant

If every gradient shrinks by `c`, Adam's step is unchanged, while `F^-1 g`
grows like `1/c`. And that is exactly the pathology: we use the **empirical**
Fisher, whose factor is the residual, so as the model fits, `delta -> 0`,
`F ~ delta^2 -> 0`, and the step diverges. Measured: at step 0 on a perfectly
fresh basis, `ngd-pion-s` predicted a loss decrease of 148 -- against a loss of
10.5 -- and delivered nothing; `rho` had median `-0.000` over 150 steps, while
the `S = I` arm sat at 0.384.

In the basis, `F` acts elementwise with eigenvalues `d_ij`, and `d_ij` **is**
the second moment of that generator component, since `F` is by construction the
covariance of the generators. So the Adam analogue is exactly one exponent:

    Y_ij = G_ij / d_ij ** power     power = 1    natural gradient
                                    power = 1/2  Adam, Shampoo
                                    power = 0    the plain gradient

**And it also fixes the cross-layer scale.** With `d ~ ||A|| ||S|| ~ x^2 W^2
delta^2` and `G_in ~ W delta x`, at `power = 1/2` the step comes out as
`(W delta x) / (x W delta) = 1` -- independent of the backward signal, the
activations and the weights alike. Every layer gets a comparable rotation and
`eta` sets it directly. That is the calibration Pion obtains by fiat with
`scaling="rms"`, and the absence of which was measured here as a spread of
4 658x to 36 496x in per-layer rotation angle.

**The trust region is switched off unless `power` is 1**, and not as a
convenience. `alpha = quad / curv` is derived from `X = F^-1 G` holding
exactly, which makes it dimensionless. At `power = 1/2`, `curv` is `sum g^2`
and `quad` is `sum g^2 / sqrt(d)`, so the ratio carries units of `1/sqrt(d)`
and is not a multiplier at all.

What is given up is the claim to be a natural gradient, and with it `eta* = 1`.
That was already gone: `eta*` measures three orders below 1 for `ngd-pion-s`,
for the reason above.
"""

from __future__ import annotations

import torch

from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .linalg import cayley, spectral_norm
from .op_damped import OpDampedNGDPion

__all__ = ["PoweredNGDPion"]


class PoweredNGDPion(OpDampedNGDPion):
    """`OpDampedNGDPion` with an exponent on the operator's eigenvalues.

    Args:
        power: `1.0` reproduces `OpDampedNGDPion` exactly, trust region and
            all. `0.5` is the Adam analogue.
    """

    def __init__(self, params, *, power: float = 0.5, **kwargs):
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"power must lie in [0, 1], got {power}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["power"] = power

    def _refactor(self, params, group):
        super()._refactor(params, group)
        power = group["power"]
        if power == 1.0:
            return
        for p in params:
            self.state[p]["bases"] = tuple(
                type(b)(b.P, b.lam, b.orthogonal, b.eps, power)
                for b in self.state[p]["bases"]
            )

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        power = group["power"]
        if power == 1.0:
            super()._apply(p, group)
            return

        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]

        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        eye_out = torch.eye(W.shape[0], dtype=dt, device=W.device)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, W.T @ W, X_in)).sum() + (
            X_out * fisher_apply(eye_out, W @ A @ W.T, X_out)
        ).sum()
        # Logged, never applied: at `power != 1` the ratio has units and is not
        # a multiplier. See the module docstring.
        state["quad"], state["curv"] = quad, curv
        state["alpha"] = 1.0
        state["quad_over_curv"] = quad / curv.clamp_min(torch.finfo(dt).tiny)

        c = group["lr"]
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        sigma_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        sigma_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        sigma = torch.maximum(sigma_in, sigma_out)
        if group["angle_max"]:
            tiny = torch.finfo(dt).tiny
            c = torch.minimum(
                torch.as_tensor(c, dtype=dt, device=W.device),
                sigma.new_tensor(group["angle_max"]) / sigma.clamp_min(tiny),
            )
        state["angle"] = c * sigma
        state["pred_drop"] = 0.5 * c * quad

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)
        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1
