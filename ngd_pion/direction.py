"""ALGORITHM.md §1, §5, §6 -- from a gradient to a preconditioned rotation.

Sign convention, stated once and relied on everywhere downstream: `X_in` and
`X_out` come out of `natural_gradient` **unsigned**, and the descent direction
is `Cayley(-eta * alpha * X)`. With that choice `quad`, `curv` and `alpha` are
all positive, and the step lowers the loss by `1/2 eta alpha quad` to first
order.
"""

from __future__ import annotations

import torch

from .factorization import Basis
from .linalg import skew

__all__ = ["generators", "fisher_apply", "natural_gradient", "trust_region_alpha"]


def generators(W: torch.Tensor, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`G_in = W^T G - G^T W`, `G_out = G W^T - W G^T` (§1).

    These are twice the Riemannian gradient with respect to a rotation: for
    any skew `X`, `<G, W X> = 1/2 <G_in, X>`. For a single sample `G = delta
    x^T` the result is the bivector `(W^T delta) ^ x`, of rank 2.
    """
    Wt = W.transpose(-1, -2)
    Gt = G.transpose(-1, -2)
    return Wt @ G - Gt @ W, G @ Wt - W @ Gt


def fisher_apply(B: torch.Tensor, C: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """`F(X) = 2(B X C + C X B)` (§2). Self-adjoint and PSD on skew `X`."""
    return 2.0 * (B @ X @ C + C @ X @ B)


def natural_gradient(G_skew: torch.Tensor, basis: Basis) -> torch.Tensor:
    """Solve `F(X) = G_skew` in the basis of §4.

    `X = P Y P^T` turns `2(BXC + CXB) = G` into an elementwise division by
    `2(lam_i + lam_j)`. No damping appears here: `basis.lam` is already
    floored, so the denominator is positive by construction.
    """
    P = basis.P
    Pt = P.transpose(-1, -2)
    Y = (Pt @ G_skew @ P) / basis.denominator
    return skew(P @ Y @ Pt)


def trust_region_alpha(
    quad: torch.Tensor, curv: torch.Tensor, alpha_max: float
) -> torch.Tensor:
    """`alpha = min(alpha_max, quad / curv)` (§6).

    Identically 1 on a fresh basis -- `X = F^-1 G` gives `curv = <X, F(X)> =
    <X, G> = quad` as an identity, not an approximation. What moves it is
    staleness: `X` is built from a basis `T_fac` steps old while `curv` is
    measured against the current statistics, so `alpha` reads out how far the
    factorisation has drifted behind `W`.

    `alpha_max = 1` keeps the mechanism one-sided: a stale basis can only
    shorten the step, never lengthen it.
    """
    ratio = quad / curv.clamp_min(torch.finfo(curv.dtype).tiny)
    return ratio.clamp(max=alpha_max)
