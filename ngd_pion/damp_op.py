"""Damping applied to the operator's spectrum rather than to its ingredients.

`factorization.basis_congruence` floors three spectra: `eig(B)`, `eig(C)`, and
the pencil's `lam`. Only the third reaches the denominator. The first two go
into `B^-1/2` and therefore change the **basis** `P` itself, so `eps` is doing
two unrelated jobs at once:

* a *numerical* one -- `B^-1/2` has to exist, and `B` may be singular;
* a *statistical* one -- small eigenvalues are badly determined by a finite
  sample and should not be trusted to set a step size.

Splitting them matters because the second is not basis-independent as written.
`F` acts on the algebra with eigenvalues `2(lam_i + lam_j)`, and eigenvalues of
a linear operator are invariant under *any* change of basis, so a floor applied
there is a property of `F`. A floor applied to `lam` first is a property of the
pencil, and which pencil one gets depends on how the pair `(B, C)` was split.
Measured consequence: on this model `null_frac` -- the share of `A`'s own
spectrum under the floor -- ran at 0.33 while `floored_frac_in`, the share of
the *pencil's* spectrum under it, ran at 0.03. Same matrix, same `eps`, an
order of magnitude apart.

So here `eps_numeric` keeps `B^-1/2` computable and nothing more, and `eps`
floors `2(lam_i + lam_j)` where it means one thing.

Whether this helps the trust region is a separate question and is not assumed:
a floored pair still gets roughly `4 eps lam_max` under either convention, so
the `1/eps` amplification that collapses `alpha` when a degenerate direction
turns over is untouched by this change alone.
"""

from __future__ import annotations

import torch

from .factorization import Basis
from .linalg import floor_eigenvalues, is_identity, safe_eigh

__all__ = ["basis_identity_anchor_op", "basis_congruence_op", "build_bases_op"]


def basis_identity_anchor_op(C: torch.Tensor, eps: float) -> Basis:
    """`F(X) = 2(XC + CX)`. `P` is orthogonal and `lam` is left raw."""
    lam, Q = safe_eigh(C)
    return Basis(P=Q, lam=lam.clamp_min(0.0), orthogonal=True, eps=eps)


def basis_congruence_op(
    B: torch.Tensor, C: torch.Tensor, eps: float, eps_numeric: float
) -> Basis:
    """`F(X) = 2(BXC + CXB)` with neither factor the identity.

    `eps_numeric` floors `B` only so that its inverse square root exists; at
    `1e-7` that leaves `B^-1/2` with a condition number of about 3200, which
    fp32 carries comfortably. `C` is not floored at all -- nothing is divided
    by it. The statistical floor is `eps`, and it is applied to
    `2(lam_i + lam_j)` by `Basis.denominator`.
    """
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, eps_numeric)
    B_inv_half = (Ub * wb.rsqrt().unsqueeze(-2)) @ Ub.transpose(-1, -2)

    M = B_inv_half @ C @ B_inv_half
    lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
    return Basis(P=B_inv_half @ Q, lam=lam.clamp_min(0.0), orthogonal=False, eps=eps)


def build_bases_op(
    W: torch.Tensor, A: torch.Tensor, eps: float, eps_numeric: float = 1e-7
) -> tuple[Basis, Basis]:
    """`build_bases` with the floor moved to the operator. `S = I` as there."""
    Wt = W.transpose(-1, -2)
    gram_in = Wt @ W
    gram_out = W @ A @ Wt
    basis_in = (
        basis_identity_anchor_op(A, eps)
        if is_identity(gram_in)
        else basis_congruence_op(A, gram_in, eps, eps_numeric)
    )
    return basis_in, basis_identity_anchor_op(gram_out, eps)
