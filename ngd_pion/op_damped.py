"""`FastNGDPion` with the floor on the operator instead of on the pencil.

Identical to `fast.py` in every other respect -- same step, same diagnostics,
same power-iteration angle. The only difference is which spectrum `eps` acts
on, and `damp_op.py` explains why that is not a cosmetic distinction.

What this is expected to change: `eps` stops meaning three different things in
three different places, and stops depending on how the pair `(B, C)` was split.
Measured on this model, the share of `A`'s own spectrum under the floor was
0.33 while the share of the *pencil's* spectrum under it was 0.03 -- one `eps`,
one matrix, an order of magnitude apart.

What this is **not** expected to change, and it is worth saying before the run
rather than after: a pair with both indices degenerate still lands at roughly
`4 eps lam_max` under either convention. So the `1/eps` amplification that
collapses `alpha` when a degenerate direction turns over between
refactorisations is untouched. That needs a slower statistic or a shorter
refactorisation cycle, not a different place to put the same floor.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .damp_op import build_bases_op
from .fast import FastNGDPion

__all__ = ["OpDampedNGDPion"]


class OpDampedNGDPion(FastNGDPion):
    """`FastNGDPion` whose bases carry a raw `lam` and an operator-level floor.

    Args:
        eps_numeric: floor on `B` alone, present only so that `B^-1/2` exists.
            At `1e-7` that leaves the root with a condition number of about
            3200, which fp32 carries comfortably. It is not a modelling choice
            and should not need tuning; `eps` is the one that does.
    """

    def __init__(self, params, *, eps_numeric: float = 1e-7, **kwargs):
        if not 0.0 < eps_numeric < 1.0:
            raise ValueError(f"eps_numeric must lie in (0, 1), got {eps_numeric}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["eps_numeric"] = eps_numeric

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        dt = group["compute_dtype"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) for p in members])
            A = torch.stack(
                [self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p in members]
            )
            basis_in, basis_out = build_bases_op(
                W, A, group["eps"], group["eps_numeric"]
            )
            for i, p in enumerate(members):
                self.state[p]["bases"] = (
                    type(basis_in)(basis_in.P[i], basis_in.lam[i], basis_in.orthogonal,
                                   basis_in.eps),
                    type(basis_out)(basis_out.P[i], basis_out.lam[i], basis_out.orthogonal,
                                    basis_out.eps),
                )
                self.state[p]["since_refactor"] = 0
