"""`FastNGDPionS` with additive Tikhonov damping and a Levenberg-Marquardt `lam`.

Why this exists
---------------

The relative floor `max(d, eps * d_max)` that every earlier variant uses can
only act on the *spread* of `F`'s spectrum, because multiplying a layer's whole
spectrum by a constant leaves `d / d_max` untouched. Three measurements say
spread is not what produces the enormous steps:

* `rho(angle, nullfrac_A) = -0.92` across 56 layers -- the *most* degenerate
  layers take the *smallest* steps;
* `rho(angle, cond_S) = -0.11`, and `cond_S` reaches 1e19 with 63% of `S`
  numerically null in one layer whose angle is entirely ordinary;
* the largest angle in the model, 6715 rad, comes from a layer with
  `null_frac = 0` and nothing below the floor.

The one strong predictor is `1 / sqrt(lam_max_S * lam_max_A)`, `rho = 0.63` --
the *scale* of the curvature, which is exactly what a relative floor cannot
see. Hence an additive term, which acts on every eigenvalue including the
largest and bounds the step at `||G|| / lam` however flat the layer is.

What the additive term actually is
----------------------------------

Not an approximation to `(F + lam I)^-1 G`. With `X = P Y P^T`, adding `lam` to
the denominator solves `F(X) + lam (P P^T)^-1 X (P P^T)^-1 = G` **exactly**;
since `P = B^-1/2 Q`, that is `F(X) + lam B X B`, Tikhonov in the metric of the
anchor. For the in-side the anchor is `A`, so the trust region is
`tr(X^T A X A)` -- the step is measured in the units in which signal actually
flows through the layer, rather than in raw parameter space. Directions of `W`
the data never excites are not penalised; directions carrying most of the input
variance are penalised most.

This matters because `W` is *not* orthogonal here. Pion preserves `W`'s
singular values; it does not keep `W` orthogonal, and measured on a real
checkpoint `||W^T W - I||_max ~ 0.83` with `cond(W) ~ 1e3`. So the in-side pair
`(A, W^T D W)` is always a genuine congruence, `P` is never orthogonal, and
damping against the Euclidean identity would be wrong -- by a factor of 7e3 at
the scale mismatch this model actually runs at.

Setting `lam`
-------------

`lam` is held **absolute**: one number shared by every layer, frozen against a
reference taken once at the first refactorisation (the median `d_max` over
layers). A per-layer `lam = c * d_max` would reintroduce exactly the blindness
being removed -- in a flat layer `d_max` is small, so `lam` would be small too.

It is then adapted by the Levenberg-Marquardt rule on the reduction ratio
`rho`, which the harness already computes every logged step and which no
variant has ever fed back into anything:

    rho > 0.75  ->  lam / 1.5     the quadratic model is holding, go bolder
    rho < 0.25  ->  lam * 1.5     the model is lying, pull in

With `lam` carrying the trust region, `eta` is meant to sit at 1 rather than be
swept. That is the point of the exercise.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import torch

from .factorization import Basis
from .linalg import floor_eigenvalues, is_identity, safe_eigh
from .with_s_fast import FastNGDPionS

__all__ = ["DampedNGDPionS"]

LAM_MIN = 1e-12
LAM_MAX = 1e12


def _basis_raw_anchor(C: torch.Tensor) -> Basis:
    """`F(X) = 2(XC + CX)`. `P` orthogonal, `lam` left raw -- nothing floored."""
    lam, Q = safe_eigh(C)
    return Basis(P=Q, lam=lam.clamp_min(0.0), orthogonal=True)


def _basis_raw_congruence(B: torch.Tensor, C: torch.Tensor, eps_numeric: float) -> Basis:
    """`F(X) = 2(BXC + CXB)`, `lam` raw.

    `eps_numeric` floors `B` only so that `B^-1/2` exists; it is not a
    modelling choice and should not need tuning. `C` is floored not at all --
    nothing is divided by it -- and neither is the pencil, because the additive
    `lam` is what keeps the denominator positive now.
    """
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, eps_numeric)
    B_inv_half = (Ub * wb.rsqrt().unsqueeze(-2)) @ Ub.transpose(-1, -2)
    M = B_inv_half @ C @ B_inv_half
    lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
    return Basis(P=B_inv_half @ Q, lam=lam.clamp_min(0.0), orthogonal=False)


class DampedNGDPionS(FastNGDPionS):
    """`FastNGDPionS` with `d + lam` and an LM-adapted `lam`.

    Args:
        lam: initial damping, as a multiple of the frozen reference scale. `0`
            disables damping and reproduces the undamped operator with raw
            (unfloored) spectra.
        lam_adapt: whether `adapt_damping` moves `lam`. Off makes this a plain
            fixed-damping arm, which is what a sweep wants.
        eps_numeric: floor on the congruence anchor so its inverse root exists.
    """

    def __init__(
        self,
        params,
        *,
        lam: float = 0.0,
        lam_adapt: bool = True,
        eps_numeric: float = 1e-7,
        **kwargs,
    ):
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if not 0.0 < eps_numeric < 1.0:
            raise ValueError(f"eps_numeric must lie in (0, 1), got {eps_numeric}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["lam"] = lam
            group["lam_adapt"] = lam_adapt
            group["eps_numeric"] = eps_numeric
            # Frozen at the first refactorisation, then never moved: `lam` is
            # absolute for the whole run, which is the property being tested.
            group["lam_reference"] = None

    # -- damping ----------------------------------------------------------

    def damping(self, group: dict) -> tuple[float, float]:
        """`(lam_in, lam_out)` in the units of each side's own `d`.

        The two sides are *not* commensurable and must not share a number.
        Measured on one 32x32 layer at the first refactorisation, the in-side
        had `d_max = 1.02e10` and the out-side `d_max = 5.17e4` -- five orders
        apart on the same weight, because each side's `d` is the spectrum of
        its own pencil in its own anchor's metric. A single `lam` split the
        difference and was wrong for both: it swamped the out-side completely
        and grazed the bottom percent of the in-side.

        Within a side the reference is shared across *layers*, which is the
        property that matters. A per-layer `lam = c * d_max` would be blind to
        a uniformly flat layer in exactly the way `max(d, eps * d_max)` is:
        scale a layer's whole spectrum by `s` and both the floor and a relative
        `lam` scale with it, so the step still grows by `1/s`. Frozen once and
        shared, `lam` stays put while a flat layer's `d` falls below it, and
        that layer's step is bounded at `||G||/lam` instead.
        """
        ref = group["lam_reference"]
        if ref is None:
            return 0.0, 0.0
        return group["lam"] * ref[0], group["lam"] * ref[1]

    def adapt_damping(self, rho: float) -> None:
        """Levenberg-Marquardt on the reduction ratio.

        Called by the harness on the steps where `rho` is measured. A NaN or a
        non-finite `rho` is not information and must not move `lam`: it means
        the predicted decrease was zero, not that the model was wrong.
        """
        if rho is None or rho != rho or rho in (float("inf"), float("-inf")):
            return
        for group in self.param_groups:
            if not group["lam_adapt"] or not group["lam"]:
                continue
            if rho > 0.75:
                group["lam"] = max(group["lam"] / 1.5, LAM_MIN)
            elif rho < 0.25:
                group["lam"] = min(group["lam"] * 1.5, LAM_MAX)

    # -- the step ---------------------------------------------------------

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        """As `NGDPionS`, but with raw spectra: the floor is gone, `lam` replaces it."""
        dt = group["compute_dtype"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        built: list[tuple[torch.Tensor, Basis, Basis, int]] = []
        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) for p in members])
            A = torch.stack(
                [self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p in members]
            )
            D = torch.stack(
                [
                    self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt)
                    for p in members
                ]
            )
            Wt = W.transpose(-1, -2)
            gram_in = Wt @ D @ W
            gram_out = W @ A @ Wt
            en = group["eps_numeric"]

            basis_in = (
                _basis_raw_anchor(A)
                if is_identity(gram_in)
                else _basis_raw_congruence(A, gram_in, en)
            )
            basis_out = (
                _basis_raw_anchor(gram_out)
                if is_identity(D)
                else _basis_raw_congruence(D, gram_out, en)
            )
            for i, p in enumerate(members):
                built.append((p, basis_in, basis_out, i))

        # One reference per side for the whole model, taken once and frozen.
        # `d_max` for a layer is `4 * lam_max`; the median over layers is used
        # rather than the max so that a single outlying layer does not set the
        # scale for all the rest.
        if group["lam_reference"] is None and built:
            per_side = []
            for which in (1, 2):
                peaks = sorted(float(4.0 * b[which].lam[b[3]].max()) for b in built)
                per_side.append(peaks[len(peaks) // 2] or 1.0)
            group["lam_reference"] = tuple(per_side)

        lam_in, lam_out = self.damping(group)
        for p, bi, bo, i in built:
            self.state[p]["bases"] = (
                Basis(bi.P[i], bi.lam[i], bi.orthogonal, lam_tikhonov=lam_in),
                Basis(bo.P[i], bo.lam[i], bo.orthogonal, lam_tikhonov=lam_out),
            )
            self.state[p]["since_refactor"] = 0

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        # `lam` moves between refactorisations, and `Basis` is frozen, so the
        # pair is rebuilt when it has drifted. `dataclasses.replace` copies the
        # dataclass and not the tensors, so this is free.
        lams = self.damping(group)
        bases = self.state[p].get("bases")
        if bases is not None and tuple(b.lam_tikhonov for b in bases) != lams:
            self.state[p]["bases"] = tuple(
                dataclasses.replace(b, lam_tikhonov=l) for b, l in zip(bases, lams)
            )
        super()._apply(p, group)
        # Removing the pencil floor leaves `B^-1/2` as the only thing keeping
        # `P` finite, and `X = P Y P^T` grows as `P^2`. With `D ~ 1e-8` -- the
        # scale the backward covariance actually runs at -- that overflows
        # fp32 silently and shows up downstream as `angle = inf`, which is a
        # bad way to find out. Fail where it happens instead.
        if not torch.isfinite(self.state[p]["angle"]).all():
            raise RuntimeError(
                f"non-finite step for a {tuple(p.shape)} weight: "
                f"lam_in={lams[0]:.3e} lam_out={lams[1]:.3e}. The damping is "
                "too small to hold the raw (unfloored) spectra, or "
                "eps_numeric is too small for this backward covariance."
            )
