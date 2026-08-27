"""What `lam_tikhonov` inverts, checked against a dense solve.

The claim these tests exist to pin down is narrow and easy to get wrong:
dividing by `d + lam` in the factorised basis is *exact* Tikhonov, but in the
metric of the congruence anchor rather than the Euclidean one. Getting that
backwards is not a small error -- at the scale mismatch this model runs at,
treating it as Euclidean damping is wrong by three to four orders.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from ngd_pion.damped import _basis_raw_anchor, _basis_raw_congruence
from ngd_pion.direction import skew

N = 6
DT = torch.float64


def _as_matrix(op):
    m = torch.zeros(N * N, N * N, dtype=DT)
    for k in range(N * N):
        e = torch.zeros(N, N, dtype=DT)
        e.reshape(-1)[k] = 1.0
        m[:, k] = op(e).reshape(-1)
    return m


def _rand_spd(scale: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(N, N, dtype=DT, generator=g)
    return scale * (m @ m.T + N * torch.eye(N, dtype=DT))


def _solve_with(basis, G, lam):
    damped = dataclasses.replace(basis, lam_tikhonov=lam)
    P = damped.P
    return skew(P @ ((P.T @ G @ P) / damped.denominator) @ P.T)


def _skew_rhs(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(N, N, dtype=DT, generator=g)
    return m - m.T


@pytest.mark.parametrize("scale_b,scale_c", [(1.0, 1.0), (0.3, 2.0), (1e-3, 1e2)])
@pytest.mark.parametrize("frac", [1e-2, 1.0, 10.0])
def test_damping_is_exact_in_the_anchor_metric(scale_b, scale_c, frac):
    B = _rand_spd(scale_b, seed=1)
    C = _rand_spd(scale_c, seed=2)
    basis = _basis_raw_congruence(B, C, eps_numeric=1e-12)
    G = _skew_rhs(seed=3)

    lam = frac * float(basis.denominator.max())
    F = _as_matrix(lambda X: 2 * (B @ X @ C + C @ X @ B))
    metric = _as_matrix(lambda X: B @ X @ B)
    want = torch.linalg.solve(F + lam * metric, G.reshape(-1)).reshape(N, N)

    got = _solve_with(basis, G, lam)
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("frac", [1e-2, 1.0, 10.0])
def test_orthogonal_basis_reduces_to_euclidean_tikhonov(frac):
    """`P P^T = I` there, so the anchor metric and the identity coincide."""
    C = _rand_spd(1.0, seed=4)
    basis = _basis_raw_anchor(C)
    assert basis.orthogonal
    G = _skew_rhs(seed=5)

    lam = frac * float(basis.denominator.max())
    F = _as_matrix(lambda X: 2 * (X @ C + C @ X))
    want = torch.linalg.solve(
        F + lam * torch.eye(N * N, dtype=DT), G.reshape(-1)
    ).reshape(N, N)

    got = _solve_with(basis, G, lam)
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-11)


def test_euclidean_damping_would_be_wrong_on_a_congruence():
    """The negative control, so the test above is not vacuous.

    If someone later 'simplifies' the anchor metric away, this is the test that
    should fail. The mismatch is not marginal: at the 1e5 spread between the
    factors that this model actually runs at, it is thousands of times off.
    """
    B = _rand_spd(1e-3, seed=6)
    C = _rand_spd(1e2, seed=7)
    basis = _basis_raw_congruence(B, C, eps_numeric=1e-12)
    G = _skew_rhs(seed=8)

    lam = float(basis.denominator.max())
    F = _as_matrix(lambda X: 2 * (B @ X @ C + C @ X @ B))
    euclid = torch.linalg.solve(
        F + lam * torch.eye(N * N, dtype=DT), G.reshape(-1)
    ).reshape(N, N)

    got = _solve_with(basis, G, lam)
    assert (got - euclid).norm() / euclid.norm() > 100.0


def test_damping_bounds_the_step_where_a_relative_floor_cannot():
    """A uniformly flat layer: scale the whole spectrum down, keep the shape.

    A relative floor is invariant to this by construction, so the step it
    permits grows without bound. The additive term is not.
    """
    C = _rand_spd(1.0, seed=9)
    G = _skew_rhs(seed=10)

    undamped, damped = [], []
    for shrink in (1.0, 1e-3, 1e-6):
        basis = _basis_raw_anchor(C * shrink)
        undamped.append(float(_solve_with(basis, G, 0.0).norm()))
        damped.append(float(_solve_with(basis, G, 1.0).norm()))

    # undamped: shrinking the curvature by 1e6 lengthens the step by 1e6
    assert undamped[-1] / undamped[0] > 1e5
    # damped: the step still rises, because at full curvature `d >> lam` and
    # the Newton step is short, while at `d << lam` it saturates. What it does
    # not do is track the curvature down: it stops at the ceiling `||G||/lam`.
    assert max(damped) <= float(G.norm()) * 1.01
    assert damped[-1] / damped[0] < 1e2
    assert (undamped[-1] / undamped[0]) / (damped[-1] / damped[0]) > 1e3
