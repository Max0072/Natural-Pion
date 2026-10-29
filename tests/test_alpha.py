"""What `alpha = min(1, quad/curv)` is, as algebra (journal 2026-10-29).

`X = F_0^-1 G` is built from the basis of the last refactorisation, `F_0`;
`quad = <G, X>`, `curv = <X, F_t X>` with the current statistics `F_t`. Four
claims, each pinned here on the method's own functions against a dense route
that shares no code with them:

1. on a fresh basis (`F_t = F_0`) `alpha == 1` identically, floor or not;
2. `quad/curv` is the exact minimiser of the quadratic model
   `q(s) = -s <G,X> + s^2/2 <X, F_t X>` along `X`;
3. exactly, not to first order, in the eigenbasis of `F_0` (`lam_i`, `g_i`)
   with `Delta = F_t - F_0`:
       1/alpha - 1 = sum_ij g_i g_j Delta_ij / (lam_i lam_j) / sum_i g_i^2/lam_i
   so the error of the preconditioner is weighted by `1/lam^2`;
4. if the eigenbasis turns at fixed spectrum (`B, C -> R B R^T, R C R^T`), the
   expected `curv` over isotropic `G` can only rise: `tr(F_0^-2 F_t) >=
   tr(F_0^-1) = E quad`, by von Neumann's trace inequality (the aligned
   pairing of `F_0^-2` with `F_0` is the anti-sorted one, hence the minimum).
"""

import numpy as np
import pytest
import torch

from ngd_pion.direction import fisher_apply, natural_gradient, trust_region_alpha
from ngd_pion.factorization import basis_congruence
from ngd_pion.linalg import skew
from ngd_pion.reference import fisher_matrix

DT = torch.float64
D = 10


def spd(n, cond=1e3, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    return (U * torch.logspace(0, -np.log10(cond), n, dtype=DT)) @ U.T


def rand_skew(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return skew(torch.randn(n, n, generator=g, dtype=DT))


def rotation(n, angle, seed=0):
    return torch.linalg.matrix_exp(angle * rand_skew(n, seed))


def skew_frame(n):
    """Orthonormal basis of vec(skew), as columns: the space F acts on."""
    cols = []
    for i in range(n):
        for j in range(i + 1, n):
            E = np.zeros((n, n))
            E[i, j], E[j, i] = 2 ** -0.5, -(2 ** -0.5)
            cols.append(E.ravel())
    return np.stack(cols, axis=1)


def quad_curv(G, B0, C0, B1, C1, eps=0.0):
    X = natural_gradient(G, basis_congruence(B0, C0, eps))
    return (G * X).sum(), (X * fisher_apply(B1, C1, X)).sum(), X


@pytest.mark.parametrize("eps", [0.0, 1e-3])
def test_fresh_basis_gives_alpha_one(eps):
    B, C = spd(D, seed=1), spd(D, seed=2)
    for s in range(5):
        quad, curv, _ = quad_curv(rand_skew(D, 10 + s), B, C, B, C, eps)
        alpha = trust_region_alpha(quad, curv, 1.0)
        if eps == 0.0:
            assert abs(float(quad / curv) - 1.0) < 1e-9
        else:
            # The floor only raises the denominators, so curv <= quad and the
            # clamp returns exactly 1.
            assert float(curv) <= float(quad) * (1 + 1e-12)
        assert float(alpha) == 1.0


def test_alpha_is_the_exact_line_search_in_the_current_metric():
    B0, C0 = spd(D, seed=1), spd(D, seed=2)
    R = rotation(D, 0.3, seed=3)
    B1, C1 = R @ B0 @ R.T, 1.3 * (R @ C0 @ R.T)
    G = rand_skew(D, 4)
    quad, curv, X = quad_curv(G, B0, C0, B1, C1)
    s_star = float(quad / curv)

    def q(s):
        Y = s * X
        return float(-(G * Y).sum() + 0.5 * (Y * fisher_apply(B1, C1, Y)).sum())

    grid = np.linspace(0.0, 3.0 * s_star, 3001)
    assert abs(grid[np.argmin([q(s) for s in grid])] - s_star) <= grid[1]
    assert all(q(s_star) <= q(s) + 1e-14 for s in grid)


def test_alpha_reads_the_preconditioner_error_weighted_by_inverse_lam_squared():
    B0, C0 = spd(D, seed=1), spd(D, seed=2)
    R = rotation(D, 0.2, seed=5)
    B1, C1 = R @ B0 @ R.T + 0.1 * spd(D, seed=6), C0
    G = rand_skew(D, 7)
    quad, curv, _ = quad_curv(G, B0, C0, B1, C1)

    Q = skew_frame(D)
    F0 = Q.T @ fisher_matrix(B0.numpy(), C0.numpy()) @ Q
    F1 = Q.T @ fisher_matrix(B1.numpy(), C1.numpy()) @ Q
    lam, V = np.linalg.eigh(F0)
    g = V.T @ (Q.T @ G.numpy().ravel())
    Delta = V.T @ (F1 - F0) @ V
    u = g / lam
    predicted = (u @ Delta @ u) / (g ** 2 / lam).sum()
    assert abs((float(curv / quad) - 1.0) - predicted) < 1e-9 * max(1.0, abs(predicted))


@pytest.mark.parametrize("angle", [0.05, 0.2, 0.8])
def test_turning_the_basis_at_fixed_spectrum_only_raises_expected_curv(angle):
    B0, C0 = spd(D, seed=1), spd(D, seed=2)
    Q = skew_frame(D)
    F0 = Q.T @ fisher_matrix(B0.numpy(), C0.numpy()) @ Q
    F0inv = np.linalg.inv(F0)
    e_quad = np.trace(F0inv)
    for s in range(20):
        R = rotation(D, angle, seed=100 + s)
        F1 = Q.T @ fisher_matrix((R @ B0 @ R.T).numpy(), (R @ C0 @ R.T).numpy()) @ Q
        # Same spectrum: a conjugation by the orthogonal R (x) R on vec(skew).
        assert np.allclose(np.linalg.eigvalsh(F1), np.linalg.eigvalsh(F0), rtol=1e-8)
        assert np.trace(F0inv @ F0inv @ F1) >= e_quad * (1 - 1e-12)
