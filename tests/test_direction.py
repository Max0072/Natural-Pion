"""§1, §5, §6: generators, the solve, and the trust-region ratio."""

import numpy as np
import pytest
import torch

from ngd_pion.direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from ngd_pion.factorization import basis_congruence, basis_identity_anchor, build_bases
from ngd_pion.linalg import skew
from ngd_pion.reference import fisher_matrix
from ngd_pion.reference import generators as ref_generators
from ngd_pion.reference import natural_gradient as ref_natural_gradient

DT = torch.float64
SHAPES = [(16, 16), (24, 12), (12, 24)]


def spd(n, cond=1e3, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    return (U * torch.logspace(0, -np.log10(cond), n, dtype=DT)) @ U.T


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DT)


def semi_ortho(d_out, d_in, seed=0):
    Q = torch.linalg.qr(rand(max(d_out, d_in), min(d_out, d_in), seed=seed))[0]
    return Q if d_out >= d_in else Q.T


@pytest.mark.parametrize("shape", SHAPES)
def test_generators_match_reference(shape):
    W, G = rand(*shape, seed=1), rand(*shape, seed=2)
    gi, go = generators(W, G)
    ri, ro = ref_generators(W.numpy(), G.numpy())
    assert np.abs(gi.numpy() - ri).max() < 1e-12
    assert np.abs(go.numpy() - ro).max() < 1e-12


@pytest.mark.parametrize("shape", SHAPES)
def test_generators_are_skew(shape):
    gi, go = generators(rand(*shape, seed=3), rand(*shape, seed=4))
    assert (gi + gi.T).abs().max() < 1e-12
    assert (go + go.T).abs().max() < 1e-12


@pytest.mark.parametrize("shape", SHAPES)
def test_descent_lemma(shape):
    """`<G, W X> = <G_in, X>` -- the identity the sign convention rests on.

    The half disappeared with the 2026-08-28 generator convention, and with it
    the half in `pred_drop`. That is not cosmetic: `pred_drop` feeds the
    reduction ratio `rho`, so leaving it would have made every `rho` on record
    after the change wrong by a factor of two.
    """
    d_out, d_in = shape
    W, G = rand(*shape, seed=5), rand(*shape, seed=6)
    X_in = skew(rand(d_in, d_in, seed=7))
    X_out = skew(rand(d_out, d_out, seed=8))
    G_in, G_out = generators(W, G)
    assert torch.allclose((G * (W @ X_in)).sum(), (G_in * X_in).sum())
    assert torch.allclose((G * (X_out @ W)).sum(), (G_out * X_out).sum())


def test_solve_inverts_the_fisher_operator():
    """The point of the whole construction: `F(natural_gradient(G)) == G`."""
    B, C = spd(10, seed=9), spd(10, seed=10)
    G = skew(rand(10, 10, seed=11))
    X = natural_gradient(G, basis_congruence(B, C, eps=0.0))
    assert (fisher_apply(B, C, X) - G).abs().max() / G.abs().max() < 1e-9


def test_solve_matches_an_explicit_kronecker_system():
    """An independent route: build `F` as a d^2 x d^2 matrix and solve it."""
    d = 8
    B, C = spd(d, seed=12), spd(d, seed=13)
    G = skew(rand(d, d, seed=14))
    X = natural_gradient(G, basis_congruence(B, C, eps=0.0)).numpy()
    ref = np.linalg.solve(fisher_matrix(B.numpy(), C.numpy()), G.numpy().reshape(-1))
    assert np.abs(X - ref.reshape(d, d)).max() / np.abs(X).max() < 1e-9


def test_solve_matches_numpy_reference():
    C = spd(9, cond=1e5, seed=15)
    G = skew(rand(9, 9, seed=16))
    b = basis_identity_anchor(C, eps=1e-8)
    got = natural_gradient(G, b).numpy()
    want = ref_natural_gradient(G.numpy(), b.P.numpy(), b.lam.numpy())
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-10


def test_solve_result_is_skew():
    b = basis_identity_anchor(spd(7, seed=17), eps=1e-8)
    X = natural_gradient(skew(rand(7, 7, seed=18)), b)
    assert (X + X.T).abs().max() < 1e-14


def test_solve_is_invariant_to_eigenvector_sign_and_degenerate_rotation():
    """`eigh` fixes neither; the step must not depend on what it happens to return."""
    d = 20
    g = torch.Generator().manual_seed(19)
    U = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))[0]
    w = torch.rand(d, generator=g, dtype=DT) + 0.1
    w[5:8] = 0.5  # a degenerate eigenspace of multiplicity 3
    C = (U * w) @ U.T
    G = skew(rand(d, d, seed=20))
    b = basis_identity_anchor(C, eps=0.0)
    base = natural_gradient(G, b)

    signs = torch.where(torch.rand(d, generator=g) > 0.5, 1.0, -1.0).to(DT)
    flipped = natural_gradient(G, type(b)(P=b.P * signs, lam=b.lam, orthogonal=True))
    assert (flipped - base).abs().max() == 0.0

    deg = torch.where((b.lam - 0.5).abs() < 1e-9)[0]
    R = torch.linalg.qr(torch.randn(len(deg), len(deg), generator=g, dtype=DT))[0]
    P2 = b.P.clone()
    P2[:, deg] = b.P[:, deg] @ R
    rotated = natural_gradient(G, type(b)(P=P2, lam=b.lam, orthogonal=True))
    assert (rotated - base).abs().max() / base.abs().max() < 1e-12


def _quad_curv(W, A, G, eps):
    d_out = W.shape[0]
    b_in, b_out = build_bases(W, A, eps=eps)
    G_in, G_out = generators(W, G)
    X_in = natural_gradient(G_in, b_in)
    X_out = natural_gradient(G_out, b_out)
    quad = (G_in * X_in).sum() + (G_out * X_out).sum()
    curv = (X_in * fisher_apply(A, W.T @ W, X_in)).sum() + (
        X_out * fisher_apply(torch.eye(d_out, dtype=DT), W @ A @ W.T, X_out)
    ).sum()
    return quad, curv


def test_alpha_is_exactly_one_for_square_w_at_zero_floor():
    """With no kernel and no floor, `curv == quad` is an identity, not an estimate."""
    W = semi_ortho(16, 16, seed=21)
    quad, curv = _quad_curv(W, spd(16, seed=22), rand(16, 16, seed=23), eps=0.0)
    assert abs(float(quad / curv) - 1.0) < 1e-9


@pytest.mark.parametrize("shape", [(24, 12), (12, 24)])
def test_alpha_is_one_up_to_the_floor_for_non_square_w(shape):
    """A non-square `W` needs a positive floor -- `eps = 0` is a true 0/0 there."""
    d_out, d_in = shape
    W = semi_ortho(d_out, d_in, seed=21)
    A = spd(d_in, seed=22)
    G = rand(*shape, seed=23)
    assert torch.isnan(_quad_curv(W, A, G, eps=0.0)[0])
    quad, curv = _quad_curv(W, A, G, eps=1e-8)
    assert abs(float(quad / curv) - 1.0) < 1e-4


def test_alpha_is_capped_and_one_sided():
    q = torch.tensor(5.0, dtype=DT)
    c = torch.tensor(1.0, dtype=DT)
    assert float(trust_region_alpha(q, c, alpha_max=1.0)) == 1.0
    assert float(trust_region_alpha(torch.tensor(0.4, dtype=DT), c, 1.0)) == pytest.approx(0.4)


@pytest.mark.parametrize("shape", SHAPES)
def test_dead_block_carries_no_gradient(shape):
    """A non-square `W` gives `0/0`, never a live numerator over a zero."""
    d_out, d_in = shape
    if d_out == d_in:
        pytest.skip("square W has no kernel")
    eps = 1e-12
    W = semi_ortho(d_out, d_in, seed=24)
    A = spd(d_in, seed=25)
    x = rand(d_in, seed=26)
    delta = rand(d_out, seed=27)
    G_in, G_out = generators(W, torch.outer(delta, x))
    for G, b in zip((G_in, G_out), build_bases(W, A, eps=eps)):
        dead = b.lam <= b.lam.max() * eps * 1.001
        if int(dead.sum()) < 2:
            continue
        Gt = b.P.T @ G @ b.P
        block = Gt[dead][:, dead]
        assert block.abs().max() < 1e-8 * Gt.abs().max()


# --- the 2026-08-28 convention change ---------------------------------------


def test_generators_are_exactly_half_the_old_convention():
    """`skew(W^T G)` rather than `W^T G - G^T W`.

    Halving is exact in binary, so this is `torch.equal` and not a tolerance.
    """
    import torch

    from ngd_pion.direction import generators

    g = torch.Generator().manual_seed(0)
    W = torch.randn(6, 4, dtype=torch.float64, generator=g)
    G = torch.randn(6, 4, dtype=torch.float64, generator=g)
    G_in, G_out = generators(W, G)
    assert torch.equal(G_in, 0.5 * (W.T @ G - G.T @ W))
    assert torch.equal(G_out, 0.5 * (G @ W.T - W @ G.T))


def test_the_generator_is_now_the_riemannian_gradient():
    """`<G, W X> = <G_in, X>` for skew `X`, without the factor of two.

    This is the reason for the change: the quantity `eta` multiplies is now the
    same one their paper's learning rate multiplies.
    """
    import torch

    from ngd_pion.direction import generators

    g = torch.Generator().manual_seed(1)
    W = torch.randn(6, 4, dtype=torch.float64, generator=g)
    G = torch.randn(6, 4, dtype=torch.float64, generator=g)
    Z = torch.randn(4, 4, dtype=torch.float64, generator=g)
    X = Z - Z.T
    G_in, _ = generators(W, G)
    assert torch.allclose((G * (W @ X)).sum(), (G_in * X).sum(), atol=1e-12)


def test_doubling_eta_reproduces_the_old_trajectory():
    """The linearity that makes the eta-doubling exact rather than approximate.

    `natural_gradient` is linear in `G` and `alpha = quad/curv` carries the
    factor in both halves, so a run at twice the gradient and rate `c` is the
    same as one at the gradient and rate `2c` -- which is precisely the relation
    between an `eta` recorded before 2026-08-28 and one recorded after.
    """
    import torch

    from ngd_pion.unified import NGDPionUnified

    DT = torch.float64
    g = torch.Generator().manual_seed(2)
    W = torch.randn(6, 4, dtype=DT, generator=g)
    x = torch.randn(64, 4, dtype=DT, generator=g)
    d = torch.randn(64, 6, dtype=DT, generator=g) * 0.3
    grads = [torch.randn(6, 4, dtype=DT, generator=g) * 0.1 for _ in range(6)]

    def run(scale, lr):
        p = torch.nn.Parameter(W.clone())
        opt = NGDPionUnified([p], lr=lr, t_fac=3, compute_dtype=DT)
        opt.observe(p, x)
        opt.observe_backward(p, d)
        for G in grads:
            p.grad = (scale * G).clone()
            opt.step()
        return p.detach().clone()

    assert torch.allclose(run(2.0, 0.02), run(1.0, 0.04), rtol=1e-10, atol=1e-12)


def test_pion_with_rms_is_untouched_by_the_factor():
    """Its update is normalised to a target RMS, so a constant on the generator
    is absorbed. This is why the anchor did not have to be re-run."""
    import torch

    from ngd_pion.pion_baseline import Pion

    DT = torch.float64
    g = torch.Generator().manual_seed(3)
    W = torch.randn(6, 4, dtype=DT, generator=g)
    grads = [torch.randn(6, 4, dtype=DT, generator=g) * 0.1 for _ in range(5)]

    def run(scale):
        p = torch.nn.Parameter(W.clone())
        opt = Pion([p], lr=0.01, scaling="rms", momentum="none")
        for G in grads:
            p.grad = (scale * G).clone()
            opt.step()
        return p.detach().clone()

    assert torch.allclose(run(1.0), run(2.0), rtol=1e-9, atol=1e-11)


def test_shampoo_is_untouched_by_the_factor():
    """`P` goes as `1/4`, `P^-1/4` as `sqrt(2)`, and the sandwich returns the
    same step -- so its `eta` means what it meant before the change."""
    import torch

    from ngd_pion.shampoo import ShampooPion

    DT = torch.float64
    g = torch.Generator().manual_seed(4)
    W = torch.randn(6, 4, dtype=DT, generator=g)
    grads = [torch.randn(6, 4, dtype=DT, generator=g) * 0.1 for _ in range(5)]

    def run(scale):
        p = torch.nn.Parameter(W.clone())
        opt = ShampooPion([p], lr=0.05, eps=1e-10, compute_dtype=DT)
        for G in grads:
            p.grad = (scale * G).clone()
            opt.step()
        return p.detach().clone()

    assert torch.allclose(run(1.0), run(2.0), rtol=1e-8, atol=1e-10)
