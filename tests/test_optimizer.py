"""The optimizer end to end, with the numpy reference as the oracle."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ngd_pion.hooks import attach, attached
from ngd_pion.optimizer import NGDPion
from ngd_pion.reference import NGDPionReference

DT = torch.float64
SHAPES = [(16, 16), (24, 12), (12, 24)]


def spd(n, cond=1e3, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    return (U * torch.logspace(0, -np.log10(cond), n, dtype=DT)) @ U.T


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DT)


@pytest.mark.parametrize("shape", SHAPES)
def test_one_step_matches_the_numpy_reference(shape):
    """The oracle test: torch must reproduce reference.py, not merely resemble it."""
    d_out, d_in = shape
    W0, G, A = rand(*shape, seed=1) * 0.1, rand(*shape, seed=2), spd(d_in, seed=3)
    lr, eps, alpha_max = 1e-2, 1e-4, 1.0

    p = nn.Parameter(W0.clone())
    p.grad = G.clone()
    opt = NGDPion([p], lr=lr, eps=eps, alpha_max=alpha_max, compute_dtype=DT)
    opt.set_covariance(p, A)
    opt.step()

    ref = NGDPionReference(d_out, d_in, eps=eps, eta=lr, alpha_max=alpha_max)
    ref.A = A.numpy()
    ref.refactor(W0.numpy())
    want = ref.step(W0.numpy(), G.numpy())

    # Not tighter than this for non-square W: the two implementations use
    # different LAPACK paths, and the congruent basis amplifies the resulting
    # 1e-16 rounding. The spectra themselves agree to 1e-14.
    err = np.abs(p.detach().numpy() - want).max() / np.abs(want).max()
    assert err < 1e-10, f"torch and reference disagree by {err:.2e}"


@pytest.mark.parametrize("shape", SHAPES)
def test_step_preserves_singular_values(shape):
    """The property the whole method exists to keep."""
    p = nn.Parameter(rand(*shape, seed=4) * 0.1)
    s0 = torch.linalg.svdvals(p.detach())
    opt = NGDPion([p], lr=0.5, compute_dtype=DT)
    opt.set_covariance(p, spd(shape[1], seed=5))
    for _ in range(20):
        p.grad = rand(*shape, seed=6)
        opt.step()
    assert (torch.linalg.svdvals(p.detach()) - s0).abs().max() < 1e-10


def test_step_descends_on_a_reachable_target():
    """Sanity that the sign is right and the thing actually optimises."""
    from ngd_pion.linalg import cayley, skew

    d_out, d_in = 12, 12
    A = spd(d_in, seed=7)
    W0 = torch.linalg.qr(rand(d_out, d_in, seed=8))[0]
    target = cayley(skew(rand(d_out, d_out, seed=9)), 0.5) @ W0 @ cayley(
        skew(rand(d_in, d_in, seed=10)), 0.5
    )
    loss = lambda W: 0.5 * torch.trace((W - target) @ A @ (W - target).T)

    p = nn.Parameter(W0.clone())
    opt = NGDPion([p], lr=0.05, compute_dtype=DT, t_fac=1)
    opt.set_covariance(p, A)
    before = loss(p.detach()).item()
    for _ in range(150):
        p.grad = (p.detach() - target) @ A
        opt.step()
    assert loss(p.detach()).item() < before / 100


def test_float32_tracks_float64():
    """fp32 is the default because it is enough, which is a claim worth pinning."""
    shape = (16, 24)
    W0, G, A = rand(*shape, seed=11) * 0.1, rand(*shape, seed=12), spd(shape[1], cond=1e4, seed=13)
    out = {}
    for dt in (DT, torch.float32):
        p = nn.Parameter(W0.clone())
        p.grad = G.clone()
        opt = NGDPion([p], lr=1e-2, compute_dtype=dt)  # default eps, tuned for fp32
        opt.set_covariance(p, A)
        opt.step()
        out[dt] = p.detach().double()
    rel = (out[torch.float32] - out[DT]).abs().max() / out[DT].abs().max()
    assert rel < 1e-3, f"fp32 drifted from fp64 by {rel:.2e}"


def test_refactor_schedule_is_fixed_and_honoured():
    """A fixed schedule is what makes a run reproducible across hardware."""
    p = nn.Parameter(rand(10, 10, seed=14) * 0.1)
    opt = NGDPion([p], lr=1e-3, t_fac=5, compute_dtype=DT)
    opt.set_covariance(p, spd(10, seed=15))
    seen = []
    for _ in range(11):
        p.grad = rand(10, 10, seed=16)
        opt.step()
        seen.append(opt.state[p]["since_refactor"])
    assert seen == [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


def test_equal_shapes_are_factorised_together():
    """Batching is a performance claim; check it produces identical bases."""
    A = spd(12, seed=17)
    ps = [nn.Parameter(rand(12, 12, seed=18 + i) * 0.1) for i in range(3)]
    opt = NGDPion(ps, lr=1e-3, compute_dtype=DT)
    for p in ps:
        opt.set_covariance(p, A)
        p.grad = rand(12, 12, seed=30)
    opt.step()
    solo = nn.Parameter(ps[0].detach().clone())
    assert all("bases" in opt.state[p] for p in ps)


def test_non_2d_parameters_are_refused():
    with pytest.raises(ValueError, match="2-D"):
        NGDPion([nn.Parameter(torch.zeros(5))], lr=1e-3)


def test_missing_activations_raise_rather_than_silently_skipping():
    p = nn.Parameter(rand(6, 6, seed=19))
    p.grad = rand(6, 6, seed=20)
    with pytest.raises(RuntimeError, match="no activations"):
        NGDPion([p], lr=1e-3).step()


def test_invalid_arguments_are_refused():
    p = nn.Parameter(rand(4, 4, seed=21))
    with pytest.raises(ValueError, match="lr"):
        NGDPion([p], lr=0.0)
    with pytest.raises(ValueError, match="t_fac"):
        NGDPion([p], lr=1e-3, t_fac=0)


def test_alternate_moves_one_side_per_step():
    """Pion's default: odd steps rotate in-side, even steps out-side."""
    p = nn.Parameter(rand(8, 8, seed=22) * 0.1)
    opt = NGDPion([p], lr=0.1, alternate=True, compute_dtype=DT)
    opt.set_covariance(p, spd(8, seed=23))
    p.grad = rand(8, 8, seed=24)
    before = p.detach().clone()
    opt.step()
    left_only = torch.linalg.lstsq(before, p.detach()).solution
    assert not torch.allclose(before, p.detach())


def test_hooks_feed_the_optimizer():
    layer = nn.Linear(9, 7, bias=False).double()
    opt = NGDPion([layer.weight], lr=1e-3, compute_dtype=DT)
    with attached([layer], opt):
        layer(torch.randn(32, 9, dtype=DT)).sum().backward()
    assert opt.state[layer.weight]["cov"].ready
    opt.step()


def test_hooks_reject_non_linear_modules():
    opt = NGDPion([nn.Parameter(rand(4, 4, seed=25))], lr=1e-3)
    with pytest.raises(TypeError, match="nn.Linear"):
        attach([nn.LayerNorm(4)], opt)


@pytest.mark.parametrize("eps,limit", [(1e-8, 5e-1), (1e-6, 1e-2), (1e-4, 1e-3)])
def test_the_floor_has_a_lower_bound_set_by_the_working_precision(eps, limit):
    """`eps` below the compute dtype's machine epsilon is meaningless.

    Pinned because it is the one hyperparameter choice that looks free and is
    not: the plateau measured in fp64 does not transfer to fp32, whose machine
    epsilon is 1.2e-7. The default exists to sit above it.
    """
    shape = (16, 24)
    W0, G = rand(*shape, seed=11) * 0.1, rand(*shape, seed=12)
    A = spd(shape[1], cond=1e4, seed=13)
    out = {}
    for dt in (DT, torch.float32):
        p = nn.Parameter(W0.clone())
        p.grad = G.clone()
        opt = NGDPion([p], lr=1e-2, eps=eps, compute_dtype=dt)
        opt.set_covariance(p, A)
        opt.step()
        out[dt] = p.detach().double()
    rel = float((out[torch.float32] - out[DT]).abs().max() / out[DT].abs().max())
    assert rel < limit
