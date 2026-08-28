"""`FastNGDPion` against the reference implementation it is allowed to replace.

`optimizer.py` is the reference and stays unoptimised on purpose. This file
exists to enforce the contract that lets `fast.py` be used in its place: every
difference is either exactly equivalent or confined to a diagnostic, and
nothing in it moves the weights.

The equivalence test compares *trajectories* rather than final weights. A
single step agreeing proves less than it appears to -- the optimizer carries
state (bases, the covariance, the refactor counter, and now the cached power
vectors), and a divergence introduced on step 3 can be invisible at step 1 and
fatal by step 300.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ngd_pion.fast import FastNGDPion
from ngd_pion.linalg import skew, spectral_norm
from ngd_pion.optimizer import NGDPion

DT = torch.float64
SHAPES = [(16, 16), (24, 12), (12, 24)]


def spd(n, cond=1e3, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    return (U * torch.logspace(0, -np.log10(cond), n, dtype=DT)) @ U.T


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DT)


def rand_skew(n, seed=0):
    return skew(rand(n, n, seed=seed))


# --- the contract ---------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("alternate", [False, True])
def test_weight_trajectory_is_identical_to_the_reference(shape, alternate):
    """The whole licence for using `fast.py`: it must not move the weights.

    Run long enough to cross a refactorisation (`t_fac=5` here), because that
    is where `fast.py` does something the reference does not -- it rebuilds the
    power-iteration vector -- and where a mistake would first show.
    """
    d_out, d_in = shape
    W0, A = rand(*shape, seed=1) * 0.1, spd(d_in, seed=3)
    kw = dict(lr=1e-2, eps=1e-4, t_fac=5, alternate=alternate, compute_dtype=DT)

    trajectories = []
    for cls in (NGDPion, FastNGDPion):
        p = nn.Parameter(W0.clone())
        opt = cls([p], **kw)
        opt.set_covariance(p, A.clone())
        seen = []
        for step in range(17):
            p.grad = rand(*shape, seed=100 + step)
            opt.step()
            seen.append(p.detach().clone())
        trajectories.append(seen)

    for step, (a, b) in enumerate(zip(*trajectories)):
        assert torch.equal(a, b), f"weights diverged at step {step}"


@pytest.mark.parametrize("shape", SHAPES)
def test_alpha_is_identical_too(shape):
    """`alpha` does reach the step, so unlike `angle` it must match exactly."""
    d_out, d_in = shape
    W0, A = rand(*shape, seed=1) * 0.1, spd(d_in, seed=3)
    got = []
    for cls in (NGDPion, FastNGDPion):
        p = nn.Parameter(W0.clone())
        opt = cls([p], lr=1e-2, t_fac=5, compute_dtype=DT)
        opt.set_covariance(p, A.clone())
        seen = []
        for step in range(11):
            p.grad = rand(*shape, seed=200 + step)
            opt.step()
            seen.append(opt.state[p]["alpha"])
        got.append(seen)
    assert got[0] == got[1]


@pytest.mark.parametrize("shape", SHAPES)
def test_angle_agrees_with_the_exact_norm_it_replaces(shape):
    """The diagnostic may be approximate, but not misleading.

    Warm, the power estimate should be a few parts in `1e-4` of the exact
    spectral norm. It is a Rayleigh quotient, so it can only err low; the
    assertion is one-sided for that reason and the tolerance is on the gap.
    """
    d_out, d_in = shape
    W0, A = rand(*shape, seed=1) * 0.1, spd(d_in, seed=3)
    p = nn.Parameter(W0.clone())
    opt = FastNGDPion([p], lr=1e-2, t_fac=100, compute_dtype=DT)
    opt.set_covariance(p, A.clone())

    for step in range(12):
        p.grad = rand(*shape, seed=300 + step)
        opt.step()

    state = opt.state[p]
    # rebuild the exact quantity from the same state the last step used
    from ngd_pion.direction import generators, natural_gradient

    W = p.detach()
    G = p.grad
    basis_in, basis_out = state["bases"]
    G_in, G_out = generators(W, G)
    X_in = natural_gradient(G_in, basis_in)
    X_out = natural_gradient(G_out, basis_out)
    exact = torch.maximum(
        torch.linalg.matrix_norm(X_in, 2), torch.linalg.matrix_norm(X_out, 2)
    )
    got_in, _ = spectral_norm(X_in, 2, state["angle_v_in"])
    got_out, _ = spectral_norm(X_out, 2, state["angle_v_out"])
    got = torch.maximum(got_in, got_out)

    # The structural half, which is exact and not a tolerance.
    assert got <= exact * (1 + 1e-12), "a Rayleigh quotient cannot exceed the norm"

    # The accuracy half. A skew matrix has its singular values in exact pairs and
    # the large ones bunch together, so the ratio governing power-iteration
    # convergence sits near 1 and the warm estimate's error depends on the gap
    # between the top two *pairs* -- a property of wherever the trajectory
    # happened to land, not of the implementation. A tight threshold here is
    # fitted to one trajectory and breaks the moment the trajectory moves, which
    # is what it did when the generator convention changed on 2026-08-28. So the
    # loose bound is documented as loose, and the implementation is checked by
    # convergence instead: more iterations must close the gap.
    assert float((exact - got) / exact) < 5e-3, "warm estimate is not even close"
    converged, _ = spectral_norm(X_in, 200, state["angle_v_in"])
    exact_in = torch.linalg.matrix_norm(X_in, 2)
    assert float((exact_in - converged) / exact_in) < 1e-6, "iteration does not converge"


def test_cached_vectors_survive_a_reload():
    """State carries tensors now; a resumed run must not silently cold-start."""
    shape = (16, 16)
    p = nn.Parameter(rand(*shape, seed=1) * 0.1)
    opt = FastNGDPion([p], lr=1e-2, compute_dtype=DT)
    opt.set_covariance(p, spd(shape[1], seed=3))
    for step in range(3):
        p.grad = rand(*shape, seed=400 + step)
        opt.step()
    v_before = opt.state[p]["angle_v_in"].clone()

    q = nn.Parameter(p.detach().clone())
    fresh = FastNGDPion([q], lr=1e-2, compute_dtype=DT)
    fresh.load_state_dict(opt.state_dict())
    assert torch.allclose(fresh.state[q]["angle_v_in"], v_before)


# --- the primitive --------------------------------------------------------


def test_spectral_norm_converges_from_a_warm_vector():
    """Cold is bad and warm is good; both are load-bearing, so both are pinned."""
    X = rand_skew(64, seed=11)
    exact = torch.linalg.matrix_norm(X, 2)

    cold, _ = spectral_norm(X, 1)
    assert float((exact - cold) / exact) > 1e-2, "cold start is supposed to be poor"

    _, v = spectral_norm(X, 200)
    warm, _ = spectral_norm(X, 1, v)
    assert float((exact - warm) / exact) < 1e-6


def test_spectral_norm_never_overestimates():
    """A Rayleigh quotient is a lower bound. The diagnostic relies on knowing which way."""
    for seed in range(5):
        X = rand_skew(32, seed=seed)
        exact = torch.linalg.matrix_norm(X, 2)
        for iters in (0, 1, 5, 50):
            got, _ = spectral_norm(X, iters)
            assert got <= exact * (1 + 1e-12)


def test_spectral_norm_batches():
    X = torch.stack([rand_skew(16, seed=s) for s in range(4)])
    got, v = spectral_norm(X, 20)
    assert got.shape == (4,) and v.shape == (4, 16)
    for i in range(4):
        one, _ = spectral_norm(X[i], 20)
        assert torch.allclose(got[i], one)


def test_spectral_norm_handles_a_zero_matrix():
    """The floor makes denominators safe; this makes the diagnostic safe too."""
    got, v = spectral_norm(torch.zeros(8, 8, dtype=DT), 5)
    assert float(got) == 0.0 and torch.isfinite(v).all()


def test_spectral_norm_rejects_negative_iterations():
    with pytest.raises(ValueError):
        spectral_norm(rand_skew(8), -1)


def test_fast_s_is_bit_identical_to_its_reference():
    """`FastNGDPionS` is no longer faster than `NGDPionS`, and must stay equal.

    The class was introduced when `NGDPionS` took `angle` from an exact
    `matrix_norm`. The power iteration has since moved into the parent, so the
    two now differ only in that the subclass records `quad`, `curv` and
    `pred_drop` for the reduction ratio. Pinning both halves keeps the docstring
    honest: if someone re-optimises one of them, this fails instead of the two
    silently diverging while the names still claim they are the same step.
    """
    import torch

    from ngd_pion.with_s import NGDPionS
    from ngd_pion.with_s_fast import FastNGDPionS

    DT = torch.float64
    M, N = 6, 4
    g = torch.Generator().manual_seed(0)
    W = torch.randn(M, N, dtype=DT, generator=g)
    x = torch.randn(64, N, dtype=DT, generator=g)
    d = torch.randn(64, M, dtype=DT, generator=g) * 0.3
    grads = [torch.randn(M, N, dtype=DT, generator=g) * 0.1 for _ in range(8)]

    def run(cls):
        p = torch.nn.Parameter(W.clone())
        opt = cls([p], lr=0.05, t_fac=3, compute_dtype=DT)
        opt.observe(p, x)
        opt.observe_backward(p, d)
        for G in grads:
            p.grad = G.clone()
            opt.step()
        return p.detach().clone(), opt.state[p]

    a, ref_state = run(NGDPionS)
    b, fast_state = run(FastNGDPionS)
    assert torch.equal(a, b), (a - b).abs().max()
    assert set(fast_state) - set(ref_state) == {"quad", "curv", "pred_drop"}
