"""The vanilla-Pion baseline, and the properties the comparison depends on."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ngd_pion.linalg import skew
from ngd_pion.pion_baseline import Pion, _truncated_exp
from ngd_pion.reference import generators as ref_generators

DT = torch.float64


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DT)


def orth(n, seed=0):
    return torch.linalg.qr(rand(n, n, seed=seed))[0]


def test_generators_are_the_published_ones():
    """`g_in = W^T G - G^T W`, `g_out = G W^T - W G^T`, as in their pion.py."""
    from ngd_pion.direction import generators

    W, G = rand(9, 6, seed=1), rand(9, 6, seed=2)
    gi, go = generators(W, G)
    ri, ro = ref_generators(W.numpy(), G.numpy())
    assert np.abs(gi.numpy() - ri).max() < 1e-13
    assert np.abs(go.numpy() - ro).max() < 1e-13


def test_truncated_exp_is_the_taylor_series():
    A = skew(rand(7, 7, seed=3)) * 0.1
    want = torch.eye(7, dtype=DT) + A + A @ A / 2
    assert torch.allclose(_truncated_exp(A, 2), want)


def test_truncated_exp_only_approximates_orthogonality():
    """The reason the spectrum drifts rather than holding."""
    X = skew(rand(16, 16, seed=4))
    X = X / torch.linalg.matrix_norm(X, 2)
    eye = torch.eye(16, dtype=DT)
    small = _truncated_exp(0.05 * X, 2)
    large = _truncated_exp(2.0 * X, 2)
    assert 0 < (small.T @ small - eye).abs().max() < 1e-5
    assert (large.T @ large - eye).abs().max() > 0.5


@pytest.mark.parametrize("retraction,limit", [("cayley", 1e-13), ("trunc", None)])
def test_spectrum_holds_only_under_cayley(retraction, limit):
    W0 = orth(20, seed=5)
    s0 = torch.linalg.svdvals(W0)
    p = nn.Parameter(W0.clone())
    opt = Pion([p], lr=1e-2, scaling="rms", momentum="none", retraction=retraction, alternate=False)
    for i in range(200):
        p.grad = rand(20, 20, seed=100 + i)
        opt.step()
    drift = float((torch.linalg.svdvals(p.detach()) - s0).abs().max() / s0.max())
    if limit is None:
        assert drift > 1e-10, "the truncated exponential is expected to drift"
    else:
        assert drift < limit


def test_unscaled_truncated_exponential_diverges():
    """Why the ablated baseline cannot use their retraction.

    RMS scaling is load-bearing for Pion, not cosmetic: it keeps the rotation
    angle small enough that the truncated exponential's inflation does not
    compound. Switch the scaling off and the same retraction blows up within
    tens of steps, which is what forces the ablated baseline onto Cayley.
    """
    W0 = orth(24, seed=6)
    diverged = {}
    for retraction in ("trunc", "cayley"):
        p = nn.Parameter(W0.clone())
        opt = Pion([p], lr=0.1, scaling="none", momentum="none", retraction=retraction, alternate=False)
        blew_up = False
        for i in range(200):
            p.grad = rand(24, 24, seed=200 + i)
            opt.step()
            if not torch.isfinite(p.detach()).all():
                blew_up = True
                break
        diverged[retraction] = blew_up
    assert diverged["trunc"], "expected the truncated exponential to blow up unscaled"
    assert not diverged["cayley"], "Cayley must survive what the truncation cannot"


def test_rms_scaling_fixes_the_ambient_update_size():
    """`alpha` normalises so the per-element RMS of the ambient update is lr * rms."""
    W = orth(12, seed=7)
    p = nn.Parameter(W.clone())
    lr, rms = 1e-2, 0.2
    opt = Pion([p], lr=lr, scaling="rms", rms=rms, momentum="none", retraction="cayley", alternate=False)
    p.grad = rand(12, 12, seed=8)
    opt.step()
    delta = p.detach() - W
    got = float(delta.pow(2).mean().sqrt())
    assert 0.2 * lr * rms < got < 5 * lr * rms


def test_momentum_none_uses_the_raw_gradient():
    W = orth(8, seed=9)
    outs = {}
    for momentum in ("none", "ambient"):
        p = nn.Parameter(W.clone())
        opt = Pion([p], lr=1e-2, scaling="none", momentum=momentum, retraction="cayley", alternate=False)
        p.grad = rand(8, 8, seed=10)
        opt.step()
        outs[momentum] = p.detach().clone()
    assert not torch.allclose(outs["none"], outs["ambient"])


def test_alternate_moves_one_side_at_a_time():
    W = orth(10, seed=11)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, scaling="none", momentum="none", retraction="cayley", alternate=True)
    p.grad = rand(10, 10, seed=12)
    opt.step()
    # step 0 is the out-side: W <- R W, so W^T W is unchanged
    assert (p.detach().T @ p.detach() - W.T @ W).abs().max() < 1e-12


def test_invalid_options_are_refused():
    p = nn.Parameter(rand(4, 4, seed=13))
    for kwargs in (dict(scaling="x"), dict(momentum="x"), dict(retraction="x")):
        with pytest.raises(ValueError):
            Pion([p], lr=1e-3, **kwargs)
    with pytest.raises(ValueError, match="2-D"):
        Pion([nn.Parameter(torch.zeros(3))], lr=1e-3)
