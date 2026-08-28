"""First-moment averaging, and the guarantee that switching it off changes nothing.

`MomentumNGDPionS._apply` is a copy of its parent's with the smoothing inserted.
Copies drift. The first test here is what makes the copy safe: with
`momentum="none"` the trajectory must be **bit-identical** to `FastNGDPionS`,
so any future edit to one and not the other fails loudly instead of quietly
changing what the momentum arm is being compared against.
"""

from __future__ import annotations

import pytest
import torch

from ngd_pion.direction import generators
from ngd_pion.momentum import MomentumNGDPionS
from ngd_pion.with_s_fast import FastNGDPionS

DT = torch.float64
M, N = 6, 4
STEPS = 6


TOK = 64


def _setup(seed: int = 0):
    """`W`, the activations and output gradients the statistics are built from,
    and a gradient per step. `A` and `D` are accumulated through the public
    `observe` / `observe_backward` path rather than installed directly, since
    there is no setter for the backward side."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(M, N, dtype=DT, generator=g)
    x = torch.randn(TOK, N, dtype=DT, generator=g)
    d = torch.randn(TOK, M, dtype=DT, generator=g) * 0.3
    grads = [torch.randn(M, N, dtype=DT, generator=g) * 0.1 for _ in range(STEPS)]
    return W, x, d, grads


def _run(cls, W, x, d, grads, lr: float = 0.05, **kw):
    p = torch.nn.Parameter(W.clone())
    opt = cls([p], lr=lr, t_fac=3, compute_dtype=DT, **kw)
    opt.observe(p, x)
    opt.observe_backward(p, d)
    for G in grads:
        p.grad = G.clone()
        opt.step()
    return p.detach().clone(), opt


def test_momentum_none_is_bit_identical_to_the_parent():
    """The whole licence for copying `_apply`. Do not delete this test."""
    W, x, d, grads = _setup()
    a, _ = _run(FastNGDPionS, W, x, d, grads)
    b, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum="none")
    assert torch.equal(a, b), (a - b).abs().max()


def test_beta1_zero_is_also_the_parent():
    """`beta1 = 0` makes the EMA the identity, so it must agree too.

    A separate route to the same trajectory: `momentum="none"` skips the buffer
    entirely, `beta1 = 0` runs it and keeps only the current term. If these
    disagree the EMA is mis-scaled.
    """
    W, x, d, grads = _setup()
    a, _ = _run(FastNGDPionS, W, x, d, grads)
    b, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum="lie", beta1=0.0)
    assert torch.allclose(a, b, atol=1e-12)


def test_the_buffer_is_the_ema_it_claims_to_be():
    """After `k` constant gradients the buffer is `(1 - beta^k) g`, exactly."""
    W, x, d, _ = _setup()
    G = torch.randn(M, N, dtype=DT, generator=torch.Generator().manual_seed(9)) * 0.1
    beta, k = 0.9, 5
    _, opt = _run(MomentumNGDPionS, W, x, d, [G] * k, momentum="ambient", beta1=beta)
    p = opt.param_groups[0]["params"][0]
    m = opt.state[p]["m_ambient"]
    assert torch.allclose(m, (1.0 - beta ** k) * G, atol=1e-12)


def test_lie_buffers_stay_skew():
    """They must, or the step leaves the Lie algebra and Cayley stops rotating."""
    W, x, d, grads = _setup(seed=3)
    _, opt = _run(MomentumNGDPionS, W, x, d, grads, momentum="lie", beta1=0.9)
    p = opt.param_groups[0]["params"][0]
    for key in ("m_in", "m_out"):
        m = opt.state[p][key]
        assert torch.allclose(m, -m.transpose(-1, -2), atol=1e-14)


def test_lie_and_ambient_differ_but_only_at_second_order():
    """They are different operations, and the difference is smaller than it looks.

    `generators` is linear in `G`, so smoothing before and after extraction
    would coincide exactly if `W` were fixed. `W` moves, and the gap is
    therefore driven by the drift of `W` across the momentum window -- second
    order in the step size, not first. Measured on this fixture:

        lr    steps   relative |lie - ambient|
        0.05      6         1.8e-11
        0.5       6         1.8e-09
        0.5      20         5.2e-08

    So the distinction Pion draws between `lie_lie` and
    `transported_ambient_ambient` is nearly immaterial at small rotations. It
    need not stay that way here: this method's measured per-step angle is 0.2
    to 0.7 radians on the real model, far outside the range above, and nothing
    in this test speaks to that regime.
    """
    W, x, d, grads = _setup(seed=4)
    long = (grads * 4)[:20]
    a, _ = _run(MomentumNGDPionS, W, x, d, long, lr=0.5, momentum="lie", beta1=0.9)
    b, _ = _run(MomentumNGDPionS, W, x, d, long, lr=0.5, momentum="ambient", beta1=0.9)
    rel = float((a - b).norm() / a.norm())
    assert rel > 1e-9, rel


def test_momentum_changes_the_trajectory():
    W, x, d, grads = _setup(seed=5)
    a, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum="none")
    b, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum="lie", beta1=0.9)
    assert not torch.allclose(a, b, atol=1e-9)


def test_spectrum_is_still_preserved():
    """Momentum must not cost the invariant the method is built on."""
    W, x, d, grads = _setup(seed=6)
    before = torch.linalg.svdvals(W)
    a, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum="lie", beta1=0.9)
    assert torch.allclose(before, torch.linalg.svdvals(a), rtol=1e-10, atol=1e-12)


def test_rejects_bad_arguments():
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=DT))
    with pytest.raises(ValueError):
        MomentumNGDPionS([p], momentum="nesterov")
    with pytest.raises(ValueError):
        MomentumNGDPionS([p], beta1=1.0)
