"""The unified step against every class it replaces, and the two kinds of flag.

**Algorithmic flags must reproduce their class bit for bit.** That is what makes
this refactor provable rather than hopeful: each setting is checked against the
class whose results are already on disk, and `torch.equal` is the assertion, not
`allclose`.

**Implementation flags are checked differently.** `angle` feeds nothing in the
step, so however it is computed the weights must be untouched. `retraction="ns"`
is an approximation, so it is checked against a bound that depends on the
rotation angle rather than against equality -- and the bound is the one
`linalg.cayley_newton_schulz` documents.
"""

from __future__ import annotations

import pytest
import torch

from ngd_pion.momentum import MomentumNGDPionS
from ngd_pion.optimizer import NGDPion
from ngd_pion.unified import NGDPionUnified
from ngd_pion.with_s_fast import FastNGDPionS

DT = torch.float64
M, N = 6, 4
STEPS = 8


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(M, N, dtype=DT, generator=g),
            torch.randn(64, N, dtype=DT, generator=g),
            torch.randn(64, M, dtype=DT, generator=g) * 0.3,
            [torch.randn(M, N, dtype=DT, generator=g) * 0.1 for _ in range(STEPS)])


def _run(cls, W, x, d, grads, *, backward=True, lr=0.05, **kw):
    p = torch.nn.Parameter(W.clone())
    opt = cls([p], lr=lr, t_fac=3, compute_dtype=DT, **kw)
    opt.observe(p, x)
    if backward:
        opt.observe_backward(p, d)
    for G in grads:
        p.grad = G.clone()
        opt.step()
    return p.detach().clone(), opt


# --- algorithmic flags: bit-identity with the classes they replace -----------


def test_use_s_true_reproduces_the_s_variant():
    W, x, d, grads = _fixture()
    a, _ = _run(FastNGDPionS, W, x, d, grads)
    b, _ = _run(NGDPionUnified, W, x, d, grads, use_s=True, momentum="none")
    assert torch.equal(a, b), (a - b).abs().max()


@pytest.mark.parametrize("mode", ["lie", "ambient"])
def test_momentum_reproduces_the_momentum_class(mode):
    W, x, d, grads = _fixture(seed=1)
    a, _ = _run(MomentumNGDPionS, W, x, d, grads, momentum=mode, beta1=0.9)
    b, _ = _run(NGDPionUnified, W, x, d, grads, use_s=True, momentum=mode, beta1=0.9)
    assert torch.equal(a, b), (a - b).abs().max()


def test_use_s_false_reproduces_the_reference_without_s():
    """`S = I` is a branch, not an identity matrix passed through the `S` path.

    Weights only: `NGDPion` takes its angle from an exact `matrix_norm` and this
    class does not, but the angle feeds nothing in the step, so the trajectories
    have to agree exactly anyway.
    """
    W, x, d, grads = _fixture(seed=2)
    a, _ = _run(NGDPion, W, x, d, grads, backward=False)
    b, _ = _run(NGDPionUnified, W, x, d, grads, backward=False, use_s=False)
    assert torch.equal(a, b), (a - b).abs().max()


def test_s_and_no_s_actually_differ():
    """The negative control: if they agreed, the flag would be doing nothing."""
    W, x, d, grads = _fixture(seed=3)
    a, _ = _run(NGDPionUnified, W, x, d, grads, use_s=True)
    b, _ = _run(NGDPionUnified, W, x, d, grads, use_s=False)
    assert not torch.allclose(a, b, atol=1e-9)


# --- implementation flags: the angle must be free ---------------------------


@pytest.mark.parametrize("mode", ["svd", "off"])
def test_angle_does_not_touch_the_trajectory(mode):
    """It is read by the instrument and by the Newton-Schulz guard, not the step."""
    W, x, d, grads = _fixture(seed=4)
    a, _ = _run(NGDPionUnified, W, x, d, grads, angle="power")
    b, _ = _run(NGDPionUnified, W, x, d, grads, angle=mode)
    assert torch.equal(a, b), (a - b).abs().max()


def test_angle_off_records_nan_rather_than_a_stale_number():
    W, x, d, grads = _fixture(seed=4)
    _, opt = _run(NGDPionUnified, W, x, d, grads, angle="off")
    p = opt.param_groups[0]["params"][0]
    assert opt.state[p]["angle"] != opt.state[p]["angle"]


# --- implementation flags: the retraction is bounded, not exact --------------


def test_newton_schulz_is_close_but_not_equal():
    """An approximation, so equality would mean the flag was inert."""
    W, x, d, grads = _fixture(seed=5)
    a, _ = _run(NGDPionUnified, W, x, d, grads, retraction="cayley")
    b, _ = _run(NGDPionUnified, W, x, d, grads, retraction="ns", ns_iters=2, ns_guard=0.0)
    assert not torch.equal(a, b)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (a - b).abs().max()


def test_newton_schulz_still_preserves_the_spectrum():
    """The invariant has to survive the cheap retraction or it is not usable."""
    W, x, d, grads = _fixture(seed=6)
    before = torch.linalg.svdvals(W)
    b, _ = _run(NGDPionUnified, W, x, d, grads, retraction="ns", ns_iters=2, ns_guard=0.0)
    assert torch.allclose(before, torch.linalg.svdvals(b), rtol=1e-5, atol=1e-7)


def test_the_guard_falls_back_where_the_angle_is_large():
    """With the guard wide open the step must equal the exact retraction.

    `ns_guard = 0` disables the fallback, so a guard set below every angle seen
    has to reproduce `cayley` exactly -- which is what makes the guard testable
    rather than a comfort.
    """
    W, x, d, grads = _fixture(seed=7)
    a, _ = _run(NGDPionUnified, W, x, d, grads, retraction="cayley")
    b, _ = _run(NGDPionUnified, W, x, d, grads, retraction="ns", ns_guard=1e-12)
    assert torch.equal(a, b), (a - b).abs().max()


# --- rejections --------------------------------------------------------------


def test_rejects_nonsense():
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=DT))
    for kw in ({"momentum": "nesterov"}, {"retraction": "pade"}, {"angle": "lanczos"},
               {"beta1": 1.0}, {"ns_iters": 0}):
        with pytest.raises(ValueError):
            NGDPionUnified([p], **kw)


def test_rejects_a_guard_it_cannot_evaluate():
    """`angle="off"` with a live Newton-Schulz guard would disable it silently."""
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=DT))
    with pytest.raises(ValueError):
        NGDPionUnified([p], angle="off", retraction="ns", ns_guard=0.5)
    NGDPionUnified([p], angle="off", retraction="ns", ns_guard=0.0)


def test_every_flag_reaches_the_optimizer():
    """The `ngd_power` bug class, closed for the flags this class introduces.

    `ngd_power` reaches the manifest, the configuration hash and the directory
    name without reaching the optimizer, so six runs on disk record a controlled
    variable that was not one. More flags means more of that surface, so each
    one is asserted to arrive in `param_groups` with the value it was given --
    and given a *non-default* value, since a default would pass whether or not
    it was wired.
    """
    from dataclasses import replace

    from harness.config import ModelConfig, RunConfig
    from harness.model import Transformer
    from harness.train import build_optimizers

    small = ModelConfig(vocab_size=256, hidden=64, layers=2, heads=2,
                        ffn_hidden=176, seq_len=32)
    cfg = replace(
        RunConfig(optimizer="ngd-pion-u", model=small),
        lr=0.037, ngd_use_s=False, ngd_momentum="ambient", ngd_beta1=0.77,
        ngd_retraction="ns", ngd_ns_iters=3, ngd_ns_guard=0.25, ngd_angle="svd",
        ngd_t_fac=11,
    )
    rot, _, _ = build_optimizers(Transformer(small), cfg)
    assert isinstance(rot, NGDPionUnified)
    g = rot.param_groups[0]
    assert (g["use_s"], g["momentum"], g["beta1"]) == (False, "ambient", 0.77)
    assert (g["retraction"], g["ns_iters"], g["ns_guard"]) == ("ns", 3, 0.25)
    assert (g["angle"], g["lr"], g["t_fac"]) == ("svd", 0.037, 11)


def test_the_named_presets_are_what_they_claim():
    """`ngd-pion-s` is `use_s` with no momentum; `ngd-pion-m` adds it.

    The names are what every sbatch script and every run directory on disk uses,
    so the presets behind them have to stay fixed even though the class is now
    shared.
    """
    from dataclasses import replace

    from harness.config import ModelConfig, RunConfig
    from harness.model import Transformer
    from harness.train import build_optimizers

    small = ModelConfig(vocab_size=256, hidden=64, layers=2, heads=2,
                        ffn_hidden=176, seq_len=32)
    for name, expect in (("ngd-pion-s", "none"), ("ngd-pion-m", "lie")):
        cfg = replace(RunConfig(optimizer=name, model=small), ngd_momentum="lie")
        rot, _, _ = build_optimizers(Transformer(small), cfg)
        g = rot.param_groups[0]
        assert g["use_s"] is True, name
        assert g["momentum"] == expect, (name, g["momentum"])
