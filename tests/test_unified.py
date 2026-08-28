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
    """`trust="exact"` consumes its paired token sample each step, so the
    statistics are fed inside the loop, which is what the forward and backward
    hooks do in a real run."""
    p = torch.nn.Parameter(W.clone())
    opt = cls([p], lr=lr, t_fac=3, compute_dtype=DT, **kw)
    for G in grads:
        opt.observe(p, x)
        if backward:
            opt.observe_backward(p, d)
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


# --- the trust region: quad/curv is not one, quad/curv_exact is --------------


def test_trust_none_pins_alpha_at_the_cap():
    W, x, d, grads = _fixture(seed=8)
    _, opt = _run(NGDPionUnified, W, x, d, grads, trust="none")
    p = opt.param_groups[0]["params"][0]
    assert opt.state[p]["alpha"] == 1.0


def test_on_a_fresh_basis_quad_curv_is_one_and_exact_is_not():
    """The structural difference, checked where the algebra applies.

    `quad/curv` is 1 on a **fresh** basis identically -- both halves come from
    the operator that built `X` -- so it cannot shorten anything there. It is
    checked at the first step for that reason: with `t_fac = 3` and eight steps
    the basis does go stale and `alpha` drifts below 1, which is the staleness
    reading and not a trust region.

    `quad/curv_exact` compares against a curvature measured on tokens, so it is
    free to differ at step one, and does.
    """
    W, x, d, grads = _fixture(seed=9)
    one = grads[:1]
    _, opt_a = _run(NGDPionUnified, W, x, d, one, trust="quad_curv")
    _, opt_b = _run(NGDPionUnified, W, x, d, one, trust="exact")
    pa = opt_a.param_groups[0]["params"][0]
    pb = opt_b.param_groups[0]["params"][0]
    assert opt_a.state[pa]["alpha"] == pytest.approx(1.0, abs=1e-9)
    assert opt_b.state[pb]["alpha"] != pytest.approx(1.0, abs=1e-3)


def test_exact_trust_moves_the_trajectory():
    W, x, d, grads = _fixture(seed=9)
    a, _ = _run(NGDPionUnified, W, x, d, grads, trust="quad_curv")
    b, _ = _run(NGDPionUnified, W, x, d, grads, trust="exact")
    assert not torch.allclose(a, b, atol=1e-9)


def test_the_cauchy_schwarz_bound_holds_on_independent_data():
    """`curv_exact / curv <= 2` when `x` and `delta` really are independent.

    This fixture draws them independently, so the independence assumption holds
    **by construction** and the ratio is bounded by `2` -- it can and does come
    out below 1. The real-model finding that `curv_exact` exceeds `curv` by 8 to
    213 times is a statement about a trained network, where they are dependent,
    and asserting it here would be transplanting a result onto data that cannot
    exhibit it.
    """
    W, x, d, grads = _fixture(seed=10)
    _, opt = _run(NGDPionUnified, W, x, d, grads[:1], trust="exact")
    p = opt.param_groups[0]["params"][0]
    ratio = float(opt.state[p]["curv_exact"]) / float(opt.state[p]["curv"])
    assert 0.0 < ratio <= 2.0 + 1e-9, ratio


def test_exact_alpha_agrees_with_the_diagnostic_class():
    """Two independent implementations of the same ratio, on the same step.

    `ExactCurvNGDPionS` computes `alpha_exact` as a diagnostic after taking its
    parent's step; this class computes the same ratio before taking a step of
    its own. At the first step the weight, the gradient and the basis are the
    same for both, and both draw their token sample from a generator seeded
    identically, so the numbers have to match.
    """
    from ngd_pion.exact_curv import ExactCurvNGDPionS

    W, x, d, grads = _fixture(seed=11)
    one = grads[:1]

    p = torch.nn.Parameter(W.clone())
    diag = ExactCurvNGDPionS([p], lr=0.05, t_fac=3, compute_dtype=DT,
                             exact_every=1, exact_tokens=64)
    diag.observe(p, x)
    diag.observe_backward(p, d)
    p.grad = one[0].clone()
    diag.step()
    from_diagnostic = diag.state[p]["alpha_exact"]

    q = torch.nn.Parameter(W.clone())
    live = NGDPionUnified([q], lr=0.05, t_fac=3, compute_dtype=DT,
                          trust="exact", exact_tokens=64)
    live.observe(q, x)
    live.observe_backward(q, d)
    q.grad = one[0].clone()
    live.step()
    from_live = live.state[q]["alpha_exact"]

    # Not bit equality: `exact_curv` casts to fp32 in both, and the two divide
    # in different dtypes afterwards, so five significant figures is the most
    # the comparison can carry. What it establishes is that two independent
    # implementations of the ratio agree, which is the point.
    assert from_live == pytest.approx(from_diagnostic, rel=1e-4), (from_live, from_diagnostic)


def test_exact_trust_refuses_without_the_backward_statistic():
    """It needs per-token output gradients, which only the measured-`S` path has."""
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=DT))
    with pytest.raises(ValueError):
        NGDPionUnified([p], trust="exact", use_s=False)
    with pytest.raises(ValueError):
        NGDPionUnified([p], trust="pumpkin")


def test_exact_beta_zero_is_the_instantaneous_estimate():
    """The default must not move, or every reading before it changes meaning."""
    W, x, d, grads = _fixture(seed=12)
    a, _ = _run(NGDPionUnified, W, x, d, grads, trust="exact")
    b, _ = _run(NGDPionUnified, W, x, d, grads, trust="exact", exact_beta=0.0)
    assert torch.equal(a, b)


def test_exact_beta_seeds_from_the_first_observation():
    """Seeded at the first draw, in the log, so the geometric mean starts there
    rather than warming from zero over 170 steps."""
    W, x, d, grads = _fixture(seed=13)
    _, opt = _run(NGDPionUnified, W, x, d, grads[:1], trust="exact", exact_beta=0.99)
    p = opt.param_groups[0]["params"][0]
    assert torch.allclose(torch.exp(opt.state[p]["curv_exact_log"]),
                          opt.state[p]["curv_exact"], rtol=1e-12)


def test_exact_beta_smooths_and_keeps_the_raw_estimate():
    """Both are recorded: the step uses the smooth one, the diagnostic keeps
    measuring the noise the flag exists to remove."""
    W, x, d, grads = _fixture(seed=14)
    _, opt = _run(NGDPionUnified, W, x, d, grads, trust="exact", exact_beta=0.9)
    p = opt.param_groups[0]["params"][0]
    st = opt.state[p]
    assert "curv_exact" in st and "curv_exact_smooth" in st
    # after several differing gradients the two must have parted company
    assert not torch.allclose(st["curv_exact"], st["curv_exact_smooth"], rtol=1e-6)


def test_exact_beta_reduces_the_variation_of_alpha():
    """The point of the flag, as a test rather than as a hope.

    **`alpha_max` is lifted out of the way on purpose.** At the cap the ratio is
    clipped, and clipping compresses variation, so a comparison taken at the cap
    measures the clamp rather than the smoothing -- with `alpha_max = 1` the
    unsmoothed arm sits at a mean of 0.84 against the cap and reads as the
    *less* variable of the two. On the real model `alpha` runs 0.02 to 0.07 and
    the cap is nowhere near, so lifting it is what makes the fixture resemble
    the thing being tested.

    What smoothing can and cannot do: `alpha = quad / curv_exact`, and only the
    denominator is smoothed. Its floor is therefore the variation of `quad`,
    which in a real run comes from the whole batch rather than from a 4096-token
    subsample and is far better determined.
    """
    import statistics as stat

    W, x, d, grads = _fixture(seed=15)
    long = (grads * 3)[:20]

    def alphas(beta):
        torch.manual_seed(7)
        p = torch.nn.Parameter(W.clone())
        opt = NGDPionUnified([p], lr=0.05, t_fac=3, compute_dtype=DT, alpha_max=1e6,
                             trust="exact", exact_beta=beta)
        out = []
        for G in long:
            opt.observe(p, x + 0.4 * torch.randn_like(x))
            opt.observe_backward(p, d + 0.4 * torch.randn_like(d))
            p.grad = G.clone()
            opt.step()
            out.append(opt.state[p]["alpha"])
        return out[5:]

    raw, smooth = alphas(0.0), alphas(0.95)
    rel = lambda v: stat.pstdev(v) / stat.mean(v)
    assert rel(smooth) < rel(raw), (rel(raw), rel(smooth))


def test_the_smoothing_is_geometric_not_arithmetic():
    """The distinction that decides whether the flag works.

    `curv_exact` is heavy-tailed, so an arithmetic EMA converges to a mean ten
    times the typical value and `alpha` collapses -- measured at `beta = 0.99`,
    the median `alpha` reached 0.0000 by step 101 and the rotation died. Fed a
    sequence with one large outlier, the geometric mean must stay near the bulk
    while the arithmetic mean would be dragged to it.
    """
    import math

    beta = 0.9
    vals = [1.0] * 9 + [1000.0]
    log_ema = math.log(vals[0])
    arith = vals[0]
    for v in vals[1:]:
        log_ema = beta * log_ema + (1 - beta) * math.log(v)
        arith = beta * arith + (1 - beta) * v
    geo = math.exp(log_ema)
    assert geo < 3.0, geo            # stays with the bulk
    assert arith > 90.0, arith       # dragged by the single outlier
