"""`NaturalPionFast` is the probe `scripts/probes/step_norm.py` (FAST, BATCH, A and D last k) bit for bit,
and `NaturalPion` keeps every weight's singular values."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from ngd_pion.natural_pion import NaturalPion, NaturalPionFast, NaturalPionTurbo

ROOT = Path(__file__).resolve().parents[1]
SHAPES = [(12, 8), (8, 8), (8, 8), (8, 12), (12, 8)]  # repeated shapes and sizes exercise the batching


def _probe_class(window: int):
    """Import the probe with its flags set; the flags are read at import time."""
    env = {"STEP_NORM_FAST": "1", "STEP_NORM_BATCH": "1",
           "STEP_NORM_A_LAST": str(window), "STEP_NORM_D_LAST": str(window)}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location(f"step_norm_probe_{window}", ROOT / "scripts/probes/step_norm.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod.StepNorm


def _weights(seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]


def _drive(opt, params, steps: int, seed: int, schedule):
    """Feed identical statistics and gradients, step with a scheduled angle; return the weights per step."""
    g = torch.Generator().manual_seed(seed)
    trace = []
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = schedule(t)
        for p in params:
            m, n = p.shape
            opt.observe(p, torch.randn(64, n, generator=g))
            opt.observe_backward(p, torch.randn(64, m, generator=g) / 64.0)
            p.grad = torch.randn(m, n, generator=g)
        opt.step()
        trace.append([p.detach().clone() for p in params])
    return trace


def _cosine(t, peak=0.2, steps=60):
    import math
    return 1e-5 + 0.5 * (peak - 1e-5) * (1 + math.cos(math.pi * min(1.0, t / steps)))


@pytest.mark.parametrize("t_fac,window", [(5, 2), (25, 6)])
def test_fast_is_the_probe_bit_for_bit(t_fac, window):
    StepNorm = _probe_class(window)
    a, b = _weights(0), _weights(0)
    probe = StepNorm(a, lr=0.2, beta=0.5, eps=1e-4, alpha_max=1.0, t_fac=t_fac, use_s=True,
                     momentum="lie", beta1=0.9, trust="none", beta_backward=0.5)
    ours = NaturalPionFast(b, lr=0.2, t_fac=t_fac, stats_window=window, retraction="cayley", batch_sigma=False)
    ta = _drive(probe, a, 60, 1, _cosine)
    tb = _drive(ours, b, 60, 1, _cosine)
    for t, (wa, wb) in enumerate(zip(ta, tb)):
        for x, y in zip(wa, wb):
            assert torch.equal(x, y), f"step {t}: max diff {(x - y).abs().max().item():.3e}"


def test_exact_form_keeps_the_spectrum_and_moves_the_weights():
    params = _weights(3)
    before = [torch.linalg.svdvals(p.detach().double()) for p in params]
    start = [p.detach().clone() for p in params]
    _drive(NaturalPion(params, lr=0.2), params, 40, 4, _cosine)
    for p, s0, w0 in zip(params, before, start):
        s1 = torch.linalg.svdvals(p.detach().double())
        assert torch.allclose(s0, s1, rtol=1e-4), (s0 - s1).abs().max()
        assert (p.detach() - w0).abs().max() > 1e-3


def test_statistics_window():
    params = _weights(5)
    opt = NaturalPionFast(params, lr=0.1, t_fac=10, stats_window=3)
    _drive(opt, params, 1, 6, lambda t: 0.1)          # first step builds the basis
    st = opt.state[params[0]]
    assert st["since_refactor"] == 1
    assert not opt._collecting(params[0])              # 1 < 10 - 3
    st["since_refactor"] = 7
    assert opt._collecting(params[0])                  # 7 >= 10 - 3


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        NaturalPion([torch.nn.Parameter(torch.zeros(3))])
    with pytest.raises(ValueError):
        NaturalPion(_weights(0), t_fac=5, stats_window=6)


def test_row_blocks_rotate_each_block_and_keep_its_spectrum():
    params = _weights(7)
    q = params[1]                                       # 8 x 8, split into 4 blocks of 2 rows
    before = [torch.linalg.svdvals(q.detach()[2 * i:2 * i + 2].double()) for i in range(4)]
    opt = NaturalPionFast(params, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(q): 4})
    _drive(opt, params, 30, 8, _cosine)
    for i in range(4):
        after = torch.linalg.svdvals(q.detach()[2 * i:2 * i + 2].double())
        assert torch.allclose(before[i], after, rtol=1e-4), i
    assert "blocks" in opt.state[q] and len(opt.state[q]["blocks"]) == 3


def test_one_block_is_the_unsplit_optimizer():
    a, b = _weights(9), _weights(9)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2)
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(b[1]): 1})
    for x, y in zip(_drive(oa, a, 20, 10, _cosine)[-1], _drive(ob, b, 20, 10, _cosine)[-1]):
        assert torch.equal(x, y)


def test_newton_schulz_retraction_matches_cayley_at_small_angles():
    a, b = _weights(11), _weights(11)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley")
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2, retraction="ns", ns_iters=2)
    ta, tb = _drive(oa, a, 30, 12, _cosine), _drive(ob, b, 30, 12, _cosine)
    for x, y in zip(ta[-1], tb[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    before = [torch.linalg.svdvals(p.detach().double()) for p in _weights(11)]
    for p, s0 in zip(b, before):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-4)


def test_truncated_exponential_is_close_to_cayley_and_inflates_the_spectrum():
    a, b = _weights(13), _weights(13)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley")
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2, retraction="trunc")
    ta, tb = _drive(oa, a, 30, 14, _cosine), _drive(ob, b, 30, 14, _cosine)
    w0 = _weights(13)
    for x, y, z in zip(ta[-1], tb[-1], w0):
        assert (x - y).abs().max() < 5e-2 * (x - z).abs().max()       # third-order differences, 30 steps at 0.2
        s0 = torch.linalg.svdvals(z.detach().double())
        assert (torch.linalg.svdvals(y.double()) >= s0 * (1 - 1e-6)).all()  # T^T T = I + A^4/4 >= I: never shrinks


def _drive_shared(opt, params, steps, seed, share):
    """q, k, v-like: params 1 and 2 (both 8 x 8) read the same input tensor, as a fused pre-norm does."""
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = _cosine(t)
        xs = {}
        for i, p in enumerate(params):
            m, n = p.shape
            if i == 2:
                x = xs[1] if share else xs[1].clone()
                torch.randn(64, n, generator=g)            # keep the generator in step with the other branch
            else:
                x = torch.randn(64, n, generator=g)
            xs[i] = x
            opt.observe(p, x)
            opt.observe_backward(p, torch.randn(64, m, generator=g) / 64.0)
            p.grad = torch.randn(m, n, generator=g)
        opt.step()
    return [p.detach().clone() for p in params]


def test_shared_input_moment_is_bit_identical():
    a, b = _weights(15), _weights(15)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, share_inputs=False)
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2)           # shares by default
    wa, wb = _drive_shared(oa, a, 30, 16, share=False), _drive_shared(ob, b, 30, 16, share=True)
    for x, y in zip(wa, wb):
        assert torch.equal(x, y)
    assert ob.state[b[1]]["cov"] is ob.state[b[2]]["cov"]
    assert oa.state[a[1]]["cov"] is not oa.state[a[2]]["cov"]


def test_bf16_statistics_are_the_gram_of_rounded_inputs():
    from ngd_pion.covariance import CovarianceAccumulator
    x = torch.randn(512, 16, generator=torch.Generator().manual_seed(17)) * 3
    acc = CovarianceAccumulator(beta=0.5, input_dtype=torch.bfloat16)
    acc.observe(x)
    q = x.to(torch.bfloat16).float()
    assert acc.matrix.dtype == torch.float32
    assert torch.allclose(acc.matrix, q.T @ q / 512, rtol=1e-6, atol=1e-6)
    assert torch.linalg.eigvalsh(acc.matrix.double()).min() > 0


def test_bf16_statistics_keep_the_spectrum_and_track_fp32():
    a, b = _weights(19), _weights(19)
    s0 = [torch.linalg.svdvals(p.detach().double()) for p in a]
    wa = _drive(NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2), a, 30, 20, _cosine)[-1]
    wb = _drive(NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2, stats_dtype="bf16"), b, 30, 20, _cosine)[-1]
    for x, y, z, s in zip(wa, wb, _weights(19), s0):
        assert (x - y).abs().max() < 0.1 * (x - z).abs().max()
        assert torch.allclose(torch.linalg.svdvals(y.double()), s, rtol=1e-4)


def test_newton_schulz_in_correction_form_matches_cayley_and_keeps_the_norm():
    a, b = _weights(21), _weights(21)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley")
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2)
    ta, tb = _drive(oa, a, 30, 22, _cosine), _drive(ob, b, 30, 22, _cosine)
    for x, y in zip(ta[-1], tb[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    for p, s0 in zip(b, [torch.linalg.svdvals(p.detach().double()) for p in _weights(21)]):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-5)


@pytest.mark.parametrize("retraction", ["ns_delta", "cayley"])
def test_turbo_is_fast_batched(retraction):
    a, b = _weights(23), _weights(23)
    fa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, retraction=retraction, row_blocks={id(a[1]): 4})
    fb = NaturalPionTurbo(b, lr=0.2, t_fac=5, stats_window=2, retraction=retraction, stats_dtype="fp32",
                          row_blocks={id(b[1]): 4})
    ta, tb = _drive(fa, a, 30, 24, _cosine), _drive(fb, b, 30, 24, _cosine)
    for x, y in zip(ta[-1], tb[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    for j, (p, s0) in enumerate(zip(b, [torch.linalg.svdvals(p.detach().double()) for p in _weights(23)])):
        if j != 1:                                      # b[1] is split into row blocks: each block keeps its own
            assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-4)


def test_batched_sigma_matches_the_loop():
    a, b = _weights(25), _weights(25)
    oa = NaturalPionFast(a, lr=0.2, t_fac=5, stats_window=2, batch_sigma=False, row_blocks={id(a[1]): 4})
    ob = NaturalPionFast(b, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(b[1]): 4})
    for x, y in zip(_drive(oa, a, 30, 26, _cosine)[-1], _drive(ob, b, 30, 26, _cosine)[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()   # batched reductions: round-off only
