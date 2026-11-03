"""`NaturalPionFast` is the probe `scripts/probes/step_norm.py` (FAST, BATCH, A and D last k) bit for bit,
and `NaturalPion` keeps every weight's singular values."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from ngd_pion.natural_pion import NaturalPion, NaturalPionFast

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
    ours = NaturalPionFast(b, lr=0.2, t_fac=t_fac, stats_window=window)
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
