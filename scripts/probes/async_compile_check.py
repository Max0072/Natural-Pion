"""Two checks for NaturalPionFast on a GPU, on the shapes of LLaMA-60M with synthetic statistics and gradients:

1. async_refresh: the same 100 steps with the refresh on a side stream and on the main stream must give
   bit-identical weights (the refresh only moves to another stream; nothing else changes).
2. torch.compile: seconds per optimizer step with `torch.compile(opt.step)` against eager, and whether the
   compiled step gives the same weights (to round-off).

    python scripts/probes/async_compile_check.py     # writes runs-record/probes/async_compile_check.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ngd_pion.natural_pion import NaturalPionFast, NaturalPionTurbo

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
N, H, F, L = 131072, 512, 1376, 8


def setup(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ps = []
    for _ in range(L):
        for shape in [(H, H)] * 4 + [(F, H), (F, H), (H, F)]:
            ps.append(torch.nn.Parameter((torch.randn(*shape, generator=g) * 0.02).to(DEV)))
    return ps


def drive(opt, ps, steps, compile_step=False, warm=0):
    gen = torch.Generator(device=DEV).manual_seed(1)
    xa = torch.randn(N, H, device=DEV, generator=gen); xo = torch.randn(N, H, device=DEV, generator=gen)
    xf = torch.randn(N, F, device=DEV, generator=gen)
    dH = torch.randn(N, H, device=DEV, generator=gen) / N; dF = torch.randn(N, F, device=DEV, generator=gen) / N
    step = torch.compile(opt.step) if compile_step else opt.step
    times = []
    for t in range(steps):
        gt = torch.Generator(device=DEV).manual_seed(100 + t)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, p in enumerate(ps):
            k = i % 7
            opt.observe(p, xa if k < 3 else xo if k == 3 else xa if k < 6 else xf)
            opt.observe_backward(p, dF if k in (4, 5) else dH)
            p.grad = torch.randn(p.shape, device=DEV, generator=gt) * 1e-3
        step()
        torch.cuda.synchronize()
        if t >= warm:
            times.append(time.perf_counter() - t0)
    return [p.detach().clone() for p in ps], (sum(times) / len(times) if times else None)


def main():
    out = {"gpu": torch.cuda.get_device_name(0)}
    a, b = setup(), setup()
    wa, _ = drive(NaturalPionTurbo(a, lr=0.2, async_refresh=False), a, 100)
    wb, _ = drive(NaturalPionTurbo(b, lr=0.2, async_refresh=True), b, 100)
    same = all(torch.equal(x, y) for x, y in zip(wa, wb))
    diff = max(float((x - y).abs().max()) for x, y in zip(wa, wb))
    out["async_bit_identical"] = same
    out["async_max_abs_diff"] = diff
    print(f"async refresh: bit-identical {same}, max |diff| {diff:.3e}", flush=True)

    c, d = setup(), setup()
    wc, te = drive(NaturalPionFast(c, lr=0.2), c, 150, warm=50)
    out["eager_s_per_step"] = te
    print(f"eager:    {te:.4f} s/step", flush=True)
    try:
        wd, tc = drive(NaturalPionFast(d, lr=0.2), d, 150, compile_step=True, warm=50)
        rel = max(float((x - y).abs().max() / x.abs().max()) for x, y in zip(wc, wd))
        out["compiled_s_per_step"] = tc
        out["compiled_max_rel_diff"] = rel
        print(f"compiled: {tc:.4f} s/step, max rel diff to eager {rel:.2e}", flush=True)
    except Exception as e:
        out["compiled"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
        print("compile failed:", out["compiled"], flush=True)
    (ROOT / "runs-record" / "probes" / "async_compile_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
