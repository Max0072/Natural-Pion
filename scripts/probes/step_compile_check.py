"""torch.compile for the optimizer step of Natural Pion Fast, on a GPU, on the shapes of LLaMA-60M (synthetic statistics
and gradients, query projection per head): seconds per optimizer step eager, compiled (default) and compiled with
mode="reduce-overhead" (CUDA graphs), and the largest relative weight difference to eager after the same steps.

    python scripts/probes/step_compile_check.py      # writes runs-record/probes/step_compile_check.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ngd_pion.natural_pion import NaturalPionFast

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
H, F, L, N = 512, 1376, 8, 131072 // 2


def run(mode: str | None, steps: int = 150, warm: int = 50):
    g = torch.Generator(device="cpu").manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(*s, generator=g) * 0.02).to(DEV))
          for _ in range(L) for s in [(H, H)] * 4 + [(F, H), (F, H), (H, F)]]
    opt = NaturalPionFast(ps, lr=0.2, row_blocks={id(p): 8 for p in ps[0::7]})
    step = opt.step if mode is None else torch.compile(opt.step, **({} if mode == "default" else {"mode": mode}))
    gen = torch.Generator(device=DEV).manual_seed(1)
    xa = torch.randn(N, H, device=DEV, generator=gen); xf = torch.randn(N, F, device=DEV, generator=gen)
    dH = torch.randn(N, H, device=DEV, generator=gen) / N; dF = torch.randn(N, F, device=DEV, generator=gen) / N
    ts = []
    for t in range(steps):
        gt = torch.Generator(device=DEV).manual_seed(100 + t)
        for i, p in enumerate(ps):
            k = i % 7
            opt.observe(p, xf if k == 6 else xa)
            opt.observe_backward(p, dF if k in (4, 5) else dH)
            p.grad = torch.randn(p.shape, device=DEV, generator=gt) * 1e-3
        torch.cuda.synchronize(); t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        if t >= warm:
            ts.append(time.perf_counter() - t0)
    return [p.detach().clone() for p in ps], sum(ts) / len(ts)


def main():
    out = {"gpu": torch.cuda.get_device_name(0)}
    w0, out["eager_sps"] = run(None)
    print(f"eager            {out['eager_sps']:.4f} s per optimizer step", flush=True)
    for mode in ("default", "reduce-overhead"):
        try:
            w, sps = run(mode)
            rel = max(float((a - b).abs().max() / a.abs().max()) for a, b in zip(w0, w))
            out[f"{mode}_sps"], out[f"{mode}_max_rel_diff"] = sps, rel
            print(f"compile {mode:16s} {sps:.4f} s per optimizer step, max rel diff to eager {rel:.1e}", flush=True)
        except Exception as e:
            out[mode] = f"failed: {type(e).__name__}: {str(e)[:300]}"
            print(f"compile {mode}: failed: {out[mode]}", flush=True)
    (ROOT / "runs-record" / "probes" / "step_compile_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
