"""Optimizer step alone, Turbo against Stiefel, on the shapes of one layer of Pion's 1.3B model (hidden 2048, ffn 5460)
or of LLaMA-60M. Synthetic statistics (131 072 tokens) and gradients; 30 timed steps after 30 warm-up steps.

    python scripts/probes/stiefel_bench.py [1.3B|60M] [layers]     # prints, writes runs-record/probes/stiefel_bench_<size>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ngd_pion.natural_pion import NaturalPionStiefel, NaturalPionTurbo

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
SIZE = sys.argv[1] if len(sys.argv) > 1 else "1.3B"
LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H, F = (2048, 5460) if SIZE == "1.3B" else (512, 1376)
N = 131072


def run(cls, parts=False):
    g = torch.Generator(device="cpu").manual_seed(0)
    ps = []
    for _ in range(LAYERS):
        for shape in [(H, H)] * 4 + [(F, H), (F, H), (H, F)]:
            ps.append(torch.nn.Parameter((torch.randn(*shape, generator=g) * 0.02).to(DEV)))
    opt = cls(ps, lr=0.2, t_fac=25)
    xa, xo, xf = torch.randn(N, H, device=DEV), torch.randn(N, H, device=DEV), torch.randn(N, F, device=DEV)
    dH, dF = torch.randn(N, H, device=DEV) / N, torch.randn(N, F, device=DEV) / N
    times, st = [], 0.0
    for t in range(60):
        if parts and t == 30:
            opt.profile = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, p in enumerate(ps):
            k = i % 7
            opt.observe(p, xa if k < 3 else xo if k == 3 else xa if k < 6 else xf)
            opt.observe_backward(p, dF if k in (4, 5) else dH)
            p.grad = torch.randn_like(p) * 1e-3
        if parts and t >= 30:
            torch.cuda.synchronize(); st += time.perf_counter() - t0
        opt.step()
        torch.cuda.synchronize()
        if t >= 30:
            times.append(time.perf_counter() - t0)
    if parts:
        prof = {"statistics + grads": st / 30, **{k: v / 30 for k, v in (opt.profile or {}).items()}}
        return sum(times) / len(times), prof
    return sum(times) / len(times)


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "size": SIZE, "layers": LAYERS}
    for name, cls in (("turbo", NaturalPionTurbo), ("stiefel", NaturalPionStiefel)):
        out[name] = run(cls)
        print(f"{SIZE} x{LAYERS} {name:8s} {out[name]:.4f} s per optimizer step", flush=True)
        torch.cuda.empty_cache()
    if "--parts" in sys.argv:
        for name, cls in (("turbo", NaturalPionTurbo), ("stiefel", NaturalPionStiefel)):
            tot, prof = run(cls, parts=True)
            out[name + "_parts"] = prof
            print(f"{SIZE} x{LAYERS} {name} parts (synchronised, total {tot:.4f}):", flush=True)
            for k, v in prof.items():
                print(f"    {k:26s} {v:.4f}", flush=True)
            torch.cuda.empty_cache()
    (ROOT / "runs-record" / "probes" / f"stiefel_bench_{SIZE}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
