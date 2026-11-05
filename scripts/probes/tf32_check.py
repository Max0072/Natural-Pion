"""TF32 against true fp32 for Natural Pion Fast, on a GPU. Two questions:

1. Spectrum: 200 two-sided Cayley steps of a 512 x 1376 matrix with random skew generators, as in paper_checks.py's
   tf32_check, but with the Cayley step computed by Newton-Schulz in Z - I (the method) instead of an exact solve; at
   angles 1e-2 and 0.1. Reported: max_i |s_i / s_i0 - 1| after 200 steps, fp32 vs TF32 (the exact solve for reference).
2. Time: seconds per optimizer step of NaturalPionFast on the shapes of LLaMA-60M (synthetic statistics and gradients,
   query projection per head) with tf32 = off / upstream / all, F^-1 in bf16, and bf16 F^-1 with TF32 upstream.

    python scripts/probes/tf32_check.py      # writes runs-record/probes/tf32_check.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ngd_pion.linalg import exact_fp32, skew, tf32_matmul
from ngd_pion.natural_pion import NaturalPionFast

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
OUT: dict = {"gpu": torch.cuda.get_device_name(0)}


def cayley(X, c, how):
    n = X.shape[0]
    eye = torch.eye(n, device=X.device)
    A = 0.5 * c * X
    if how == "solve":
        return torch.linalg.solve(eye + A, eye - A)
    z = -A                                              # Newton-Schulz in Z - I, two iterations (NaturalPion)
    for _ in range(2):
        r = A + z + A @ z
        z = z - r - z @ r
    return eye + 2.0 * z


def drift():
    g = torch.Generator(device="cpu").manual_seed(7)
    W0 = torch.randn(512, 1376, generator=g).to(DEV)
    s0 = torch.linalg.svdvals(W0.double())
    for angle in (1e-2, 1e-1):
        for how in ("ns_delta", "solve"):
            for prec, ctx in (("fp32", exact_fp32), ("tf32", tf32_matmul)):
                W = W0.clone()
                gg = torch.Generator(device="cpu").manual_seed(8)
                with ctx():
                    for _ in range(200):
                        Xo = skew(torch.randn(512, 512, generator=gg)).to(DEV)
                        Xi = skew(torch.randn(1376, 1376, generator=gg)).to(DEV)
                        sig = max(float(torch.linalg.matrix_norm(Xi.double(), 2)), float(torch.linalg.matrix_norm(Xo.double(), 2)))
                        c = angle / sig
                        W = cayley(Xo, c, how) @ W @ cayley(Xi, c, how)
                d = float((torch.linalg.svdvals(W.double()) / s0 - 1).abs().max())
                OUT[f"drift_{how}_{prec}_{angle:g}"] = d
                print(f"angle {angle:g} {how:8s} {prec}: max |s/s0 - 1| after 200 steps = {d:.2e}", flush=True)


def timing():
    H, F, L, N = 512, 1376, 8, 131072 // 2
    g = torch.Generator(device="cpu").manual_seed(0)
    for mode, kw in (("off", {}), ("upstream", {"tf32": "upstream"}), ("all", {"tf32": "all"}),
                     ("bf16F", {"fisher_dtype": "bf16"}), ("bf16F_upstream", {"fisher_dtype": "bf16", "tf32": "upstream"})):
        ps = []
        for _ in range(L):
            for shape in [(H, H)] * 4 + [(F, H), (F, H), (H, F)]:
                ps.append(torch.nn.Parameter((torch.randn(*shape, generator=g) * 0.02).to(DEV)))
        opt = NaturalPionFast(ps, lr=0.2, row_blocks={id(p): 8 for p in ps[0::7]}, **kw)
        gen = torch.Generator(device=DEV).manual_seed(1)
        xa = torch.randn(N, H, device=DEV, generator=gen); xf = torch.randn(N, F, device=DEV, generator=gen)
        dH = torch.randn(N, H, device=DEV, generator=gen) / N; dF = torch.randn(N, F, device=DEV, generator=gen) / N
        ts = []
        for t in range(125):
            for i, p in enumerate(ps):
                k = i % 7
                opt.observe(p, xf if k == 6 else xa)
                opt.observe_backward(p, dF if k in (4, 5) else dH)
                p.grad = torch.randn(p.shape, device=DEV, generator=gen) * 1e-3
            torch.cuda.synchronize(); t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            if t >= 25:
                ts.append(time.perf_counter() - t0)
        OUT[f"sps_{mode}"] = sum(ts) / len(ts)
        print(f"{mode:15s} {OUT[f'sps_{mode}']:.4f} s per optimizer step (with refreshes, mean over 100)", flush=True)


if __name__ == "__main__":
    drift()
    timing()
    (ROOT / "runs-record" / "probes" / "tf32_check.json").write_text(json.dumps(OUT, indent=1))
