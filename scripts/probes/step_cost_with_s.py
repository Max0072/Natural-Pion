"""What measuring `S` costs, timed away from the data loader.

End-to-end `s/step` is not a portable number on this cluster. One step reads
512 scattered windows from a 20 GB memmap, so it is bound by page-cache state
far more often than by arithmetic: measured at 0.85 s/step on one node, 1.42 on
another with no co-tenant on either, and 1.26 for two concurrent jobs that were
in fact both waiting on the same pages. Four attempts to compare the two
variants that way returned "identical", every time because the difference sat
under an I/O ceiling.

So time the optimizer step on its own, with the GPU actually drained before the
clock stops, on the real shapes of LLaMA-60M. That number is a property of the
algorithm.

Run:
    apptainer exec --nv $SIF python scripts/probes/step_cost_with_s.py
"""

from __future__ import annotations

import os
import time

import torch

from ngd_pion.fast import FastNGDPion
from ngd_pion.with_s_fast import FastNGDPionS

# hidden 512, 8 layers, ffn 1376: 32 square, 16 tall, 8 wide.
SHAPES = [((512, 512), 32), ((1376, 512), 16), ((512, 1376), 8)]
TOKENS = 512 * 256          # one micro-batch, which is one whole step here


def timed(fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build(cls, device, **kw):
    params, mats, by_shape = [], [], {}
    for (m, n), count in SHAPES:
        by_shape[(m, n)] = len(params)
        for _ in range(count):
            w = torch.linalg.qr(torch.randn(max(m, n), min(m, n), device=device))[0]
            w = w.T.contiguous() if m < n else w
            p = torch.nn.Parameter(w[:m, :n].contiguous())
            p.grad = torch.randn(m, n, device=device) * 1e-3
            params.append(p)
            mats.append((m, n))
    opt = cls(params, lr=1e-3, t_fac=int(os.environ.get("TFAC", 100)), **kw)
    for p, (m, n) in zip(params, mats):
        x = torch.randn(4096, n, device=device)
        opt.set_covariance(p, (x.T @ x) / 4096)
        if isinstance(opt, FastNGDPionS):
            d = torch.randn(4096, m, device=device)
            opt.observe_backward(p, d)
    return opt, params, mats, by_shape


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("needs a GPU: the point is the cost on the card the runs use")
    print(f"  {torch.cuda.get_device_name(0)}, {len(SHAPES)} shape groups, 56 weights\n")

    rows = []
    for name, cls, kw in (("S = I", FastNGDPion, {}),
                          ("S measured", FastNGDPionS, {"beta_backward": 0.5})):
        torch.manual_seed(0)
        opt, params, mats, by_shape = build(cls, device, **kw)
        opt.step()                                   # force the first factorisation

        def one_step():
            for p in params:
                p.grad.normal_()
            opt.step()

        step = timed(one_step)

        # the covariance accumulations, at the real token count
        # one representative parameter per shape group -- an accumulator's
        # matrix is sized by the first tensor it saw, so it must be fed a
        # tensor of the same width
        acc = 0.0
        for (m, n), count in SHAPES:
            rep = params[by_shape[(m, n)]]
            x = torch.randn(TOKENS, n, device=device)
            acc += count * timed(lambda r=rep, t=x: opt._accumulator(r).observe(t), iters=3)
            del x
            torch.cuda.empty_cache()
        back = 0.0
        if isinstance(opt, FastNGDPionS):
            for (m, n), count in SHAPES:
                rep = params[by_shape[(m, n)]]
                d = torch.randn(TOKENS, m, device=device)
                back += count * timed(
                    lambda r=rep, t=d: opt._backward_accumulator(r).observe(t, scale=1.0),
                    iters=3,
                )
                del d
                torch.cuda.empty_cache()
        rows.append((name, step, acc, back))
        del opt, params
        torch.cuda.empty_cache()

    print(f"  {'variant':<12}{'opt.step()':>12}{'A accum':>10}{'D accum':>10}{'total':>10}")
    print("  " + "-" * 54)
    for name, step, acc, back in rows:
        print(f"  {name:<12}{step*1000:11.1f}ms{acc*1000:9.1f}ms{back*1000:9.1f}ms"
              f"{(step+acc+back)*1000:9.1f}ms")
    a = sum(rows[0][1:])
    b = sum(rows[1][1:])
    print(f"\n  measuring S costs x{b/a:.2f} of the optimizer's own work")
    print("  (against 0.46 s/step for Pion end to end, which builds no covariance at all)")


if __name__ == "__main__":
    main()
