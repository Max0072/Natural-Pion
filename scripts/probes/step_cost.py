"""Where an NGD-Pion step actually spends its time.

`docs/JOURNAL.md` records that NGD-Pion runs 158x slower than Pion and names
`optimizer.py:236` -- two SVDs per weight per step, for a diagnostic logged
every fiftieth -- as the suspect. That entry says plainly it is "a hypothesis
with an arithmetic argument behind it, not a measurement". This is the
measurement.

It times each stage of `_apply` separately on the real shapes of LLaMA-60M, so
the answer is a ranking rather than a single number, and it also times the
things that would become the bottleneck once the SVD is gone: the unbatched
Python loop over 56 parameters, the host syncs, the refactorisation, and the
cost of holding TF32 off across the whole step rather than only across Cayley.

Run:
    apptainer exec --nv $SIF python scripts/probes/step_cost.py
"""

from __future__ import annotations

import argparse
import time

import torch

from ngd_pion.direction import fisher_apply, generators, natural_gradient
from ngd_pion.factorization import build_bases
from ngd_pion.linalg import cayley, exact_fp32

# LLaMA-60M as harness/model.py builds it: hidden 512, 8 layers, ffn 1376.
SHAPES = [((512, 512), 32), ((1376, 512), 16), ((512, 1376), 8)]


def timed(fn, *, warmup: int = 3, iters: int = 10) -> float:
    """Seconds per call, with the GPU actually drained before the clock stops."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def make(shape, device, dt):
    m, n = shape
    W = torch.linalg.qr(torch.randn(max(m, n), min(m, n), device=device, dtype=dt))[0]
    W = W if W.shape == (m, n) else W.T.contiguous()
    G = torch.randn(m, n, device=device, dtype=dt) * 1e-2
    X = torch.randn(n, device=device, dtype=dt)
    A = torch.randn(4096, n, device=device, dtype=dt)
    A = (A.T @ A) / 4096 + 1e-3 * torch.eye(n, device=device, dtype=dt)
    return W, G, A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: run inside the container with --nv")
    device = torch.device("cuda")
    dt = torch.float32
    print(f"device      {torch.cuda.get_device_name(0)}")
    print(f"torch       {torch.__version__}\n")

    rows: list[tuple[str, str, int, float]] = []

    for shape, count in SHAPES:
        m, n = shape
        W, G, A = make(shape, device, dt)

        with exact_fp32():
            basis_in, basis_out = build_bases(W, A, args.eps)
            G_in, G_out = generators(W, G)
            X_in = natural_gradient(G_in, basis_in)
            X_out = natural_gradient(G_out, basis_out)
            gram_in = W.T @ W
            gram_out = W @ A @ W.T
            eye_out = torch.eye(m, device=device, dtype=dt)

            def stage(name, fn):
                rows.append((f"{m}x{n}", name, count, timed(fn)))

            stage("generators", lambda: generators(W, G))
            stage("natural_gradient x2", lambda: (
                natural_gradient(G_in, basis_in), natural_gradient(G_out, basis_out)))
            stage("grams (W^T W, W A W^T)", lambda: (W.T @ W, W @ A @ W.T))
            stage("curv: fisher_apply x2", lambda: (
                (X_in * fisher_apply(A, gram_in, X_in)).sum()
                + (X_out * fisher_apply(eye_out, gram_out, X_out)).sum()))
            stage("curv: contracted x2", lambda: (
                4.0 * ((A @ X_in) * (X_in @ gram_in)).sum()
                + 4.0 * ((eye_out @ X_out) * (X_out @ gram_out)).sum()))
            stage("SVD (matrix_norm 2) x2", lambda: (
                torch.maximum(torch.linalg.matrix_norm(X_in, 2),
                              torch.linalg.matrix_norm(X_out, 2))))
            stage("power iter x2 (5 it)", lambda: (
                _power(X_in, 5), _power(X_out, 5)))
            stage("cayley x2", lambda: (cayley(X_in, 1e-3), cayley(X_out, 1e-3)))
            stage("retraction apply", lambda: cayley(X_out, 1e-3) @ W @ cayley(X_in, 1e-3))
            stage("host sync (float x2)", lambda: (float(X_in.sum()), float(X_out.sum())))
            stage("refactor (build_bases)", lambda: build_bases(W, A, args.eps))

        # the same generator work with TF32 left on, to price `exact_fp32`
        rows.append((f"{m}x{n}", "natural_gradient x2 [TF32]", count, timed(
            lambda: (natural_gradient(G_in, basis_in),
                     natural_gradient(G_out, basis_out)))))

    print(f"{'shape':>10} {'stage':<28} {'n':>4} {'ms/call':>9} {'ms/step':>9}")
    print("-" * 66)
    per_step: dict[str, float] = {}
    for shape, name, count, secs in rows:
        total = secs * count * 1e3
        per_step[name] = per_step.get(name, 0.0) + total
        print(f"{shape:>10} {name:<28} {count:>4} {secs*1e3:>9.3f} {total:>9.2f}")

    print("\nper optimizer step, summed over all 56 weights:")
    for name, total in sorted(per_step.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32} {total:>9.2f} ms")


def _power(X: torch.Tensor, iters: int) -> torch.Tensor:
    """Top singular value by power iteration on X^T X -- the cheap stand-in."""
    v = torch.randn(X.shape[-1], device=X.device, dtype=X.dtype)
    v /= v.norm()
    for _ in range(iters):
        v = X.T @ (X @ v)
        v /= v.norm().clamp_min(1e-30)
    return (X @ v).norm()


if __name__ == "__main__":
    main()
