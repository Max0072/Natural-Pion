#!/usr/bin/env python
"""Check the machine can actually run this, before spending queue time on it.

Version numbers are not the check. `torch.linalg.eigh` on the GPU is at the
centre of the method -- it runs for every layer at every refactor -- and it is
exactly the sort of operation that is present, slow, or subtly wrong depending
on the CUDA build. Same for the `linalg.solve` inside Cayley. So the check is
to run them.

    apptainer exec --nv ngd-pion.sif python scripts/preflight.py
"""

from __future__ import annotations

import sys
import time

import torch


def ok(label: str, detail: str = "") -> None:
    print(f"  ok    {label}" + (f"  {detail}" if detail else ""))


def fail(label: str, detail: str) -> None:
    print(f"  FAIL  {label}  {detail}")


def main() -> int:
    failures = 0
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")

    if not torch.cuda.is_available():
        fail("cuda", "no device visible -- was --nv passed to apptainer?")
        return 1
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok("device", f"{name}, sm_{cap[0]}{cap[1]}, {total:.0f} GB")

    # The architecture has to be in the build, not merely newer than it.
    arches = torch.cuda.get_arch_list()
    tag = f"sm_{cap[0]}{cap[1]}"
    if tag in arches:
        ok("arch in build", tag)
    else:
        fail("arch in build", f"{tag} not among {arches} -- kernels will be JIT'd or fail")
        failures += 1

    try:
        a = torch.randn(4096, 4096, device=dev)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()
        flops = 10 * 2 * 4096**3 / (time.perf_counter() - t)
        ok("fp32 matmul", f"{flops/1e12:.0f} TFLOPS")
    except Exception as exc:  # noqa: BLE001
        fail("fp32 matmul", str(exc)[:80])
        failures += 1

    try:
        b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            b @ b
        torch.cuda.synchronize()
        flops = 10 * 2 * 4096**3 / (time.perf_counter() - t)
        ok("bf16 matmul", f"{flops/1e12:.0f} TFLOPS")
    except Exception as exc:  # noqa: BLE001
        fail("bf16 matmul", str(exc)[:80])
        failures += 1

    # The two operations the optimizer cannot do without.
    for size, label in ((512, "square layers"), (1376, "ffn down")):
        try:
            m = torch.randn(size, size, device=dev)
            m = m @ m.T / size + torch.eye(size, device=dev)
            torch.cuda.synchronize()
            t = time.perf_counter()
            w, _ = torch.linalg.eigh(m)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1e3
            if not torch.isfinite(w).all():
                raise RuntimeError("eigh returned non-finite eigenvalues")
            ok(f"eigh {size}", f"{ms:.0f} ms  ({label})")
        except Exception as exc:  # noqa: BLE001
            fail(f"eigh {size}", str(exc)[:80])
            failures += 1

    try:
        n = 512
        x = torch.randn(n, n, device=dev)
        x = 0.5 * (x - x.T)
        eye = torch.eye(n, device=dev)
        r = torch.linalg.solve(eye + 0.5 * x, eye - 0.5 * x)
        err = (r.T @ r - eye).abs().max().item()
        if err < 1e-3:
            ok("cayley solve", f"orthogonality error {err:.1e}")
        else:
            fail("cayley solve", f"orthogonality error {err:.1e} is too large")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        fail("cayley solve", str(exc)[:80])
        failures += 1

    try:
        from harness.model import ModelConfig, Transformer

        model = Transformer(ModelConfig()).to(dev)
        n = sum(p.numel() for p in model.parameters())
        idx = torch.randint(0, ModelConfig().vocab_size, (8, 256), device=dev)
        _, loss = model(idx, idx)
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        ok("model fwd+bwd", f"{n/1e6:.1f}M params, peak {peak:.1f} GB at 8 sequences")
    except Exception as exc:  # noqa: BLE001
        fail("model fwd+bwd", str(exc)[:120])
        failures += 1

    print("\nready" if not failures else f"\n{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
