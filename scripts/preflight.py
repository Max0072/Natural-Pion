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
from pathlib import Path

import torch

# Run as `python scripts/x.py`, so sys.path[0] is scripts/ and the repository
# root is not on it. Without this the harness import fails, and in preflight it
# fails inside a try that reports it as a failed GPU check.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



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

    # Every timing below warms up first. Without that the first call of each
    # kind pays cuBLAS or cuSOLVER's one-off setup and the number reported is
    # the setup, not the operation -- which is how an earlier run here made one
    # image look ten times slower at `eigh` than it is.
    for label, dtype in (("fp32 matmul", torch.float32), ("bf16 matmul", torch.bfloat16)):
        try:
            a = torch.randn(4096, 4096, device=dev, dtype=dtype)
            for _ in range(3):
                a @ a
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                a @ a
            torch.cuda.synchronize()
            flops = 10 * 2 * 4096**3 / (time.perf_counter() - t)
            ok(label, f"{flops/1e12:.0f} TFLOPS")
        except Exception as exc:  # noqa: BLE001
            fail(label, str(exc)[:80])
            failures += 1

    # The two operations the optimizer cannot do without.
    for size, label in ((512, "square layers"), (1376, "ffn down")):
        try:
            m = torch.randn(size, size, device=dev)
            m = m @ m.T / size + torch.eye(size, device=dev)
            for _ in range(3):
                w, _ = torch.linalg.eigh(m)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                w, _ = torch.linalg.eigh(m)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1e2
            if not torch.isfinite(w).all():
                raise RuntimeError("eigh returned non-finite eigenvalues")
            ok(f"eigh {size}", f"{ms:.1f} ms  ({label})")
        except Exception as exc:  # noqa: BLE001
            fail(f"eigh {size}", str(exc)[:80])
            failures += 1

    # The retraction as the project performs it, not as a bare solve. The
    # difference is the point: with this machine's default settings a raw
    # `linalg.solve` is done in TF32, ten bits of mantissa, and the error is a
    # thousand times worse. `ngd_pion.linalg.cayley` turns that off for itself,
    # and this checks the function the optimizer actually calls.
    try:
        from ngd_pion.linalg import cayley

        n = 512
        x = torch.randn(n, n, device=dev)
        x = 0.5 * (x - x.T)
        eye = torch.eye(n, device=dev)
        # The residual is formed in fp64. Measuring it in fp32 on this machine
        # means measuring it in TF32, and the instrument then reports its own
        # error rather than the retraction's: 4e-04 where the truth is 3e-06.
        def residual(R):
            d = R.double()
            return (d.T @ d - eye.double()).abs().max().item()

        r = cayley(x, 1.0)
        err = residual(r)
        raw_err = residual(torch.linalg.solve(eye + 0.5 * x, eye - 0.5 * x))
        if err < 1e-3:
            ok("cayley", f"orthogonality error {err:.1e}")
        else:
            fail("cayley", f"orthogonality error {err:.1e} is too large")
            failures += 1
        note = "matches" if raw_err < 10 * err else f"{raw_err/err:.0f}x worse"
        print(f"  info  unguarded solve here is {note} ({raw_err:.1e}) -- "
              f"tf32 matmul is {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}")
    except Exception as exc:  # noqa: BLE001
        fail("cayley", str(exc)[:80])
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
