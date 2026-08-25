#!/usr/bin/env python
"""Check on a card what the CPU test suite cannot see.

The suite runs on CPU, and on CPU a checkpoint loaded with `map_location="cpu"`
and parameters that are already there cannot disagree about device. On a GPU
they can. `torch.optim.Optimizer.load_state_dict` relocates the tensors it
finds -- values that are tensors, and tensors inside dicts, lists and tuples --
and `NGDPion` keeps neither: it keeps a `CovarianceAccumulator` and two `Basis`
dataclasses, which torch does not descend into.

So this runs the sequence a requeued 24-hour job performs: train a few steps,
checkpoint, resume, train a few more. Everything it exercises is invisible to
`pytest`, and the first thing that would have found it otherwise is a 9.6B run
a day into counting.

    apptainer exec --nv <image> python scripts/gpu_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Run as `python scripts/x.py`, so sys.path[0] is scripts/ and the repository
# root is not on it. Without this the harness import fails, and in preflight it
# fails inside a try that reports it as a failed GPU check.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import RunConfig
from harness.model import ModelConfig
from harness.train import train


def spectrum_drift(device: str, steps: int = 50) -> float:
    """Relative movement of the singular values over a short run.

    Pion freezes the spectrum of every 2-D weight for the whole of training,
    and NGD-Pion keeps that property with an exact retraction; if this number
    is not tiny, the property is gone and with it the reason to prefer Cayley
    over a truncated exponential.
    """
    from ngd_pion import NGDPion, attach

    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 256, bias=False).to(device)
    opt = NGDPion([layer.weight], lr=1e-2, t_fac=10)
    recorder = attach([layer], opt)
    before = torch.linalg.svdvals(layer.weight.detach().double())
    for _ in range(steps):
        x = torch.randn(64, 256, device=device)
        opt.zero_grad(set_to_none=True)
        layer(x).pow(2).sum().backward()
        opt.step()
    after = torch.linalg.svdvals(layer.weight.detach().double())
    recorder.remove()
    return ((after - before).abs() / before).max().item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("  FAIL  no cuda device -- was --nv passed to apptainer?")
        return 1
    if args.device.startswith("cuda"):
        print(f"device  {torch.cuda.get_device_name(0)}")

    tmp = Path(tempfile.mkdtemp(prefix="gpu-smoke-"))
    small = ModelConfig(vocab_size=512, hidden=64, layers=2, heads=4,
                        ffn_hidden=128, seq_len=32)
    rng = np.random.default_rng(0)
    for name in ("train", "val"):
        rng.integers(0, small.vocab_size, size=200_000, dtype=np.uint16).tofile(tmp / f"{name}.bin")

    cfg = RunConfig(
        optimizer="ngd-pion", model=small, batch_sequences=8, micro_batch=8,
        train_steps=8, ngd_t_fac=2, eval_every=2, eval_batches=1, log_every=1,
        data_path=str(tmp / "train.bin"), val_path=str(tmp / "val.bin"),
        out_dir=str(tmp / "runs"),
    )

    failures = 0
    try:
        out = train(cfg, device=args.device, max_steps=4)
        print("  ok    four steps and a checkpoint")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  first leg: {type(exc).__name__}: {exc}")
        return 1

    try:
        out = train(cfg, device=args.device, max_steps=8)
        rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        resumed = [r for r in rows if r.get("event") == "resume"]
        steps = [r["step"] for r in rows if "train_loss" in r]
        if not resumed:
            print("  FAIL  the second leg did not resume; it started over")
            failures += 1
        elif max(steps) != 7:
            print(f"  FAIL  resumed run reached step {max(steps)}, expected 7")
            failures += 1
        else:
            print(f"  ok    resumed from step {resumed[-1]['from_step']} and finished at 7")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  resume: {type(exc).__name__}: {exc}")
        failures += 1

    # The property the method is built on, checked where it can actually break.
    # TF32 is switched on first, deliberately: the package has to defend the
    # spectrum against a hostile environment, not merely inside a tidy one.
    if args.device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            drift = spectrum_drift(args.device)
            if drift < 1e-2:
                print(f"  ok    spectrum held under TF32-on: relative drift {drift:.1e}")
            else:
                print(f"  FAIL  spectrum moved {drift:.1e} -- the guard is not holding")
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  spectrum check: {type(exc).__name__}: {exc}")
            failures += 1

    print("\nready" if not failures else f"\n{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
