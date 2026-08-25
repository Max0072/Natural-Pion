"""The 1.076 s/step of job 246531, decomposed by measurement rather than by
subtraction.

`docs/JOURNAL.md` split that step into roughly 0.22 s of optimizer, 0.48 s of
model and a 0.38 s remainder attributed to `observe`. Only the first of those
was measured; the second was borrowed from a Pion run at a *different*
micro-batch, and the third was whatever was left over. That is how a 1.5x
prediction turned into a measured 2.3x, so this probe measures all three on the
same model, at the same micro-batch, in the same process.

The structure mirrors `harness/train.py`: `batch_sequences / micro_batch`
forward-backward passes, the covariance recorder enabled on the first of them
only, then one optimizer step.

Run:
    apptainer exec --nv $SIF python scripts/probes/observe_cost.py
"""

from __future__ import annotations

import argparse
import time

import torch

from harness.config import RunConfig
from harness.model import Transformer
from harness.train import build_optimizers


def timed(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--optimizer", default="ngd-pion")
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: run inside the container with --nv")
    device = torch.device("cuda")

    cfg = RunConfig(optimizer=args.optimizer, micro_batch=args.micro_batch)
    accum = cfg.batch_sequences // cfg.micro_batch
    print(f"device       {torch.cuda.get_device_name(0)}")
    print(f"optimizer    {cfg.optimizer}")
    print(f"micro-batch  {cfg.micro_batch}   accumulation {accum}   "
          f"tokens/step {cfg.batch_sequences * cfg.model.seq_len:,}")

    model = Transformer(cfg.model).to(device)
    rot, adamw, recorder = build_optimizers(model, cfg)

    tokens = torch.randint(
        0, cfg.model.vocab_size,
        (cfg.micro_batch, cfg.model.seq_len + 1), device=device,
    )
    x, y = tokens[:, :-1], tokens[:, 1:]

    def fwd_bwd(observe: bool) -> None:
        if recorder is not None:
            recorder.enabled = observe
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
            )
        loss.backward()
        model.zero_grad(set_to_none=False)

    with_obs = timed(lambda: fwd_bwd(True), warmup=3, iters=args.iters)
    without = timed(lambda: fwd_bwd(False), warmup=3, iters=args.iters)

    # one real optimizer step, with gradients already in place
    fwd_bwd(True)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    def opt_step() -> None:
        if rot is not None:
            rot.step()
        adamw.step()

    step_cost = timed(opt_step, warmup=2, iters=args.iters)

    model_only = without
    observe_only = with_obs - without
    total = model_only * accum + observe_only + step_cost

    print()
    print(f"{'component':<34}{'s':>10}{'% of step':>12}")
    print("-" * 56)
    for name, secs in (
        (f"model fwd+bwd, one micro-batch", model_only),
        (f"model fwd+bwd x {accum} (a whole step)", model_only * accum),
        ("observe (once per step)", observe_only),
        ("optimizer step", step_cost),
    ):
        print(f"{name:<34}{secs:>10.4f}{100*secs/total:>11.1f}%")
    print("-" * 56)
    print(f"{'predicted step':<34}{total:>10.4f}")
    print(f"{'tokens/s implied':<34}{cfg.batch_sequences*cfg.model.seq_len/total:>10,.0f}")
    print(f"{'peak GB':<34}{torch.cuda.max_memory_allocated()/2**30:>10.2f}")


if __name__ == "__main__":
    main()
