"""Is the *direction* noisy, given that `A` and `D` are already time-averaged?

The objection this answers: `A` and `D` are EMAs (`beta = 0.95`,
`beta_backward = 0.5`), so the method already carries memory across steps --
is that not already a kind of momentum?

The distinction is where the memory sits.

    now         X = EMA(F)^-1 G      the *metric* is averaged
    momentum    X = F^-1 EMA(G)      the *direction* is averaged

Averaging the denominator stabilises the geometry; it cannot cancel zero-mean
noise in the numerator, because it enters multiplicatively. `docs/JOURNAL.md`
states the same thing from the other side: "switching to the true Fisher
changes the curvature estimate but does nothing to the numerator, which is
where the problem lives".

`ALGORITHM.md` already carries a split-half number -- the step disagreeing by
47% between two independent samples -- but that was measured at `d_out = 256`
on roughly 80k tokens, and the same section flags the sample-size dependence as
unverified. The real run accumulates over 131072 tokens a step. So it is
re-measured here, on a checkpoint of the real model.

METHOD. One batch, split in half. Both halves are pushed through the **same
frozen bases** and the same `A`, `D` restored from the checkpoint, so the only
thing that differs between them is the gradient. Any disagreement is numerator
noise, which is exactly the part `S` cannot reach and momentum can.

With `X1`, `X2` two independent estimates of the same mean `mu`,

    d = X1 - X2      m = (X1 + X2) / 2
    Var(one half) = |d|^2 / 2          |mu|^2 = |m|^2 - |d|^2 / 4

and the signal fraction of a single half's step is `|mu|^2 / (|mu|^2 + Var)`.
A half carries 256 sequences against the real step's 512, so the real step's
variance is half of it; both are reported.

Averaging `k` independent draws divides the variance by `k`, and an EMA with
factor `beta` is worth `k = (1 + beta) / (1 - beta)`. That converts the
measurement directly into a predicted gain from momentum, which is the number
the decision needs.

    python scripts/probes/split_half_step.py <run_dir> [beta]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/onyx/data/p330/Natural-Pion")

import torch

from dataclasses import replace
from harness.config import ModelConfig, RunConfig
from harness.data import TokenCorpus
from harness.model import Transformer
from harness.train import build_optimizers
from ngd_pion.direction import generators, natural_gradient

run = sys.argv[1]
beta_m = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg_d = json.load(open(os.path.join(run, "manifest.json")))["config"]
mcfg = ModelConfig(**cfg_d["model"])
cfg = RunConfig(**{k: v for k, v in cfg_d.items() if k != "model"}, model=mcfg)

torch.manual_seed(cfg.seed)
model = Transformer(mcfg).to(dev)
rot, adamw, recorder = build_optimizers(model, cfg)

state = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
rot.load_state_dict(state["rot"])
step = state["step"]
print(f"{os.path.basename(run)}   step {step}   {dev}\n")

names = {id(m.weight): n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
data = TokenCorpus(cfg.data_path, mcfg.seq_len, seed=cfg.seed + 1)


def grads_for(x: torch.Tensor, y: torch.Tensor) -> dict[int, torch.Tensor]:
    """Gradients of the mean loss on one half, without touching optimizer state."""
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        _, loss = model(x, y)
    loss.backward()
    return {id(p): p.grad.detach().float().clone()
            for p in rot.param_groups[0]["params"] if p.grad is not None}


half = cfg.batch_sequences // 2
inputs, targets = data.batch(cfg.batch_sequences, device=dev)
# The recorder, if any, would fold these activations into `A` and `D`. It must
# not: the bases are the checkpoint's and stay frozen, or the two halves would
# differ in their metric as well as their gradient and the measurement would
# not isolate anything.
if recorder is not None:
    recorder.remove()

g1 = grads_for(inputs[:half], targets[:half])
g2 = grads_for(inputs[half:], targets[half:])

print(f"  {'layer':<24}{'cos(X1,X2)':>12}{'signal frac':>13}{'at 512 seq':>12}")
print("  " + "-" * 61)
rows = []
for p in rot.param_groups[0]["params"]:
    st = rot.state.get(p)
    if not st or "bases" not in st or id(p) not in g1:
        continue
    W = p.detach().float()
    bi, bo = st["bases"]
    Xs = []
    for g in (g1, g2):
        G_in, G_out = generators(W, g[id(p)])
        Xs.append((natural_gradient(G_in, bi), natural_gradient(G_out, bo)))
    # treat the two sides as one vector
    def flat(t):
        return torch.cat([t[0].reshape(-1), t[1].reshape(-1)])
    x1, x2 = flat(Xs[0]), flat(Xs[1])
    d = x1 - x2
    m = 0.5 * (x1 + x2)
    var_half = float(d.pow(2).sum()) / 2.0
    mu2 = max(float(m.pow(2).sum()) - var_half / 2.0, 0.0)
    cos = float(torch.dot(x1, x2) / (x1.norm() * x2.norm() + 1e-30))
    sig_half = mu2 / (mu2 + var_half) if mu2 + var_half > 0 else float("nan")
    sig_full = mu2 / (mu2 + var_half / 2.0) if mu2 + var_half > 0 else float("nan")
    rows.append((names.get(id(p), "?"), cos, sig_half, sig_full, mu2, var_half))

for n, c, sh, sf, _, _ in rows[:8]:
    print(f"  {n:<24}{c:>12.4f}{sh:>13.4f}{sf:>12.4f}")
print("  ...")
for n, c, sh, sf, _, _ in rows[-3:]:
    print(f"  {n:<24}{c:>12.4f}{sh:>13.4f}{sf:>12.4f}")

mu2 = sum(r[4] for r in rows)
var_half = sum(r[5] for r in rows)
var_full = var_half / 2.0
k = (1.0 + beta_m) / (1.0 - beta_m)
print(f"\n  over all {len(rows)} weights, pooled:")
print(f"    signal fraction of one real step (512 seq):  {mu2 / (mu2 + var_full):.4f}")
print(f"    i.e. the step is {100 * var_full / (mu2 + var_full):.1f}% sampling noise")
print(f"\n  a momentum EMA with beta = {beta_m} averages k = {k:.1f} draws:")
print(f"    signal fraction would become {mu2 / (mu2 + var_full / k):.4f}")
print(f"\n  For reference, ALGORITHM.md records 47% noise, measured at d_out = 256")
print(f"  on ~80k tokens, and flags the sample-size dependence as unverified.")
