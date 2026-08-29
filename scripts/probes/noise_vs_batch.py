"""Does the step's signal fraction rise with batch size, and how fast?

`split_half_step.py` measured one number on the real model: the step at 512
sequences is **96.4% sampling noise**. That number is the best single
explanation this project has for why natural gradient does not deliver its
promised speedup here. `F^-1` lengthens the step in the directions of least
curvature -- that is the whole idea -- but those are exactly the directions the
data says least about, so the method takes its longest steps where its estimate
is worst. Amari's advantage is stated for the *deterministic* gradient flow,
and at 96% noise we are nowhere near that regime.

If that reading is right it makes a sharp prediction: the advantage should
appear as the noise falls, and the cheapest way to lower the noise is a bigger
batch. This probe measures the noise-versus-batch curve directly, before any
training run is spent on the question.

METHOD, unchanged from `split_half_step.py` except for the accumulation. One
batch of `B` sequences, split in half; both halves pushed through the **same
frozen bases** and the same `A`, `D` from the checkpoint, so the only thing
differing between them is the gradient. With `X1`, `X2` two independent
estimates of the same mean `mu`,

    d = X1 - X2                    Var(a half, B/2 seqs) = |d|^2 / 2
    m = (X1 + X2) / 2              |mu|^2 = |m|^2 - |d|^2 / 4

and `m` is itself the estimate at the full `B`, with `Var(B) = |d|^2 / 4`. So
one draw at `B` gives the signal fraction at `B` directly.

Halves beyond a few hundred sequences do not fit on one card, so each half is
accumulated over micro-batches: the gradient of the mean loss over `H`
sequences is the mean of the per-micro-batch gradients when the micro-batches
are equal, which is what dividing each backward by their count achieves.

Variance falls as `1/B` for an unbiased mean, so the signal fraction should
follow `s(B) = mu2 / (mu2 + c/B)`. The fitted `c` says how much batch is needed
for any target, and that is the number the decision needs.

    python scripts/probes/noise_vs_batch.py <run_dir> [B1,B2,...]
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
from harness.config import RunConfig  # noqa: E402
from harness.data import TokenCorpus  # noqa: E402
from harness.model import ModelConfig, Transformer  # noqa: E402
from harness.train import build_optimizers  # noqa: E402
from ngd_pion.direction import generators, natural_gradient  # noqa: E402

run = sys.argv[1]
BATCHES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [512, 1024, 2048, 4096]
MICRO = 256
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
print(f"{os.path.basename(run)}   step {state['step']}   {dev}")
print(f"micro-batch {MICRO}, batches {BATCHES}\n")

# The recorder would fold these activations into `A`. It must not: the bases are
# the checkpoint's and stay frozen, or the halves would differ in their metric
# as well as their gradient and the measurement would isolate nothing.
if recorder is not None:
    recorder.remove()

data = TokenCorpus(cfg.data_path, mcfg.seq_len, seed=cfg.seed + 1)
params = [p for p in rot.param_groups[0]["params"] if rot.state.get(p, {}).get("bases")]


def grads_for(x, y):
    """Gradient of the mean loss over all of `(x, y)`, accumulated in chunks."""
    model.zero_grad(set_to_none=True)
    n = max(1, -(-x.shape[0] // MICRO))
    for i in range(0, x.shape[0], MICRO):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            _, loss = model(x[i : i + MICRO], y[i : i + MICRO])
        (loss / n).backward()
    return {id(p): p.grad.detach().float().clone() for p in params if p.grad is not None}


def step_of(g):
    """The preconditioned step this gradient would produce, as one flat vector."""
    out = []
    for p in params:
        if id(p) not in g:
            continue
        bi, bo = rot.state[p]["bases"]
        G_in, G_out = generators(p.detach().float(), g[id(p)])
        out.append(natural_gradient(G_in, bi).reshape(-1))
        out.append(natural_gradient(G_out, bo).reshape(-1))
    return torch.cat(out)


print(f"{'B (seqs)':>9} {'tokens':>10} {'cos(X1,X2)':>11} {'signal frac':>12} {'noise frac':>11}")
print("-" * 58)
fit = []
for B in BATCHES:
    x, y = data.batch(B, device=dev)
    h = B // 2
    x1 = step_of(grads_for(x[:h], y[:h]))
    x2 = step_of(grads_for(x[h:], y[h:]))
    d, m = x1 - x2, 0.5 * (x1 + x2)
    var_half = float(d.pow(2).sum()) / 2.0        # variance of one half (B/2 seqs)
    var_full = var_half / 2.0                     # variance at the full B
    mu2 = max(float(m.pow(2).sum()) - var_full, 0.0)
    cos = float(torch.dot(x1, x2) / (x1.norm() * x2.norm() + 1e-30))
    sig = mu2 / (mu2 + var_full) if mu2 + var_full > 0 else float("nan")
    print(f"{B:>9} {B * mcfg.seq_len:>10} {cos:>11.4f} {sig:>12.4f} {1 - sig:>11.4f}")
    fit.append((B, mu2, var_full))
    del x, y, x1, x2, d, m
    torch.cuda.empty_cache()

print()
mu2 = sum(f[1] for f in fit) / len(fit)
c = sum(f[2] * f[0] for f in fit) / len(fit)
print(f"fit  s(B) = mu2 / (mu2 + c/B)   with mu2 = {mu2:.6g}, c = {c:.6g}")
for target in (0.10, 0.25, 0.50):
    need = c * target / (mu2 * (1 - target))
    print(f"  signal fraction {target:.0%} needs B = {need:,.0f} sequences"
          f"  ({need / 512:.0f}x the current batch)")
