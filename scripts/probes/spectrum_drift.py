"""Does the singular spectrum of each weight stay near its initialisation?

The premise behind excluding the embedding and the head from Pion is that a
spectrum-preserving optimizer can only own a weight whose spectrum does not
need to move. That cannot be tested under Pion, which freezes the spectrum by
construction -- it needs an optimizer that leaves the spectrum free, i.e. AdamW.

Initialisation is `normal(0, init_std)` under `torch.manual_seed(cfg.seed)`, so
it is reproducible exactly rather than needing to have been saved.
"""
import sys, glob, os, json
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from harness.config import RunConfig, ModelConfig
from harness.model import Transformer

run = sorted(glob.glob("/onyx/data/p330/runs/spectrum/adamw-*"))[0]
man = json.load(open(os.path.join(run, "manifest.json")))
cfg_d = man["config"]
mcfg = ModelConfig(**cfg_d["model"])
seed = cfg_d["seed"]

torch.manual_seed(seed)                      # exactly as harness.train does
init = Transformer(mcfg)
final = Transformer(mcfg)
sd = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=False)
final.load_state_dict(sd["model"])
print(f"{run.split('/')[-1]}   step {sd['step']}   seed {seed}\n")

names_i = dict(init.named_parameters())
names_f = dict(final.named_parameters())
owned = {id(m.weight) for m in init.parameter_split()[0]}

rows = []
for n, p0 in names_i.items():
    if p0.dim() != 2:
        continue
    p1 = names_f[n]
    s0 = torch.linalg.svdvals(p0.double())
    s1 = torch.linalg.svdvals(p1.double())
    rel = float((s1 - s0).norm() / s0.norm())
    top = float(s1[0] / s0[0])
    # how much of the change is a uniform rescale rather than a change of shape
    scale = float((s0 @ s1) / (s0 @ s0))
    shape = float((s1 - scale * s0).norm() / s0.norm())
    rows.append((n, tuple(p0.shape), id(p0) in owned, rel, top, scale, shape))

print(f"  {'weight':<28} {'shape':>13} {'pion':>5} {'||ds||/||s||':>12}"
      f" {'s_max ratio':>11} {'best scale':>10} {'shape drift':>11}")
print("  " + "-" * 96)
for n, sh, own, rel, top, scale, shape in rows:
    print(f"  {n:<28} {str(sh):>13} {'yes' if own else 'NO':>5} {rel:12.4f}"
          f" {top:11.4f} {scale:10.4f} {shape:11.4f}")

inner = [r for r in rows if r[2]]
outer = [r for r in rows if not r[2]]
def avg(rs, i): return sum(r[i] for r in rs) / max(len(rs), 1)
print()
print(f"  Pion-owned  (n={len(inner):>2}): mean ||ds||/||s|| {avg(inner,3):.4f}"
      f"   mean shape drift {avg(inner,6):.4f}")
print(f"  excluded    (n={len(outer):>2}): mean ||ds||/||s|| {avg(outer,3):.4f}"
      f"   mean shape drift {avg(outer,6):.4f}")
print()
print("  'shape drift' removes the best uniform rescale first, so it isolates the")
print("  part of the change a spectrum-preserving optimizer genuinely cannot make.")
