"""Does a rotational optimizer actually freeze the spectrum, on real hardware?

`tests/test_shampoo.py::test_spectrum_is_preserved` pins this in fp64 on a
toy. That is not the same claim: the run trains in bf16 with an fp32 optimizer
on a card where fp32 matmul means TF32 unless disabled, over 150 two-sided
Cayley steps. `linalg.exact_fp32` exists precisely because leaving TF32 on
moves the singular values by a relative 1.0 over 200 steps. Whether the guard
holds in a real run is a measurement, not an inference.

Pion-owned weights are touched by the rotational optimizer alone -- AdamW never
sees them and never decays them -- so in exact arithmetic their singular values
are unchanged from initialisation. Initialisation is reproducible from the seed
rather than needing to have been saved.

    python scripts/probes/spectrum_preserved.py <run_dir>
"""
import sys, os, json
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from harness.config import ModelConfig
from harness.model import Transformer

run = sys.argv[1]
man = json.load(open(os.path.join(run, "manifest.json")))
cfg_d = man["config"]
mcfg = ModelConfig(**cfg_d["model"])
seed = cfg_d["seed"]

torch.manual_seed(seed)                      # exactly as harness.train does
init = Transformer(mcfg)
final = Transformer(mcfg)
sd = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=False)
final.load_state_dict(sd["model"])
print(f"{os.path.basename(run)}   optimizer {cfg_d['optimizer']}   step {sd['step']}   seed {seed}\n")

names_i = dict(init.named_parameters())
names_f = dict(final.named_parameters())
owned = {id(m.weight) for m in init.parameter_split()[0]}

rel_all, orth_all = [], []
worst = None
for n, p0 in names_i.items():
    if p0.dim() != 2 or id(p0) not in owned:
        continue
    p1 = names_f[n]
    s0 = torch.linalg.svdvals(p0.double())
    s1 = torch.linalg.svdvals(p1.double())
    rel = float((s1 - s0).norm() / s0.norm())
    rel_all.append((rel, n, tuple(p0.shape), float(s1[0] / s0[0]), float(s1[-1] / s0[-1])))
    if worst is None or rel > worst[0]:
        worst = (rel, n)

rel_all.sort(reverse=True)
print(f"  {'weight':<26} {'shape':>13} {'||ds||/||s||':>13} {'s_max ratio':>12} {'s_min ratio':>12}")
print("  " + "-" * 80)
for rel, n, sh, top, bot in rel_all[:6]:
    print(f"  {n:<26} {str(sh):>13} {rel:13.3e} {top:12.6f} {bot:12.6f}")
print("  ...")
for rel, n, sh, top, bot in rel_all[-3:]:
    print(f"  {n:<26} {str(sh):>13} {rel:13.3e} {top:12.6f} {bot:12.6f}")

vals = [r[0] for r in rel_all]
print(f"\n  {len(vals)} Pion-owned weights")
print(f"  relative spectrum drift:  median {sorted(vals)[len(vals)//2]:.3e}   max {max(vals):.3e}")
print(f"\n  For reference, from docs/JOURNAL.md: TF32 left on moves the spectrum by")
print(f"  a relative 1.0 over 200 steps; the exact-fp32 solve gives 2.6e-04.")
