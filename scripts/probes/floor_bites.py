"""Is `F^-1` still a preconditioner after the floor, or has it become a scalar?

The question behind this is the user's: natural gradient is supposed to
converge much faster than ordinary methods, and here it does not obviously do
so. One way that happens without any bug being visible is if the spectral floor
swallows the spectrum. `natural_gradient` divides elementwise by
`denominator = 2(lam_i + lam_j)` with `lam` already floored at `eps * lam_max`,
so if most of `lam` sits on the floor the denominator is nearly constant, the
division is nearly a scalar multiply, and what the method is really running is
the *raw* gradient on the rotation group with an expensive no-op attached.

Three numbers per layer per side, from the covariances the full-length run
actually accumulated:

  floored     fraction of `lam` raised by the floor. High means the floor, not
              the data, is setting the geometry.
  spread      `max(d)/min(d)` over the denominator. This is the whole dynamic
              range the preconditioner has to work with. 1 is a scalar.
  iso-cos     cosine between `F^-1 G` and `G` for isotropic `G`, in closed form
              `mean(1/d) / sqrt(mean(1/d^2))`. Exactly 1 iff `d` is constant,
              so it reads "how much does preconditioning rotate a typical
              direction". This is an operator property and does not depend on
              the real gradient, which the checkpoint does not carry.

Swept over `eps`, because `eps = 1e-4` is the only value any of the 163 `ngd`
runs in this project has ever used and nothing establishes that it is right.

CPU, fp64. Reads a finished run's checkpoint; changes nothing.
"""
import sys

import torch

sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
from ngd_pion.factorization import basis_congruence  # noqa: E402
from ngd_pion.linalg import floor_eigenvalues  # noqa: E402

CKPT = "/onyx/data/p330/runs/full/ngd-pion-s-lr0.01-s0-9314af4d3e/checkpoint.pt"
EPS_GRID = [1e-2, 1e-4, 1e-6, 1e-8, 0.0]

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
opt = None
for k, v in sd.items():
    if isinstance(v, dict) and "state" in v and "param_groups" in v:
        if any(isinstance(s, dict) and "cov" in s for s in v["state"].values()):
            opt = v
            break
if opt is None:
    sys.exit("no optimizer state with covariances in this checkpoint")

params = sd.get("model") or sd.get("model_state") or sd.get("state_dict")
shapes = {}
if isinstance(params, dict):
    for name, t in params.items():
        if isinstance(t, torch.Tensor) and t.dim() == 2:
            shapes.setdefault(tuple(t.shape), []).append(name)


def summarise(lam, eps):
    """`floored`, `spread`, `iso-cos` for one floored spectrum."""
    lam = lam.clamp_min(0.0)
    n_floored = int((lam < eps * lam.max()).sum()) if eps else 0
    lamt = floor_eigenvalues(lam, eps) if eps else lam.clamp_min(lam.max() * 1e-30)
    d = 2.0 * (lamt.unsqueeze(-1) + lamt.unsqueeze(-2))
    d = d[torch.triu(torch.ones_like(d), diagonal=1) > 0]
    inv = 1.0 / d
    cos = float(inv.mean() / inv.pow(2).mean().sqrt())
    return n_floored / lam.numel(), float(d.max() / d.min()), cos


rows = []
for key, st in sorted(opt["state"].items(), key=lambda kv: str(kv[0])):
    if not isinstance(st, dict) or "cov" not in st or "cov_backward" not in st:
        continue
    A = st["cov"].matrix.double()
    D = st["cov_backward"].matrix.double()
    rows.append((str(key), A, D))

print(f"{len(rows)} layers with both covariances\n")
print(f"{'eps':>8} {'side':>5} {'floored':>9} {'spread':>12} {'iso-cos':>9}"
      f" {'floored':>9} {'spread':>12} {'iso-cos':>9}")
print(f"{'':>8} {'':>5} {'--- median over layers ---':^32} {'--- worst layer ---':^32}")
print("-" * 82)
for eps in EPS_GRID:
    for name, get in (("A", lambda A, D: A), ("D", lambda A, D: D)):
        f, s, c = [], [], []
        for _, A, D in rows:
            M = get(A, D)
            lam = torch.linalg.eigvalsh(M)
            a, b, d = summarise(lam, eps)
            f.append(a); s.append(b); c.append(d)
        f.sort(); s.sort(); c.sort()
        mid = len(f) // 2
        print(f"{eps:>8.0e} {name:>5} {f[mid]:>9.3f} {s[mid]:>12.3e} {c[mid]:>9.4f}"
              f" {f[-1]:>9.3f} {s[-1]:>12.3e} {c[0]:>9.4f}")
