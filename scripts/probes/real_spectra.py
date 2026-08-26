"""Can the spectral floor alone produce quad/curv = 1e32? Ask the real A.

quad is measured against the FLOORED operator, curv against the RAW one:
    quad = sum g^2 / lamt      curv = sum g^2 * lam / lamt^2
so curv/quad is bounded below by min_ij (lam_i+lam_j)/(lamt_i+lamt_j), taken
over the real spectrum of each layer. Computing that needs no gradient -- only
the covariance the run accumulated. CPU, fp64.
"""
import sys, glob
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.linalg import floor_eigenvalues

CKPT = "/onyx/data/p330/runs/qoc/ngd-pion-lr0.003-s0-235faf3851/checkpoint.pt"
EPS = 1e-4
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top-level keys:", list(sd)[:12])

# locate the optimizer state holding CovarianceAccumulator objects
opt = None
for k, v in sd.items():
    if isinstance(v, dict) and "state" in v and "param_groups" in v:
        if any("cov" in s for s in v["state"].values() if isinstance(s, dict)):
            opt, opt_key = v, k
            break
    if isinstance(v, dict) and any(
        isinstance(inner, dict) and "cov" in inner for inner in v.values()):
        opt, opt_key = {"state": v}, k
        break
print("optimizer state found under:", opt_key if opt else "NOT FOUND")
if opt is None:
    sys.exit(1)

print(f"\n{'layer':>5} {'n':>6} {'lam_max':>11} {'lam_min':>11} {'true cond':>11}"
      f" {'#floored':>9} {'logged cond':>11} {'min ratio':>11}")
print("-" * 92)
for i, (key, st) in enumerate(sorted(opt["state"].items(), key=lambda kv: str(kv[0]))):
    cov = st.get("cov") if isinstance(st, dict) else None
    if cov is None:
        continue
    A = cov.matrix.double()
    w = torch.linalg.eigvalsh(A).clamp_min(0.0)
    wt = floor_eigenvalues(w, EPS)
    n_floored = int((w < EPS * w.max()).sum())
    pos = w[w > w.max() * 1e-12]
    logged = float(w.max() / pos.min()) if pos.numel() else float("inf")
    true_cond = float(w.max() / w.min()) if float(w.min()) > 0 else float("inf")
    # min over pairs: both spectra are sorted ascending, so the min is at (0,0)
    ratio = float((w[0] + w[0]) / (wt[0] + wt[0]))
    print(f"{i:>5} {w.numel():>6} {float(w.max()):11.4e} {float(w.min()):11.4e}"
          f" {true_cond:11.4e} {n_floored:>9} {logged:11.4e} {ratio:11.4e}")
