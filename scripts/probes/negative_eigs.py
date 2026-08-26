"""Does the accumulated covariance actually carry negative eigenvalues?

curv = <X, F_raw(X)> is provably >= 0 for PSD factors, and it is measured at
-1.3 in the working run, so the factors cannot be PSD. `floor_eigenvalues`
clamps at 0 before building the basis, and `harness.instrument._spectrum`
clamps before reporting, so neither can show this. Read A straight off the
checkpoint and do not clamp.
"""
import sys
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch, glob, os

for tag, pat in (("eta 3e-3 (working)", "/onyx/data/p330/runs/floor2/ngd-pion-lr0.003-*"),
                 ("eta 1.0  (broken)",  "/onyx/data/p330/runs/floor2/ngd-pion-lr1-*")):
    ck = glob.glob(os.path.join(pat, "checkpoint.pt"))
    if not ck:
        print(f"{tag}: no checkpoint"); continue
    sd = torch.load(ck[0], map_location="cpu", weights_only=False)
    st = sd["rot"]["state"] if "state" in sd["rot"] else sd["rot"]
    print(f"\n===== {tag} =====  step {sd.get('step')}")
    print(f"{'layer':>5} {'n':>6} {'lam_max':>11} {'lam_min RAW':>13} {'#negative':>10}"
          f" {'most negative':>14} {'|neg|/lam_max':>14}")
    worst = []
    for i, (k, s) in enumerate(sorted(st.items(), key=lambda kv: str(kv[0]))):
        cov = s.get("cov") if isinstance(s, dict) else None
        if cov is None:
            continue
        w = torch.linalg.eigvalsh(cov.matrix.double())        # NO clamp
        lam_max, lam_min = float(w.max()), float(w.min())
        n_neg = int((w < 0).sum())
        worst.append((lam_min / lam_max, i, w.numel(), lam_max, lam_min, n_neg))
        if i < 6 or n_neg > 0 and i % 12 == 0:
            print(f"{i:>5} {w.numel():>6} {lam_max:11.4e} {lam_min:13.5e} {n_neg:>10}"
                  f" {lam_min:14.5e} {abs(lam_min)/lam_max:14.3e}")
    worst.sort()
    r, i, n, lmax, lmin, nneg = worst[0]
    tot_neg = sum(w[5] for w in worst)
    print(f"  --- {len(worst)} layers, {tot_neg} negative eigenvalues in total")
    print(f"  --- most negative: layer {i} (n={n}), lam_min={lmin:.5e}, "
          f"lam_max={lmax:.4e}, ratio={r:.3e}")
