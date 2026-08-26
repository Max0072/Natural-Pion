"""What is the covariance actually, and what does its spectrum look like at zero?"""
import sys, glob, os
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch

ck = glob.glob("/onyx/data/p330/runs/floor2/ngd-pion-lr0.003-*/checkpoint.pt")[0]
sd = torch.load(ck, map_location="cpu", weights_only=False)
st = sd["rot"]["state"] if "state" in sd["rot"] else sd["rot"]
items = sorted(st.items(), key=lambda kv: str(kv[0]))

for idx in (0, 2, 3, 29):
    k, s = items[idx]
    A = s["cov"].matrix
    acc = s["cov"]
    asym = float((A - A.transpose(-1, -2)).abs().max() / A.abs().max())
    A64 = A.double()
    w = torch.linalg.eigvalsh(A64)
    tr = float(A64.trace())
    print(f"\n===== layer {idx}  n={A.shape[0]} =====")
    print(f"  dtype {A.dtype}   count {acc.count}   beta {acc.beta}")
    print(f"  max|A - A^T| / max|A| = {asym:.3e}   trace {tr:.4e}   lam_max {float(w.max()):.4e}")
    print(f"  diag: min {float(A64.diag().min()):.4e}  max {float(A64.diag().max()):.4e}"
          f"   (a covariance diagonal must be >= 0)")
    print(f"  10 smallest eigenvalues: " + "  ".join(f"{float(v):+.4e}" for v in w[:10]))
    print(f"  around zero, sorted:     " + "  ".join(f"{float(v):+.3e}" for v in w[
        max(0, int((w < 0).sum()) - 4): int((w < 0).sum()) + 4]))
    print(f"  10 largest:              " + "  ".join(f"{float(v):.4e}" for v in w[-10:]))
    neg = w[w < 0]
    print(f"  negatives: {neg.numel()}  sum {float(neg.sum()):+.4e}"
          f"  mean {float(neg.mean()) if neg.numel() else 0:+.4e}")
