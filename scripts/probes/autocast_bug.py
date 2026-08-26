"""Does autocast turn the covariance into bf16 despite exact_fp32()?"""
import sys
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.covariance import CovarianceAccumulator
from ngd_pion.linalg import exact_fp32

torch.manual_seed(0)
x = torch.randn(4096, 128)                    # fp32 activations
print(f"input x dtype: {x.dtype}")

print("\n--- the matmul on its own ---")
g = (x.transpose(0, 1) @ x) / x.shape[0]
print(f"  outside autocast:              {g.dtype}")
with exact_fp32():
    g = (x.transpose(0, 1) @ x) / x.shape[0]
print(f"  inside exact_fp32():           {g.dtype}")
with torch.autocast("cpu", dtype=torch.bfloat16):
    with exact_fp32():
        g = (x.transpose(0, 1) @ x) / x.shape[0]
print(f"  inside autocast + exact_fp32:  {g.dtype}   <-- exact_fp32 does not stop autocast")

print("\n--- the accumulator, which is how it actually runs ---")
acc = CovarianceAccumulator(beta=0.95)
print(f"  requested dtype (the check in __init__ passes): {acc.dtype}")
acc.observe(x)
print(f"  after observe() outside autocast:  {acc.matrix.dtype}")

acc2 = CovarianceAccumulator(beta=0.95)
with torch.autocast("cpu", dtype=torch.bfloat16):
    acc2.observe(x)                            # exactly what the forward hook does
    for _ in range(20):
        acc2.observe(torch.randn(4096, 128))
print(f"  after observe() inside autocast:   {acc2.matrix.dtype}   <-- the bug")

print("\n--- consequence for positive definiteness ---")
for tag, a in (("fp32 (outside autocast)", acc), ("bf16 (inside autocast)", acc2)):
    w = torch.linalg.eigvalsh(a.matrix.double())
    lo, hi = float(w.min()), float(w.max())
    print(f"  {tag:>26}: lam_max {hi:9.4e}  lam_min {lo:+11.4e}"
          f"  ratio {abs(lo)/hi:9.3e}  #neg {int((w<0).sum()):>4}/{w.numel()}")
