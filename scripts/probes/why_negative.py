"""Which precision puts negative eigenvalues into X^T X?

The run shows |lam_min|/lam_max clustering around 5e-3 with 119-190 negative
eigenvalues per layer. Build X of EXACTLY known rank so the null directions
are truly zero, then see how far each arithmetic pushes them below zero.

  fp64            reference
  fp32            what CovarianceAccumulator claims to do
  bf16 -> fp32    activations quantised by autocast, gram still in fp32
"""
import sys, torch
torch.manual_seed(0)
torch.set_num_threads(8)

N, n, RANK = 16384, 256, 180         # scaled from 131072 x 512, rank ~0.7 n
print(f"X: {N} x {n}, exact rank {RANK}  ->  {n-RANK} truly-zero eigenvalues")
print(f"eps: fp64 {2**-53:.2e}   fp32 {2**-24:.2e}   bf16 {2**-8:.2e}\n")

# a realistic-ish anisotropic subspace: heavy leading direction, decaying tail
B = torch.randn(RANK, n, dtype=torch.float64)
B *= torch.logspace(0, -1.5, RANK, dtype=torch.float64).unsqueeze(1)
Z = torch.randn(N, RANK, dtype=torch.float64)
X64 = Z @ B

def report(tag, X, dt):
    g = (X.to(dt).transpose(0, 1) @ X.to(dt)) / N
    w = torch.linalg.eigvalsh(g.double())
    lmax, lmin = float(w.max()), float(w.min())
    nneg = int((w < 0).sum())
    print(f"  {tag:>28}: lam_max {lmax:10.4e}  lam_min {lmin:12.4e}"
          f"  |lam_min|/lam_max {abs(lmin)/lmax:9.3e}  #neg {nneg:>4}/{n}")

report("fp64 (reference)", X64, torch.float64)
report("fp32", X64, torch.float32)
X_bf = X64.to(torch.bfloat16).to(torch.float64)     # quantise the activations
report("bf16 activations, fp32 gram", X_bf, torch.float32)
report("bf16 activations, fp64 gram", X_bf, torch.float64)

print("\n  and the EMA on top: A <- 0.95 A + 0.05 gram, 120 steps, fp32")
A = None
for s in range(120):
    Zs = torch.randn(N, RANK, dtype=torch.float64)
    Xs = (Zs @ B).to(torch.bfloat16).to(torch.float32)
    g = (Xs.transpose(0, 1) @ Xs) / N
    A = g.clone() if A is None else A.mul_(0.95).add_(g, alpha=0.05)
    if s in (0, 19, 59, 119):
        w = torch.linalg.eigvalsh(A.double())
        print(f"    step {s:>4}: lam_max {float(w.max()):10.4e}"
              f"  lam_min {float(w.min()):12.4e}"
              f"  ratio {abs(float(w.min()))/float(w.max()):9.3e}"
              f"  #neg {int((w<0).sum()):>4}/{n}")
