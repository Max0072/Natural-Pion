import torch
from ngd_pion.linalg import cayley_newton_schulz, cayley, skew
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

n = 64
X = skew(torch.randn(n, n, dtype=torch.float64))
X = X / torch.linalg.matrix_norm(X, 2)          # unit norm -> ||A|| = angle/2
I = torch.eye(n, dtype=torch.float64)

print("orthogonality error ||R^T R - I||_inf, fp64 so the iteration is not")
print("hidden by the rounding floor.  Predicted: residual ~ ||A||^(2^(k+1))\n")
print(f"{'angle':>8}{'||A||':>9}" + "".join(f"{'NS '+str(k):>12}{'pred':>11}" for k in (1,2,3)))
print("-" * 76)
for angle in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
    a = angle / 2
    row = f"{angle:>8.2f}{a:>9.3f}"
    for k in (1, 2, 3):
        R = cayley_newton_schulz(X, angle, k)
        err = float((R.T @ R - I).abs().max())
        row += f"{err:>12.1e}{a**(2**(k+1)):>11.1e}"
    print(row)

print("\nthe exact solve for reference:")
for angle in (0.02, 0.1, 0.5, 1.2):
    R = cayley(X, angle)
    print(f"  angle {angle:>4}: fp64 {float((R.T@R-I).abs().max()):.1e}", end="")
    Rf = cayley(X.float(), angle).double()
    print(f"   fp32 {float((Rf.T@Rf-torch.eye(n,dtype=torch.float64)).abs().max()):.1e}")

print("\nthreshold needed for NS-k to stay under the fp32 solve's own 1.1e-6:")
for k in (1, 2, 3):
    print(f"  NS {k}: ||A|| < {1.1e-6 ** (1/2**(k+1)):.3f}  "
          f"(angle < {2*1.1e-6 ** (1/2**(k+1)):.3f} rad)")
