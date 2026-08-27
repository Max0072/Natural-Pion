import torch
torch.manual_seed(0)

def skew(M): return 0.5*(M - M.transpose(-1,-2))

def cayley_solve(X, c, dt):
    n = X.shape[-1]
    I = torch.eye(n, dtype=dt)
    A = (0.5*c) * X
    return torch.linalg.solve(I + A, I - A)

def cayley_ns(X, c, iters, dt):
    """Cayley = 2(I+A)^-1 - I, with the inverse by Newton-Schulz.
    Z <- Z(2I - MZ); starting at Z0 = I - A gives residual A^2, which squares."""
    n = X.shape[-1]
    I = torch.eye(n, dtype=dt)
    A = (0.5*c) * X
    M = I + A
    Z = I - A                      # residual R0 = A^2
    for _ in range(iters):
        Z = Z @ (2.0*I - M @ Z)
    return 2.0*Z - I

n = 512
X = skew(torch.randn(n, n, dtype=torch.float64))
X = X / torch.linalg.matrix_norm(X, 2)     # unit spectral norm, so c IS the angle

print(f"{'angle c':>9}{'||A||':>9}   " + "".join(f"{'NS '+str(k):>12}" for k in (0,1,2,3))
      + f"{'solve fp32':>13}")
print("-" * 78)
for c in (1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0):
    exact = cayley_solve(X, c, torch.float64)
    row = f"{c:>9.0e}{0.5*c:>9.1e}   "
    for k in (0, 1, 2, 3):
        R = cayley_ns(X.float(), c, k, torch.float32).double()
        err = (R.T @ R - torch.eye(n, dtype=torch.float64)).abs().max()
        row += f"{err:>12.1e}"
    S = cayley_solve(X.float(), c, torch.float32).double()
    row += f"{(S.T @ S - torch.eye(n, dtype=torch.float64)).abs().max():>13.1e}"
    print(row)

print("\northogonality error above; NS k = number of Newton-Schulz iterations")
print("cost: solve ~2.7n^3 (latency-bound LU); NS k iterations = 4k*n^3 of matmul")
