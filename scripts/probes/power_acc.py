import torch
torch.manual_seed(0)

def skew(M): return 0.5*(M - M.transpose(-1,-2))

def power(X, iters, v=None):
    n = X.shape[-1]
    if v is None:
        v = torch.ones(n, dtype=X.dtype)
    v = v / v.norm()
    for _ in range(iters):
        v = X.T @ (X @ v)
        v = v / v.norm().clamp_min(1e-30)
    return (X @ v).norm(), v

def make(kind, n=512):
    if kind == "random":
        return skew(torch.randn(n, n, dtype=torch.float64))
    if kind == "decaying":
        # skew with a fast-decaying spectrum, closer to a preconditioned step
        M = skew(torch.randn(n, n, dtype=torch.float64))
        U, s, Vh = torch.linalg.svd(M)
        s = s[0] * torch.exp(-torch.arange(n, dtype=torch.float64) / 40)
        return skew(U @ torch.diag(s) @ Vh)
    raise ValueError(kind)

for kind in ("random", "decaying"):
    X = make(kind)
    true = torch.linalg.matrix_norm(X, 2)
    fro  = torch.linalg.matrix_norm(X, "fro")
    print(f"\n--- {kind}: sigma_max={true:.6f}  ||X||_F={fro:.4f}  F/2={fro/true:.2f}x ---")
    print("  cold start (v = ones):")
    for it in (1, 3, 5, 10, 20, 50):
        est, _ = power(X, it)
        print(f"    {it:3d} iters  est={est:.6f}  rel err={(true-est)/true:.3e}")

    # warm start: perturb X slightly, reuse the converged vector -- what a run does
    _, v = power(X, 50)
    Xp = X + 1e-3 * skew(torch.randn(512, 512, dtype=torch.float64)) * X.abs().mean()
    truep = torch.linalg.matrix_norm(Xp, 2)
    print("  warm start (converged v, X perturbed by 1e-3 relative):")
    for it in (1, 2, 3, 5):
        est, _ = power(Xp, it, v)
        print(f"    {it:3d} iters  est={est:.6f}  rel err={(truep-est)/truep:.3e}")
