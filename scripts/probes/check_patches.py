import sys
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.linalg import spectral_norm
from ngd_pion.factorization import basis_identity_anchor
from ngd_pion.direction import generators
from ngd_pion.fast import _floor_share

torch.manual_seed(0)

print("=== 1. spectral_norm: the absorbing zero ===")
n = 256
R = torch.randn(n, n); X = (R - R.T)
exact = float(torch.linalg.matrix_norm(X, 2))
for scale in (1e0, 1e-6, 1e-12, 1e-18):
    Xs = X * scale
    v = None
    sig = None
    for step in range(6):                    # warm start, then reuse -- as _apply does
        sig, v = spectral_norm(Xs, 50 if step == 0 else 2, v)
    rel = abs(float(sig) - exact * scale) / (exact * scale)
    print(f"  ||X|| scale {scale:8.0e}: sigma={float(sig):.6e}"
          f"  exact={exact*scale:.6e}  rel err={rel:.2e}"
          f"  v_norm={float(v.norm()):.3f}")

print("\n=== 2. spectral_norm: a deliberately dead cached vector recovers ===")
dead = torch.zeros(n)
sig, v = spectral_norm(X, 50, dead)
print(f"  fed v=0: sigma={float(sig):.6e}  exact={exact:.6e}"
      f"  rel err={abs(float(sig)-exact)/exact:.2e}")

print("\n=== 3. floor_share: 0 when nothing is floored, 1 when the spectrum dies ===")
for orders, tag in ((3, "no floor active (3 decades, eps=1e-4)"),
                    (10, "floor active (10 decades)"),
                    (None, "rank-deficient (exact zeros), as the real A is")):
    n2, B = 128, 2048
    if orders is None:
        lam = torch.cat([torch.logspace(0, -2, n2 // 2), torch.zeros(n2 // 2)])
    else:
        lam = torch.logspace(0, -orders, n2)
    Q = torch.linalg.qr(torch.randn(n2, n2))[0]
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    W = torch.linalg.qr(torch.randn(n2, n2))[0]
    # G from tokens that do NOT live in A's range -- as in the run, where A is
    # an EMA over one micro-batch and G is the full current batch
    Xs = torch.randn(B, n2)
    G = (torch.randn(B, n2).T @ Xs) / B
    G_in, _ = generators(W, G)
    b = basis_identity_anchor(A, 1e-4)
    n_fl = int((lam < lam.max() * 1e-4).sum())
    print(f"  {tag:>44}: #floored={n_fl:>4}/{n2}  floor_share={_floor_share(G_in, b, 1e-4):.4f}")
