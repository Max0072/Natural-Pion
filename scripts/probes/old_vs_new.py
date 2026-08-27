import sys; sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.linalg import spectral_norm

def old(X, iters, v=None):
    tiny = torch.finfo(X.dtype).tiny
    if v is None:
        v = torch.ones(X.shape[-1], dtype=X.dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
    Xt = X.transpose(-1, -2)
    for _ in range(iters):
        v = (Xt @ (X @ v.unsqueeze(-1))).squeeze(-1)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
    return (X @ v.unsqueeze(-1)).squeeze(-1).norm(dim=-1), v

torch.manual_seed(0)
R = torch.randn(256, 256); X0 = R - R.T
print(f"{'||X||':>10} {'old sigma':>14} {'new sigma':>14} {'exact':>14}")
for scale in (1e-1, 1e-4, 1e-8, 1e-10, 1e-12):
    X = X0 * scale
    ex = float(torch.linalg.matrix_norm(X, 2))
    so, vo = old(X, 50); sn, vn = spectral_norm(X, 50)
    for _ in range(5):
        so, vo = old(X, 2, vo); sn, vn = spectral_norm(X, 2, vn)
    print(f"{scale:10.0e} {float(so):14.6e} {float(sn):14.6e} {ex:14.6e}")
