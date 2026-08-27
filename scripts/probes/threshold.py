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
R = torch.randn(1376, 1376); X0 = (R - R.T)
X0 = X0 / torch.linalg.matrix_norm(X0, 2)          # unit spectral norm

print("fp32 tiny =", torch.finfo(torch.float32).tiny)
print(f"\n{'||X||_2':>10} {'old sigma':>14} {'new sigma':>14} {'old v norm':>11}  verdict")
for e in range(0, 26, 2):
    scale = 10.0 ** (-e)
    X = X0 * scale
    so, vo = old(X, 50)
    sn, vn = spectral_norm(X, 50)
    for _ in range(5):
        so, vo = old(X, 2, vo)
        sn, vn = spectral_norm(X, 2, vn)
    dead = float(so) == 0.0
    print(f"{scale:10.0e} {float(so):14.6e} {float(sn):14.6e} {float(vo.norm()):11.4f}"
          f"  {'DEAD' if dead else ''}")
