import math, torch
from ngd_pion.linalg import cayley, cayley_newton_schulz, skew

import os
N, STEPS, POOL = int(os.environ.get("N",128)), int(os.environ.get("STEPS",4000)), 64
ckpt = [c for c in (50,200,500,1000,2000,4000) if c <= STEPS] or [STEPS]

g = torch.Generator().manual_seed(1)
pool = []
for _ in range(POOL):                       # normalise once, reuse -- the SVD here
    X = skew(torch.randn(N, N, generator=g, dtype=torch.float64))
    pool.append(X / torch.linalg.matrix_norm(X, 2))

def drift(W0, angle, method, dt):
    W = W0.to(dt).clone()
    s0 = torch.linalg.svdvals(W0.double())
    P = [x.to(dt) for x in pool]
    out = {}
    for step in range(1, STEPS + 1):
        Xo, Xi = P[(2*step) % POOL], P[(2*step + 1) % POOL]
        W = method(Xo, angle) @ W @ method(Xi, angle)
        if step in ckpt:
            s = torch.linalg.svdvals(W.double())
            out[step] = float(((s - s0).abs() / s0).max())
    return out

methods = [("fp64 solve", torch.float64, lambda X,c: cayley(X,c)),
           ("fp32 solve", torch.float32, lambda X,c: cayley(X,c)),
           ("fp32 NS 1", torch.float32, lambda X,c: cayley_newton_schulz(X,c,1)),
           ("fp32 NS 2", torch.float32, lambda X,c: cayley_newton_schulz(X,c,2)),
           ("fp32 NS 3", torch.float32, lambda X,c: cayley_newton_schulz(X,c,3))]

gw = torch.Generator().manual_seed(0)
W0 = torch.randn(N, N, generator=gw, dtype=torch.float64) * 0.02

for angle in (1e-2, 1e-1):
    print(f"\n=== W {N}x{N}, angle {angle:g} per side per step, {STEPS} steps ===")
    print(f"{'method':<12}" + "".join(f"{c:>11}" for c in ckpt)
          + f"{'exponent':>10}{'@73242':>11}")
    for name, dt, fn in methods:
        d = drift(W0, angle, fn, dt)
        p = math.log(d[ckpt[-1]]/d[ckpt[-2]]) / math.log(ckpt[-1]/ckpt[-2]) if d[ckpt[-2]] > 0 else float("nan")
        extrap = d[ckpt[-1]] * (73242/ckpt[-1]) ** p
        print(f"{name:<12}" + "".join(f"{d[c]:>11.2e}" for c in ckpt)
              + f"{p:>10.2f}{extrap:>11.2e}")
