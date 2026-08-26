import sys; sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.fast import FastNGDPion

torch.manual_seed(0)
def build(angle_max, lr):
    ps = []
    for _ in range(4):
        w = torch.linalg.qr(torch.randn(96, 64))[0]
        p = torch.nn.Parameter(w.contiguous()); p.grad = torch.randn(96, 64) * 1e-2
        ps.append(p)
    o = FastNGDPion(ps, lr=lr, t_fac=100, angle_max=angle_max)
    for p in ps:
        X = torch.randn(2048, 64)
        o.set_covariance(p, (X.T @ X) / 2048)
    return o, ps

CAP = 0.1
for lr in (3e-3, 0.5, 1.0):
    for cap in (0.0, CAP):
        o, ps = build(cap, lr)
        seen = []
        for s in range(6):
            for p in ps:
                p.grad = torch.randn(*p.shape) * 1e-2
            o.step()
            seen += [float(o.state[p]["angle"]) for p in ps]
        req = max(float(o.state[p].get("angle_requested", 0)) for p in ps)
        tag = f"cap {'off' if not cap else CAP}"
        print(f"  lr {lr:<6g} {tag:>8}: max angle applied {max(seen):9.4f}"
              f"   last requested {req:9.4f}"
              f"   {'VIOLATES CAP' if cap and max(seen) > cap * 1.001 else 'ok'}")
