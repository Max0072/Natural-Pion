"""Time FastNGDPion._apply across the real layer shapes. CPU, no cluster.

Run as:  bench_step.py <repo_root> <label> [diag_every]
The repo root is put first on sys.path so the same script can time the old
checkout and the current tree in separate processes.
"""
import sys, time
ROOT, LABEL = sys.argv[1], sys.argv[2]
DIAG = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.path.insert(0, ROOT)
import torch
from ngd_pion.fast import FastNGDPion

torch.manual_seed(0)
torch.set_num_threads(8)

# LLaMA-60M: per block, wq/wk/wv/wo are 512x512 and the MLP touches 1376.
SHAPES = []
for _ in range(8):
    SHAPES += [(512, 512)] * 4 + [(1376, 512), (512, 1376), (1376, 512)]
SHAPES = SHAPES[:56]

params = []
for (o, i) in SHAPES:
    w = torch.linalg.qr(torch.randn(max(o, i), min(o, i)))[0]
    w = w.T.contiguous() if o < i else w
    p = torch.nn.Parameter(w[:o, :i].contiguous())
    p.grad = torch.randn(o, i) * 1e-3
    params.append(p)

kw = dict(lr=3e-3, t_fac=100)
try:
    opt = FastNGDPion(params, diag_every=DIAG, **kw)
    has_diag = True
except TypeError:
    opt = FastNGDPion(params, **kw)          # old checkout has no diag_every
    has_diag = False

for p in params:                              # A from real-ish samples
    n = p.shape[1]
    X = torch.randn(4096, n)
    opt.set_covariance(p, (X.T @ X) / 4096)

STEPS = 6
times = []
for s in range(STEPS):
    for p in params:
        p.grad = torch.randn(*p.shape) * 1e-3
    t0 = time.perf_counter()
    opt.step()
    times.append(time.perf_counter() - t0)

first, rest = times[0], times[1:]
print(f"{LABEL:>34} | diag_every={DIAG if has_diag else 'n/a':>6} | "
      f"step 0 {first:6.2f} s | steps 1-{STEPS-1} mean {sum(rest)/len(rest):6.3f} s | "
      f"min {min(rest):6.3f} max {max(rest):6.3f}")
