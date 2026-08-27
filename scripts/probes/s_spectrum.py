"""Is the near-degeneracy that inflates the step in S, not in A?

A and E[dd^T] both sit in the checkpoint, so this is free: no GPU, no job.
For every layer report cond and scale of both factors, and rank them against
the angle that layer actually took.
"""
import json, math, os, sys
import torch

RUN = '/onyx/data/p330/runs/mcfisher2/ngd-pion-s-lr0.01-s0-1974d5fe08'
ETA = 0.01
torch.set_num_threads(4)

ck = torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location='cpu',
                mmap=True, weights_only=False)
state = ck['rot']['state']
model = ck['model']

rows = [json.loads(l) for l in open(os.path.join(RUN, 'diagnostics.jsonl'))]
last = max(r['step'] for r in rows)
diag = [r for r in rows if r['step'] == last]

# the optimizer iterates params in registration order, which is the order the
# diagnostics rows are emitted in; check that before trusting the pairing
idx = sorted(state.keys())
if len(idx) != len(diag):
    sys.exit(f"state has {len(idx)} entries, diagnostics {len(diag)}")


def mat(acc):
    m = getattr(acc, '_matrix', None)
    return None if m is None else m.double()


bad = 0
for i, r in zip(idx, diag):
    a = mat(state[i]['cov'])
    if a is not None and tuple(a.shape) != (r['shape'][1],) * 2:
        bad += 1
if bad:
    sys.exit(f"pairing is wrong: {bad} shape mismatches -- do not trust this")
print(f"pairing checked: {len(idx)} layers, all A shapes match diagnostics\n")


def spec(m):
    if m is None:
        return None
    w = torch.linalg.eigvalsh(0.5 * (m + m.T))
    hi = float(w.max())
    pos = w[w > 0]
    lo = float(pos.min()) if pos.numel() else 0.0
    return hi, lo, (hi / lo if lo > 0 else float('inf')), float((w < hi * 1e-7).float().mean())


out = []
for i, r in zip(idx, diag):
    s = state[i]
    A = spec(mat(s['cov']))
    D = spec(mat(s['cov_backward']))
    if A is None or D is None:
        continue
    W = model[r['name'] + '.weight'].double()
    S = W.T @ mat(s['cov_backward']) @ W
    Sp = spec(S)
    out.append(dict(name=r['name'], angle=r['angle'] / ETA,
                    condA=A[2], maxA=A[0], condD=D[2], maxD=D[0],
                    condS=Sp[2], maxS=Sp[0], nullS=Sp[3]))

print(f"{'layer':<24} {'angle@eta1':>10} {'cond_A':>10} {'cond_S':>11} "
      f"{'lam_max_A':>10} {'lam_max_S':>11} {'nullfrac_S':>10}")
for r in sorted(out, key=lambda r: -r['angle'])[:8]:
    print(f"{r['name']:<24} {r['angle']:>10.1f} {r['condA']:>10.1f} "
          f"{r['condS']:>11.3e} {r['maxA']:>10.3f} {r['maxS']:>11.3e} "
          f"{r['nullS']:>10.3f}")
print("  ...")
for r in sorted(out, key=lambda r: -r['angle'])[-8:]:
    print(f"{r['name']:<24} {r['angle']:>10.1f} {r['condA']:>10.1f} "
          f"{r['condS']:>11.3e} {r['maxA']:>10.3f} {r['maxS']:>11.3e} "
          f"{r['nullS']:>10.3f}")


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x == x and y == y and abs(x) != float('inf')]
    n = len(pairs)
    if n < 3:
        return float('nan'), n
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i]); rk = [0.0] * n
        for p, i in enumerate(o):
            rk[i] = p
        return rk
    rx, ry = rank(xs), rank(ys)
    m = (n - 1) / 2
    num = sum((a - m) * (b - m) for a, b in zip(rx, ry))
    d = sum((a - m) ** 2 for a in rx)
    return (num / d if d else float('nan')), n


ang = [r['angle'] for r in out]
print(f"\n=== what the angle tracks, step {last} (n={len(out)}) ===")
for k in ['condA', 'condS', 'condD', 'maxA', 'maxS', 'maxD', 'nullS']:
    rho, n = spearman([r[k] for r in out], ang)
    print(f"  rho(angle, {k:<7}) = {rho:>7.3f}   (n={n})")

print("\n  a degeneracy account, relocated to S, needs rho(angle, cond_S) strongly")
print("  positive and rho(angle, lam_max_S) strongly negative.")
inv = [1.0 / math.sqrt(r['maxS'] * r['maxA']) for r in out]
rho, n = spearman(inv, ang)
print(f"\n  rho(angle, 1/sqrt(lam_max_S * lam_max_A)) = {rho:.3f}"
      "   <- the scale account: step ~ 1/curvature scale")
