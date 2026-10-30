"""Where does the variation of log(alpha) live: layer, basis age, or training time?
(journal 2026-10-30.) CPU only; reads the runs' existing diagnostics.jsonl.

Candidates for what alpha really contributes: a per-layer calibration (layer
effect, stable across seeds and over training), a sawtooth in the rate (age
effect), or a schedule (time effect). Least squares on one-hot factors:
    R2 alone   the factor by itself
    unique     R2(all three) - R2(without the factor)
Whatever is left over is step-to-step variation these three cannot express, so a
frozen profile could not reproduce it. Candidate "smoothing" (the basis frozen
against D's noise) concerns the direction, not alpha, and is not tested here.

    python scripts/probes/alpha_variance.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

T_FAC = 25
BIN = 250


def load(run):
    rows = [json.loads(x) for x in open(Path(run) / "diagnostics.jsonl")]
    return [(r["name"], r["step"], float(r["alpha"])) for r in rows if r["alpha"] == r["alpha"]]


def onehot(labels):
    _, inv = np.unique(labels, return_inverse=True)
    return np.eye(inv.max() + 1)[inv]


def r2(y, xs):
    X = np.hstack(xs)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)


def decompose(rows, lo):
    rows = [r for r in rows if r[1] > lo and r[2] > 0]
    y = np.log([r[2] for r in rows])
    L = onehot([r[0] for r in rows])
    A = onehot([(r[1] - 1) % T_FAC for r in rows])
    T = onehot([r[1] // BIN for r in rows])
    full = r2(y, [L, A, T])
    print(f"  steps>{lo}: n={len(y)} sd(log alpha)={y.std():.3f} "
          f"alpha>=0.999: {np.mean(y > np.log(0.999)):.3f}")
    print(f"    R2 alone  layer {r2(y, [L]):.3f}  age {r2(y, [A]):.3f}  time {r2(y, [T]):.3f}")
    print(f"    unique    layer {full - r2(y, [A, T]):.3f}  age {full - r2(y, [L, T]):.3f}  "
          f"time {full - r2(y, [L, A]):.3f}  | all three {full:.3f}  left over {1 - full:.3f}")


def profile(rows, lo, hi=10**9):
    d = {}
    for n, s, a in rows:
        if lo < s <= hi and a > 0:
            d.setdefault(n, []).append(np.log(a))
    return {n: float(np.mean(v)) for n, v in d.items()}


def corr(p, q):
    k = sorted(set(p) & set(q))
    return float(np.corrcoef([p[n] for n in k], [q[n] for n in k])[0, 1])


def main():
    runs = sys.argv[1:]
    data = {r: load(r) for r in runs}
    for r, rows in data.items():
        print(Path(r).name)
        for lo in (100, 1000):
            decompose(rows, lo)

    rows = data[runs[0]]
    print("\nmedian alpha by age, steps>1000, first run")
    by = {}
    for n, s, a in rows:
        if s > 1000:
            by.setdefault((s - 1) % T_FAC, []).append(a)
    print("  " + " ".join(f"{k}:{np.median(v):.2f}" for k, v in sorted(by.items())))

    print("\nmedian alpha by layer type, steps>1000, first run")
    by = {}
    for n, s, a in rows:
        if s > 1000:
            by.setdefault(n.split(".")[-1], []).append(a)
    for k, v in sorted(by.items()):
        print(f"  {k:12s} n={len(v):5d} median {np.median(v):.3f}  p10 {np.quantile(v,.1):.3f}  p90 {np.quantile(v,.9):.3f}")

    print("\nmedian alpha by time, first run")
    by = {}
    for n, s, a in rows:
        by.setdefault(s // 500 * 500, []).append(a)
    print("  " + " ".join(f"{k}:{np.median(v):.3f}" for k, v in sorted(by.items())))

    print("\nis the per-layer profile (mean log alpha, steps>1000) a stable thing?")
    ps = [profile(data[r], 1000) for r in runs]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            print(f"  seed-to-seed r = {corr(ps[i], ps[j]):.3f}")
    for r in runs:
        print(f"  steps 1000-2000 vs 2000-3000, one run: r = "
              f"{corr(profile(data[r], 1000, 2000), profile(data[r], 2000)):.3f}")
    v = np.array(list(ps[0].values()))
    print(f"  spread of the profile: exp(p10..p90) = {np.exp(np.quantile(v,.1)):.3f}..{np.exp(np.quantile(v,.9)):.3f}, "
          f"max/min {np.exp(v.max()-v.min()):.1f}x over {len(v)} layers")


if __name__ == "__main__":
    main()
