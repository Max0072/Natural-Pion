"""Does alpha react to the individual step? (journal 2026-10-30.) CPU, existing diagnostics.

Frozen tables of alpha failed, so the live part matters. Hypothesis: alpha shortens the
steps that are long before it is applied. sigma = spectral norm of X = angle / (rot_t * alpha),
the step's length before alpha. All quantities are demeaned within layer x 250-step bin,
so layer and schedule are out and only step-to-step variation is left.

Prediction: slope of log(alpha) on log(sigma) well below 0 (-1 would make the angle
constant), and sd(log(alpha sigma)) < sd(log sigma). A slope near 0 refutes it. The
logged angle is not used for the sd: it also carries the cosine's change within a bin.
Clamped rows (alpha >= 0.999) cannot respond, so unclamped rows are shown too.

    python scripts/probes/alpha_response.py <run_dir> [...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STEPS, FLOOR, BIN = 3000, 1e-5, 250


def within(keys, x):
    out = np.array(x, dtype=float)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    for idx in groups.values():
        out[idx] -= out[idx].mean()
    return out


def report(rows, peak, label):
    keys, la, ls, lq, lc, ff = [], [], [], [], [], []
    for r in rows:
        t, a, ang = r["step"] - 1, float(r["alpha"]), float(r["angle"])
        if t < 100 or not (a > 0 and ang > 0):
            continue
        rate = FLOOR + 0.5 * (peak - FLOOR) * (1 + math.cos(math.pi * t / STEPS))
        keys.append((r["name"], t // BIN))
        la.append(math.log(a)); ls.append(math.log(ang / (rate * a)))
        lq.append(math.log(max(r["quad"], 1e-30))); lc.append(math.log(max(r["curv"], 1e-30)))
        ff.append(max(r["floored_frac_in"], r["floored_frac_out"]))
    a_, s, q, c, f = (within(keys, v) for v in (la, ls, lq, lc, ff))
    n = len(a_)
    slope = (a_ @ s) / (s @ s)
    print(f"{label}: n={n} clamped {np.mean(np.array(la) > math.log(.999)):.2f}")
    print(f"  all        slope {slope:+.3f} r(alpha,sigma) {np.corrcoef(a_, s)[0,1]:+.3f}  "
          f"sd log sigma {s.std():.3f} sd log(alpha sigma) {(a_ + s).std():.3f}  "
          f"r(alpha,quad) {np.corrcoef(a_, q)[0,1]:+.3f} r(alpha,curv) {np.corrcoef(a_, c)[0,1]:+.3f} "
          f"r(alpha,floored) {np.corrcoef(a_, f)[0,1]:+.3f}")
    keep = np.array(la) <= math.log(.999)
    k2 = [k for k, y in zip(keys, keep) if y]
    a2, s2 = (within(k2, np.array(v)[keep]) for v in (la, ls))
    print(f"  unclamped  n={keep.sum()} slope {(a2 @ s2) / (s2 @ s2):+.3f} r {np.corrcoef(a2, s2)[0,1]:+.3f}  "
          f"sd log sigma {s2.std():.3f} sd log(alpha sigma) {(a2 + s2).std():.3f}")


def main():
    for run in sys.argv[1:]:
        p = Path(run)
        peak = json.load(open(p / "manifest.json"))["config"]["lr"]
        rows = [json.loads(x) for x in open(p / "diagnostics.jsonl")]
        report(rows, peak, f"{p.parent.name}/{p.name}")


if __name__ == "__main__":
    main()
