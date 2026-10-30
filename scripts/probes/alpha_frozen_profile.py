"""Build the frozen alpha tables for alpha_frozen.py from runs' diagnostics.jsonl.
(journal 2026-10-30.) Means of alpha itself, not of its log: c = rot * alpha is
linear in alpha, so the mean keeps the rate the run really used. Steps are the
optimizer's own (0-based). Layer index = order of rows within a logged step.

    time    mean over layers in each 100-step bin
    layer   per layer, weighted by the cosine rate, so that the layer's total
            rate over training is kept
    both    per layer and bin

    python scripts/probes/alpha_frozen_profile.py OUT.json RUN [RUN ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BIN, STEPS = 100, 3000


def cos_rate(t):
    return 0.5 * (1 + math.cos(math.pi * t / STEPS))


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    time, both = defaultdict(list), defaultdict(list)
    wl, w = defaultdict(float), defaultdict(float)
    names = None
    for run in runs:
        steps = defaultdict(list)
        for x in open(Path(run) / "diagnostics.jsonl"):
            r = json.loads(x)
            steps[r["step"] - 1].append(r)
        for t, rows in steps.items():
            if t == 0:  # alpha = 1 by construction; 1 of 5 logged rows in bin 0 would inflate it
                continue
            assert len(rows) == 56, (run, t, len(rows))
            here = [r["name"] for r in rows]
            names = names or here
            assert here == names, (run, t)
            b = min(t // BIN, STEPS // BIN - 1)
            for i, r in enumerate(rows):
                a = float(r["alpha"])
                time[b].append(a)
                both[f"{i}:{b}"].append(a)
                wl[i] += cos_rate(t) * a
                w[i] += cos_rate(t)
    mean = lambda v: sum(v) / len(v)
    prof = {"bin": BIN, "names": names,
            "time": {str(b): mean(v) for b, v in sorted(time.items())},
            "layer": {str(i): wl[i] / w[i] for i in sorted(wl)},
            "both": {k: mean(v) for k, v in both.items()}}
    assert len(prof["time"]) == STEPS // BIN and len(prof["both"]) == 56 * STEPS // BIN
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(prof, f, indent=0)
    t = prof["time"]
    print("time  ", " ".join(f"{int(b)*BIN}:{t[b]:.3f}" for b in list(t)[:6]), "...",
          " ".join(f"{int(b)*BIN}:{t[b]:.3f}" for b in list(t)[::6][1:]))
    l = prof["layer"]
    print("layer  min %.3f max %.3f" % (min(l.values()), max(l.values())))
    print("both   min %.4f max %.3f" % (min(prof["both"].values()), max(prof["both"].values())))


if __name__ == "__main__":
    main()
