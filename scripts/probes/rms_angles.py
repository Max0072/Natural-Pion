"""Which matrices the RMS length rule turns more, and which less. CPU, from a Natural Pion Fast checkpoint of a run with
length="rms". For every rotated unit (a matrix, or one head block of the query projection) the step direction is
rebuilt from the checkpoint's own bases and momenta, X = F^-1 M per side, and the RMS rule's coefficient
c = theta sqrt(m n) / ||W X_in + X_out W||_F gives the realised angle a = c max(||X_in||_2, ||X_out||_2). theta cancels
in the comparison: reported is a / median(a), by matrix type and by layer. The angle rule would give 1 everywhere.

    python scripts/probes/rms_angles.py <checkpoint.pt>    # writes runs-record/probes/rms_angles.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

from ngd_pion.linalg import skew

ROOT = Path(__file__).resolve().parents[2]


def snorm(X):
    return float(torch.linalg.eigvalsh(-(X @ X)).amax().clamp_min(0).sqrt())


def direction(b, M):
    P, lam = b.P.double(), b.lam.double()
    den = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
    return skew(P @ ((P.T @ M.double() @ P) / den) @ P.T)


def main():
    torch.set_num_threads(8)
    sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    names = [k for k, v in sd["model"].items() if k.startswith("blocks.") and v.ndim == 2]
    st = sd["rot"]["state"]
    rows = []
    for j, name in enumerate(names):
        W = sd["model"][name].double()
        s = st[j]
        units = [(0, s)] + sorted(s.get("blocks", {}).items())
        k = len(units)
        h = W.shape[0] // k
        for i, u in units:
            Wu = W[i * h:(i + 1) * h]
            b_in, b_out = u["bases"]
            Xi, Xo = direction(b_in, u["m_in"]), direction(b_out, u["m_out"])
            m, n = Wu.shape
            rms = float(torch.linalg.matrix_norm(Wu @ Xi + Xo @ Wu)) / (m * n) ** 0.5
            rows.append({"name": name, "unit": i, "angle": max(snorm(Xi), snorm(Xo)) / max(rms, 1e-30)})
    med = statistics.median(r["angle"] for r in rows)
    for r in rows:
        r["rel"] = r["angle"] / med
    by_type, by_layer = defaultdict(list), defaultdict(list)
    for r in rows:
        mt = re.match(r"blocks\.(\d+)\.(.+)\.weight", r["name"])
        by_layer[int(mt.group(1))].append(r["rel"]); by_type[mt.group(2)].append(r["rel"])
    summ = lambda xs: {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}
    out = {"checkpoint": sys.argv[1], "by_type": {t: summ(v) for t, v in by_type.items()},
           "by_layer": {str(l): summ(v) for l, v in sorted(by_layer.items())}, "units": rows}
    for t, v in out["by_type"].items():
        print(f"{t:18s} median {v['median']:.2f}  min {v['min']:.2f}  max {v['max']:.2f}  ({v['n']} units)")
    for l, v in out["by_layer"].items():
        print(f"layer {l}            median {v['median']:.2f}  min {v['min']:.2f}  max {v['max']:.2f}")
    (ROOT / "runs-record" / "probes" / "rms_angles.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
