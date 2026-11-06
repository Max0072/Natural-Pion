"""Is the angle rule what it says? In a training run of Natural Pion Fast (the method: 3000 steps, angle 0.2, seed 0,
query projection per head), every 25th step and every step right after a basis refresh, the realised rotation angle
of each rotated unit is computed with the exact spectral norm: a = c * max(||X_in||_2, ||X_out||_2), with c the step's
coefficient (theta / the power-iteration estimate). The rule promises a = theta for every unit.

Reported per measured step: theta, and the quantiles of a / theta over units (p10, median, p90, min, max), for
all units and separately for the query-projection head blocks and for the other matrices.

    python scripts/probes/angle_check.py <data dir> <out dir>    # writes runs-record/probes/angle_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import harness.train as T
from ngd_pion.natural_pion import NaturalPionFast

ROOT = Path(__file__).resolve().parents[2]
REC: dict = {"steps": []}


def snorm(X: torch.Tensor) -> torch.Tensor:
    """||X||_2 of a batch of skew matrices: sqrt of the largest eigenvalue of -X^2 (eigvalsh; svd is slow on skew)."""
    return torch.linalg.eigvalsh(-(X.double() @ X.double())).amax(dim=-1).clamp_min(0).sqrt()


class Probe(NaturalPionFast):
    @torch.no_grad()
    def step(self, closure=None):
        ps = [p for g in self.param_groups for p in g["params"]]
        self._t = getattr(self, "_t", -1) + 1
        self._refresh_now = any(self.state[p].get("since_refactor", 0) == 0 for p in ps if "bases" in self.state[p])
        return super().step(closure)

    def _scales(self, rows, group):
        cs = super()._scales(rows, group)
        if not (self._t % 25 == 0 or self._refresh_now):
            return cs
        a, q = [], []
        for (p, W, X_in, X_out, b, r, whole), c in zip(rows, cs):
            s = torch.maximum(snorm(X_in[None]), snorm(X_out[None]))[0]
            a.append(float(c) * float(s) / group["lr"])
            q.append(not whole)                              # the query projection's head blocks
        qs = lambda xs: {"p10": xs[int(0.1 * (len(xs) - 1))], "median": xs[len(xs) // 2],
                         "p90": xs[int(0.9 * (len(xs) - 1))], "min": xs[0], "max": xs[-1]} if xs else None
        REC["steps"].append({"t": self._t, "theta": group["lr"], "after_refresh": self._refresh_now,
                             "all": qs(sorted(a)), "query_blocks": qs(sorted(x for x, k in zip(a, q) if k)),
                             "other": qs(sorted(x for x, k in zip(a, q) if not k))})
        return cs


T.NATURAL_PION["natural-pion-fast"] = Probe
from harness import run  # noqa: E402

sys.argv = ["run", "--optimizer", "natural-pion-fast", "--lr", "0.2", "--ngd-beta1", "0.9", "--train-steps", "3000",
            "--batch-sequences", "512", "--micro-batch", "256", "--rho-micro", "128", "--precision", "bf16",
            "--seed", "0", "--device", "cuda", "--log-every", "23", "--eval-every", "100000", "--adamw-lr", "8e-3",
            "--data-path", sys.argv[1] + "/c4_train.bin", "--val-path", sys.argv[1] + "/c4_val.bin",
            "--out-dir", sys.argv[2], "--no-resume"]
try:
    run.main()
finally:
    st = REC["steps"]
    if st:
        med = sorted(s["all"]["median"] for s in st)
        REC["summary"] = {"measured_steps": len(st), "median_of_medians": med[len(med) // 2],
                          "worst_min": min(s["all"]["min"] for s in st), "worst_max": max(s["all"]["max"] for s in st),
                          "median_p90_over_p10": sorted(s["all"]["p90"] / s["all"]["p10"] for s in st)[len(st) // 2]}
        print(json.dumps(REC["summary"], indent=1))
    (ROOT / "runs-record" / "probes" / "angle_check.json").write_text(json.dumps(REC, indent=1))
