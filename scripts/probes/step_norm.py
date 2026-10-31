"""NGD-Pion with alpha off and the step's length set by the schedule. (journal 2026-10-30.)

c = rot_t / sigma, sigma = spectral norm of X, the larger of in and out sides as the
parent measures it. So c * sigma = rot_t: `--lr` is the nominal angle c * sigma at the
peak, for every layer and every step, and the direction stays X = F^-1 G. Run with
`--ngd-trust none`. The logged `angle` is then rot_t by construction. `pred_drop` keeps
the parent's c and is wrong here; it steers nothing while trust_lr is off.

Two controls, both off by default, so the default is what job 393882 ran:

    STEP_NORM_PRECOND = none       X = G: no Fisher, the same code path and normalisation.
                                   Not `--ngd-s-power 0`: P is not orthogonal, so
                                   P P^T G P P^T is not G.
    STEP_NORM_SCHEDULE = constant | linear | warmcos, with STEP_NORM_PEAK (= --lr) and
                                   STEP_NORM_STEPS. Rotational angle only; AdamW
                                   keeps its cosine. warmcos: linear up over the first 20%,
                                   then cosine, roughly the shape alpha gave.
"""

from __future__ import annotations

import math
import os
import sys

import torch

import harness.train as train
import ngd_pion.unified as unified
from harness.run import main as run_main
from ngd_pion.unified import NGDPionUnified


PRECOND = os.environ.get("STEP_NORM_PRECOND", "fisher")
SCHEDULE = os.environ.get("STEP_NORM_SCHEDULE", "cosine")
FLOOR = 1e-5


def scheduled(c: float, t: int) -> float:
    """The angle at optimizer step `t`; `c` is the parent's cosine value."""
    if SCHEDULE == "cosine":
        return c
    peak, steps = float(os.environ["STEP_NORM_PEAK"]), int(os.environ["STEP_NORM_STEPS"])
    u = min(1.0, t / steps)
    if SCHEDULE == "constant":
        return peak
    if SCHEDULE == "linear":
        return FLOOR + (peak - FLOOR) * (1.0 - u)
    if SCHEDULE == "warmcos":
        if u < 0.2:
            return peak * (t + 1) / (0.2 * steps)
        return FLOOR + 0.5 * (peak - FLOOR) * (1.0 + math.cos(math.pi * (u - 0.2) / 0.8))
    raise SystemExit(f"unknown schedule {SCHEDULE}")


class StepNorm(NGDPionUnified):
    def _measure_angle(self, state, group, X_in, X_out, c):
        _, sigma = super()._measure_angle(state, group, X_in, X_out, c)
        self._angle = scheduled(c, state["step"])
        return self._angle, sigma

    def _retract(self, X, c, group, sigma):
        s = max(float(sigma), 1e-30)
        return super()._retract(X, self._angle / s, group, s)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if PRECOND == "none":
        unified.natural_gradient = lambda G_skew, basis: G_skew
    elif PRECOND != "fisher":
        raise SystemExit(f"unknown preconditioner {PRECOND}")
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = StepNorm
    # `-m` is the name under which the harness honours --ngd-momentum; `-s` forces "none".
    train.NGD_IMPLEMENTATIONS["ngd-pion-m"] = StepNorm
    run_main()
