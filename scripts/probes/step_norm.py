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
                                   hyper: peak / (1 + t / STEP_NORM_T0).
    STEP_NORM_NORM = fisher | rms  What the angle fixes in place of sigma: c = rot_t / sqrt(X F X),
                                   parent's curv; or c = rot_t * sqrt(m n) / ||W X_in + X_out W||_F,
                                   Pion's RMS on the same X. The logged `angle` is then the real c * sigma.
                                   pre: c = rot_t / sigma(G), the normalisation before F^-1, not after.
    STEP_NORM_CAL = path           Append sqrt(curv) / sigma, the RMS base / sigma and sigma(X) / sigma(G)
                                   per layer every 50 steps.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

import harness.train as train
import ngd_pion.unified as unified
from harness.run import main as run_main
from ngd_pion.unified import NGDPionUnified, spectral_norm


PRECOND = os.environ.get("STEP_NORM_PRECOND", "fisher")
SCHEDULE = os.environ.get("STEP_NORM_SCHEDULE", "cosine")
NORM = os.environ.get("STEP_NORM_NORM", "spectral")
CAL = os.environ.get("STEP_NORM_CAL")
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
    if SCHEDULE == "hyper":
        return peak / (1.0 + t / float(os.environ["STEP_NORM_T0"]))
    raise SystemExit(f"unknown schedule {SCHEDULE}")


G_SIGMAS: list[float] = []
G_VECTORS: dict = {}


def recording_natural_gradient(parent):
    """Record sigma(G) per side on its way into F^-1; what is returned is unchanged."""
    def wrapped(G_skew, basis):
        key = (CURRENT_P, len(G_SIGMAS))
        iters = 8 if key in G_VECTORS else 30
        s, G_VECTORS[key] = spectral_norm(G_skew, iters, G_VECTORS.get(key))
        G_SIGMAS.append(float(s))
        return parent(G_skew, basis)
    return wrapped


CURRENT_P = None


class StepNorm(NGDPionUnified):
    def _apply(self, p, group):
        global CURRENT_P
        CURRENT_P = id(p)
        G_SIGMAS.clear()
        self._W = p.detach().to(group["compute_dtype"])
        super()._apply(p, group)

    def _rms_base(self, X_in, X_out):
        W = self._W
        m, n = W.shape
        return float(torch.linalg.matrix_norm(W @ X_in + X_out @ W, "fro")) / math.sqrt(m * n)

    def _measure_angle(self, state, group, X_in, X_out, c):
        _, sigma = super()._measure_angle(state, group, X_in, X_out, c)
        s = max(float(sigma), 1e-30)
        self._angle = scheduled(c, state["step"])
        if CAL and state["step"] % 50 == 0:
            with open(CAL, "a") as f:
                f.write(json.dumps({"step": state["step"], "shape": list(self._W.shape),
                                    "fisher": math.sqrt(max(float(state["curv"]), 0.0)) / s,
                                    "rms": self._rms_base(X_in, X_out) / s,
                                    "pre": s / max(max(G_SIGMAS), 1e-30)}) + "\n")
        if NORM == "spectral":
            self._c = self._angle / s
            return self._angle, sigma
        if NORM == "fisher":
            self._c = self._angle / max(math.sqrt(max(float(state["curv"]), 0.0)), 1e-30)
        elif NORM == "pre":
            self._c = self._angle / max(max(G_SIGMAS), 1e-30)
        elif NORM == "rms":
            self._c = self._angle / max(self._rms_base(X_in, X_out), 1e-30)
        else:
            raise SystemExit(f"unknown norm {NORM}")
        return self._c * s, sigma

    def _retract(self, X, c, group, sigma):
        s = max(float(sigma), 1e-30)
        return super()._retract(X, self._c, group, s)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if PRECOND == "none":
        unified.natural_gradient = lambda G_skew, basis: G_skew
    elif PRECOND != "fisher":
        raise SystemExit(f"unknown preconditioner {PRECOND}")
    if NORM == "pre" or CAL:
        unified.natural_gradient = recording_natural_gradient(unified.natural_gradient)
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = StepNorm
    # `-m` is the name under which the harness honours --ngd-momentum; `-s` forces "none".
    train.NGD_IMPLEMENTATIONS["ngd-pion-m"] = StepNorm
    run_main()
