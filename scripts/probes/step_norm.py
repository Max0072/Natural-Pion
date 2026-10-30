"""NGD-Pion with alpha off and the step's length set by the schedule. (journal 2026-10-30.)

c = rot_t / sigma, sigma = spectral norm of X, the larger of in and out sides as the
parent measures it. So c * sigma = rot_t: `--lr` is the nominal angle c * sigma at the
peak, for every layer and every step, and the direction stays X = F^-1 G. Run with
`--ngd-trust none`. The logged `angle` is then rot_t by construction. `pred_drop` keeps
the parent's c and is wrong here; it steers nothing while trust_lr is off.
"""

from __future__ import annotations

import sys

import torch

import harness.train as train
from harness.run import main as run_main
from ngd_pion.unified import NGDPionUnified


class StepNorm(NGDPionUnified):
    def _measure_angle(self, state, group, X_in, X_out, c):
        _, sigma = super()._measure_angle(state, group, X_in, X_out, c)
        return c, sigma

    def _retract(self, X, c, group, sigma):
        s = max(float(sigma), 1e-30)
        return super()._retract(X, c / s, group, s)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = StepNorm
    run_main()
