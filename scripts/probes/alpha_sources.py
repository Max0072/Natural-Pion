"""What does `alpha = quad/curv` read: drift of W, of A, or of D? (journal 2026-10-29)

`X = F_0^-1 G` comes from the basis built at the last refactorisation out of
`(W_0, A_0, D_0)`; `curv = <X, F(W_t, A_t, D_t) X>`. The Fisher enters as

    in-side   F(X) = 2(B X C + C X B),  B = A,  C = W^T D W
    out-side  F(X) = 2(B X C + C X B),  B = D,  C = W A W^T

so `curv` can be recomputed with any one of the three replaced by its
snapshot. For each logged layer-step this writes

    r_W = curv(W_t, A_0, D_0) / curv_0      only the weight has moved
    r_A = curv(W_0, A_t, D_0) / curv_0      only the input covariance
    r_D = curv(W_0, A_0, D_t) / curv_0      only the backward covariance
    r   = curv(W_t, A_t, D_t) / curv_0      what the optimizer actually used

with `curv_0 = curv(W_0, A_0, D_0)` (equal to `quad` but for the floor). A
source whose `r` sits near 1 is not what alpha is reading.

The subclass only adds measurement; the step is the parent's, unchanged. As a
check, `train_loss` must match the same seed and flags without the probe.

Run (one GPU, through the harness so the run is the real one):
    ALPHA_SOURCES_OUT=... python scripts/probes/alpha_sources.py <harness.run args>
"""

from __future__ import annotations

import json
import os
import sys

import torch

import harness.train as train
from harness.run import main as run_main
from ngd_pion.direction import fisher_apply, generators, natural_gradient
from ngd_pion.unified import NGDPionUnified

OUT = os.environ["ALPHA_SOURCES_OUT"]
EVERY = int(os.environ.get("ALPHA_SOURCES_EVERY", "23"))


def _curv(W, A, D, X_in, X_out):
    Wt = W.transpose(-1, -2)
    return float((X_in * fisher_apply(A, Wt @ D @ W, X_in)).sum()
                 + (X_out * fisher_apply(D, W @ A @ Wt, X_out)).sum())


class AlphaSources(NGDPionUnified):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._out = open(OUT, "a")
        self._index = {id(p): i for g in self.param_groups for i, p in enumerate(g["params"])}

    def _refactor(self, params, group):
        super()._refactor(params, group)
        dt = group["compute_dtype"]
        for p in params:
            st = self.state[p]
            st["snap"] = (p.detach().to(dt).clone(),
                          st["cov"].matrix.to(device=p.device, dtype=dt).clone(),
                          self._D(p, group).clone())

    def _apply(self, p, group):
        st = self.state[p]
        if (group["use_s"] and group["momentum"] == "none" and "snap" in st
                and st["step"] % EVERY == 0):
            dt = group["compute_dtype"]
            W = p.detach().to(dt)
            G_in, G_out = generators(W, p.grad.detach().to(dt))
            b_in, b_out = st["bases"]
            X_in, X_out = natural_gradient(G_in, b_in), natural_gradient(G_out, b_out)
            A = st["cov"].matrix.to(device=p.device, dtype=dt)
            D = self._D(p, group)
            W0, A0, D0 = st["snap"]
            c0 = _curv(W0, A0, D0, X_in, X_out)
            quad = float((G_in * X_in).sum() + (G_out * X_out).sum())
            row = {
                "step": st["step"], "i": self._index[id(p)], "shape": list(p.shape),
                "age": st["since_refactor"], "quad_over_c0": quad / c0,
                "r_W": _curv(W, A0, D0, X_in, X_out) / c0,
                "r_A": _curv(W0, A, D0, X_in, X_out) / c0,
                "r_D": _curv(W0, A0, D, X_in, X_out) / c0,
                "r": _curv(W, A, D, X_in, X_out) / c0,
            }
            self._out.write(json.dumps(row) + "\n")
            self._out.flush()
        super()._apply(p, group)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = AlphaSources
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    run_main()
