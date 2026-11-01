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
    STEP_NORM_ATTN_MULT = m        Multiply the angle of the attention matrices (the square ones, wq wk wv wo)
                                   by m, leaving the FFN's (rectangular) at rot_t. Spectral norm only.
    STEP_NORM_POWER = p            Reshape the spectrum of X before the retraction: X = U S V^T goes to
                                   U (S / sigma)^p V^T. 1 is X / sigma, today's step; 0 turns every plane by
                                   the same angle, as a polar step does. Spectral norm only.
    STEP_NORM_A_LAST, _D_LAST = k  Feed A or D only in the last k steps before each refactor. With alpha off they
                                   are read only there, so this asks how much of the collection is used.
    STEP_NORM_A_EVERY = n          Feed A every n-th step. Set --ngd-beta to beta^n to keep its time constant.
    STEP_NORM_FAST = 1             Skip what nothing reads with alpha off: quad, curv, pred_drop and the per-layer
                                   diagnostics with their probe. Must leave the trajectory bit-identical; check it
                                   against the same run without the flag. Refuses the settings that read them.
    STEP_NORM_CAL = path           Append sqrt(curv) / sigma, the RMS base / sigma and sigma(X) / sigma(G)
                                   per layer every 50 steps.
    STEP_NORM_PROFILE = path       Rewrite cumulative seconds per optimizer part, synchronised with the GPU,
                                   every 50 optimizer steps. The sync slows the run slightly.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

import harness.train as train
import ngd_pion.unified as unified
from harness.run import main as run_main
from ngd_pion.unified import NGDPionUnified, generators, spectral_norm


PRECOND = os.environ.get("STEP_NORM_PRECOND", "fisher")
SCHEDULE = os.environ.get("STEP_NORM_SCHEDULE", "cosine")
NORM = os.environ.get("STEP_NORM_NORM", "spectral")
CAL = os.environ.get("STEP_NORM_CAL")
POWER = float(os.environ.get("STEP_NORM_POWER", "1"))
A_LAST = int(os.environ.get("STEP_NORM_A_LAST", "0"))
D_LAST = int(os.environ.get("STEP_NORM_D_LAST", "0"))
A_EVERY = int(os.environ.get("STEP_NORM_A_EVERY", "1"))
ATTN_MULT = float(os.environ.get("STEP_NORM_ATTN_MULT", "1"))
FAST = os.environ.get("STEP_NORM_FAST") == "1"
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


def reshaped(X: torch.Tensor, sigma: float, power: float) -> torch.Tensor:
    """U (S / sigma)^power V^T, kept skew. A plane at round-off (below 1e-6 sigma) stays still, it has no direction."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = S / max(sigma, 1e-30)
    Y = (U * torch.where(r > 1e-6, r.pow(power), torch.zeros_like(r))) @ Vh
    return 0.5 * (Y - Y.transpose(-1, -2))


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
        if FAST:
            return self._apply_fast(p, group)
        super()._apply(p, group)

    def _apply_fast(self, p, group):
        """The parent's `_apply` for trust none, Lie momentum or none, no alternation, without quad and curv."""
        state = self.state[p]
        dt = group["compute_dtype"]
        W = self._W
        basis_in, basis_out = state["bases"]
        G_in, G_out = generators(W, p.grad.detach().to(dt))
        if group["momentum"] == "lie":
            G_in = self._ema(state, "m_in", G_in, group["beta1"])
            G_out = self._ema(state, "m_out", G_out, group["beta1"])
        X_in = unified.natural_gradient(G_in, basis_in)
        X_out = unified.natural_gradient(G_out, basis_out)
        alpha = float(torch.as_tensor(group["alpha_max"], dtype=dt))
        c = group["lr"] * group.get("lr_scale", 1.0) * alpha
        _, sigma = self._measure_angle(state, group, X_in, X_out, c)
        W = self._retract(X_out, c, group, sigma) @ W @ self._retract(X_in, c, group, sigma)
        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1

    def load_state_dict(self, state_dict):
        # The parent reloads `cov` but not `cov_backward` to the GPU, so a resume dies at the first backward hook (job 394420).
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                cov = self.state.get(p, {}).get("cov_backward")
                if cov is not None:
                    cov.to(p.device)

    def _outside_window(self, param, last):
        state = self.state[param]
        return "bases" in state and state["since_refactor"] < self._group_of(param)["t_fac"] - last

    def observe(self, param, x):
        if A_LAST and self._outside_window(param, A_LAST):
            return
        if A_EVERY > 1 and "bases" in self.state[param] and self.state[param]["step"] % A_EVERY:
            return
        super().observe(param, x)

    def observe_backward(self, param, delta):
        if D_LAST and self._outside_window(param, D_LAST):
            return
        super().observe_backward(param, delta)

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
            if ATTN_MULT != 1.0 and self._W.shape[0] == self._W.shape[1]:
                self._angle *= ATTN_MULT
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
        if POWER != 1.0:
            return super()._retract(reshaped(X, s, POWER), self._angle, group, 1.0)
        return super()._retract(X, self._c, group, s)


def profile(cls, path, names):
    """Wrap `names` with timers. Nested parts count in their parents: _step holds _refactor and _apply."""
    totals = {n: [0.0, 0] for n in names}

    def wrap(name):
        orig = getattr(cls, name)

        def timed(self, *a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = orig(self, *a, **k)
            torch.cuda.synchronize()
            totals[name][0] += time.perf_counter() - t
            totals[name][1] += 1
            if name == "_step" and totals[name][1] % 50 == 0:
                with open(path, "w") as f:
                    json.dump(totals, f)
            return out
        setattr(cls, name, timed)

    for n in names:
        wrap(n)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if FAST:
        if CAL or NORM != "spectral":
            raise SystemExit("STEP_NORM_FAST has no curv: spectral norm only, no CAL")
        train.DIAGNOSED = ()
    if PRECOND == "none":
        unified.natural_gradient = lambda G_skew, basis: G_skew
    elif PRECOND != "fisher":
        raise SystemExit(f"unknown preconditioner {PRECOND}")
    if NORM == "pre" or CAL:
        unified.natural_gradient = recording_natural_gradient(unified.natural_gradient)
    if os.environ.get("STEP_NORM_PROFILE"):
        profile(StepNorm, os.environ["STEP_NORM_PROFILE"],
                ["_step", "_refactor", "_apply", "_measure_angle", "_retract", "observe", "observe_backward"])
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = StepNorm
    # `-m` is the name under which the harness honours --ngd-momentum; `-s` forces "none".
    train.NGD_IMPLEMENTATIONS["ngd-pion-m"] = StepNorm
    run_main()
