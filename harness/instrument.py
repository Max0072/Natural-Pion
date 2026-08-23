"""Per-layer diagnostics for NGD-Pion.

Three quantities, each of which the design discussion left as a question that
only a real run can answer:

* `alpha` -- the trust-region ratio. It is identically 1 on a fresh basis, so
  whatever it reads is a measurement of how far the factorisation has drifted
  behind `W`. It is the only feedback on `T_fac` the method produces for free.
* `angle` -- the rotation applied this step, `lr * alpha * ||X||_2`. Dropping
  Pion's RMS scaling removed the mechanism that pinned this; whether the
  Fisher keeps it bounded on its own is the open question.
* `cond(A)` -- decides whether fp32 suffices, and sets the useful range of the
  spectral floor.

Cheap enough to run every few hundred steps, too expensive for every step.
"""

from __future__ import annotations

import torch

__all__ = ["layer_diagnostics", "summarise"]


@torch.no_grad()
def layer_diagnostics(optimizer, names: dict | None = None) -> list[dict]:
    """One row per parameter the optimizer holds."""
    rows = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "bases" not in state:
                continue
            basis_in, basis_out = state["bases"]
            lam = basis_in.lam
            positive = lam[lam > 0]
            rows.append(
                {
                    "name": (names or {}).get(id(p), f"{tuple(p.shape)}"),
                    "shape": tuple(p.shape),
                    "alpha": float(state.get("alpha", float("nan"))),
                    "angle": float(state.get("angle", float("nan"))),
                    "cond_A": float(_cond(state["cov"].matrix)),
                    "lam_ratio": float(positive.max() / positive.min()) if positive.numel() else float("nan"),
                    "step": int(state.get("step", 0)),
                }
            )
    return rows


def _cond(A: torch.Tensor) -> float:
    w = torch.linalg.eigvalsh(A.float())
    w = w.clamp_min(0.0)
    positive = w[w > w.max() * 1e-12]
    return float(w.max() / positive.min()) if positive.numel() else float("inf")


def summarise(rows: list[dict]) -> dict:
    """Collapse per-layer rows to what belongs on a training log line."""
    if not rows:
        return {}
    def stat(key):
        vals = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
        return (min(vals), max(vals)) if vals else (float("nan"),) * 2
    a_lo, a_hi = stat("alpha")
    g_lo, g_hi = stat("angle")
    c_lo, c_hi = stat("cond_A")
    return {
        "alpha_min": a_lo, "alpha_max": a_hi,
        "angle_min": g_lo, "angle_max": g_hi,
        "condA_min": c_lo, "condA_max": c_hi,
    }
