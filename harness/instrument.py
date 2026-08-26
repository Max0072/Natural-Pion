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
* `quad_over_curv` -- `alpha` before the clamp at `alpha_max`. `alpha` sits at
  the cap in every run measured so far, which means it reports only "at least
  1" and hides how far above it the ratio really is. That distance is what says
  whether the curvature term is missing a factor.
* `skew_ratio` -- `||skew(W^T G)|| / ||sym(W^T G)||`. The method uses only the
  antisymmetric part, obtained as a difference of nearly equal quantities from
  a gradient that came through bf16 autocast. Below about `1e-2` that part is
  lost to rounding, so this says whether a small rotation is a fact about the
  geometry or an artefact of precision.

`layer_diagnostics` returns one row per parameter and `summarise` collapses
them to a log line. Both are used: the summary goes on the training log, the
rows go to `diagnostics.jsonl`, because questions like "does the required step
size depend on depth" cannot be answered from a min and a max.

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
            spec = _spectrum(state["cov"].matrix, group["eps"])
            fl_in, n_in = _at_floor(basis_in, group["eps"])
            fl_out, n_out = _at_floor(basis_out, group["eps"])
            rows.append(
                {
                    "name": (names or {}).get(id(p), f"{tuple(p.shape)}"),
                    "shape": tuple(p.shape),
                    "alpha": float(state.get("alpha", float("nan"))),
                    "angle": float(state.get("angle", float("nan"))),
                    "angle_requested": float(state.get("angle_requested", float("nan"))),
                    "cond_A": spec["cond_trunc"],
                    "lam_max_A": spec["lam_max"],
                    "lam_min_A": spec["lam_min"],
                    "n_below_floor": spec["n_below_floor"],
                    "null_frac": spec["null_frac"],
                    "n_negative": spec["n_negative"],
                    "neg_frac": spec["neg_frac"],
                    "floored_in": fl_in,
                    "floored_frac_in": fl_in / n_in,
                    "floored_frac_out": fl_out / n_out,
                    "orthogonal_in": bool(basis_in.orthogonal),
                    "skew_ratio": float(state.get("skew_ratio", float("nan"))),
                    "quad": float(state.get("quad", float("nan"))),
                    "curv": float(state.get("curv", float("nan"))),
                    "quad_over_curv": float(state.get("quad_over_curv", float("nan"))),
                    "floor_share_in": float(state.get("floor_share_in", float("nan"))),
                    "floor_share_out": float(state.get("floor_share_out", float("nan"))),
                    "depth": _depth(names, p),
                    "lam_ratio": float(positive.max() / positive.min()) if positive.numel() else float("nan"),
                    "step": int(state.get("step", 0)),
                }
            )
    return rows


def _depth(names: dict | None, p: torch.Tensor) -> int:
    """Block index parsed out of the module name, or -1 when there is none.

    The whole point of the per-layer rows is comparing early layers against
    late ones, and doing that by string matching at analysis time is how a
    naming change silently turns into a wrong plot.
    """
    name = (names or {}).get(id(p), "")
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def _at_floor(basis, eps: float) -> tuple[int, int]:
    """How many of a basis's eigenvalues the floor actually raised.

    Not the same question as `_spectrum`'s `n_below_floor`, and the difference
    is not academic. `n_below_floor` counts the degenerate directions of `A`,
    but on the in-side `build_bases` takes the congruence path whenever
    `W^T W != I`, and what gets floored there is the spectrum of the pencil
    `A^-1/2 (W^T W) A^-1/2`, not of `A`. Job 252299 showed `blocks.0.attn.wq`
    with `cond(A) = 86` and nothing below the floor, while the basis it was
    solved in had been floored to a ratio of exactly `1/eps`. Measuring the
    covariance alone would have reported that layer as healthy.
    """
    lam = basis.lam
    hit = lam <= lam.amax(dim=-1, keepdim=True) * eps * (1.0 + 1e-5)
    return int(hit.sum()), int(lam.numel())


def _spectrum(A: torch.Tensor, eps: float) -> dict:
    """What the accumulated covariance's spectrum actually looks like.

    `cond_trunc` is the number this module reported on its own until
    2026-08-26, kept so the older logs stay comparable, and it is an artefact:
    it drops every eigenvalue below `lam_max * 1e-12` and reports the
    condition number of the remainder. On the real runs it reads `1e5` to
    `1e7` while the true condition number is infinite, because `A` is
    rank-deficient on every layer -- which is precisely the thing that makes
    `quad` and `curv` disagree, and precisely the thing this number cannot
    show. `n_below_floor` and `lam_min` are the honest ones.
    """
    # Not clamped. An earlier version wrote `.clamp_min(0.0)` here, on the same
    # reasoning as `floor_eigenvalues` -- that `A` is PSD by construction and a
    # negative value is rounding -- and so reported `lam_min = 0` for every
    # layer of the working run while the truth was 6253 negative eigenvalues
    # across 56 layers, one of them at -2.13 against a `lam_max` of 446. That
    # is the quantity that sends `curv` negative, and the diagnostic that was
    # supposed to find it was hiding it.
    w = torch.linalg.eigvalsh(A.float())
    lam_max = float(w.max())
    pos = w.clamp_min(0.0)
    kept = pos[pos > lam_max * 1e-12]
    return {
        "cond_trunc": float(lam_max / kept.min()) if kept.numel() else float("inf"),
        "lam_max": lam_max,
        "lam_min": float(w.min()),
        "n_negative": int((w < 0).sum()),
        "neg_frac": float((w < 0).sum()) / w.numel(),
        "n_below_floor": int((w < lam_max * eps).sum()),
        "null_frac": float((w < lam_max * eps).sum()) / w.numel(),
    }


def summarise(rows: list[dict]) -> dict:
    """Collapse per-layer rows to what belongs on a training log line."""
    if not rows:
        return {}
    def stat(key):
        vals = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
        return (min(vals), max(vals)) if vals else (float("nan"),) * 2
    q_lo, q_hi = stat("quad_over_curv")
    s_lo, s_hi = stat("skew_ratio")
    a_lo, a_hi = stat("alpha")
    g_lo, g_hi = stat("angle")
    r_lo, r_hi = stat("angle_requested")
    c_lo, c_hi = stat("cond_A")
    n_lo, n_hi = stat("null_frac")
    f_lo, f_hi = stat("floored_frac_in")
    g_neg_lo, g_neg_hi = stat("neg_frac")
    return {
        "alpha_min": a_lo, "alpha_max": a_hi,
        "angle_min": g_lo, "angle_max": g_hi,
        "angle_req_min": r_lo, "angle_req_max": r_hi,
        "condA_min": c_lo, "condA_max": c_hi,
        "nullfrac_min": n_lo, "nullfrac_max": n_hi,
        "negfrac_min": g_neg_lo, "negfrac_max": g_neg_hi,
        "floored_min": f_lo, "floored_max": f_hi,
        "skew_ratio_min": s_lo, "skew_ratio_max": s_hi,
        "qoc_min": q_lo, "qoc_max": q_hi,
    }
