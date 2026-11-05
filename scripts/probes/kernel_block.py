"""How much of the natural-gradient generator sits in the block that cannot move a rectangular W. CPU.

For W (m x n) with m > n, the out-side rotation acts on R^m, but W's columns span only n directions (U, from the
thin SVD); rotations inside the (m - n)-dimensional complement K = null(W^T) leave W unchanged. For m < n the same
holds on the in-side with the row space. On a NaturalPionFast checkpoint, with its own momenta M and bases, this
computes X = F^-1 M on the wide side of every rectangular matrix and reports
  frac_kk  = ||K^T X K||_F^2 / ||X||_F^2        share of X in the invisible block
  sigma_inflation = ||X||_2 / ||X - K K^T X K K^T||_2   by how much that block inflates sigma (1 = not at all)
Since c = theta / max(sigma_in, sigma_out), an inflation > 1 on the deciding side shrinks the useful rotation.

    python scripts/probes/kernel_block.py <checkpoint.pt>    # writes runs-record/probes/kernel_block.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ngd_pion.linalg import skew

ROOT = Path(__file__).resolve().parents[2]


def snorm(X):
    """||X||_2 for skew X: sqrt of the largest eigenvalue of -X^2."""
    return float(torch.linalg.eigvalsh(-(X @ X)).amax().clamp_min(0).sqrt())


def main():
    torch.set_num_threads(4)
    sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    names = [k for k, v in sd["model"].items() if k.startswith("blocks.") and v.ndim == 2]
    st = sd["rot"]["state"]
    rows = []
    for j, name in enumerate(names):
        W = sd["model"][name].double()
        m, n = W.shape
        if m == n:
            continue
        s = st[j]
        b_in, b_out = s["bases"]
        # the wide side and its generator
        if m > n:
            P, lam, M, side = b_out.P.double(), b_out.lam.double(), s["m_out"].double(), "out"
            U = torch.linalg.svd(W, full_matrices=True)[0]
            K = U[:, n:]                       # complement of the column space, m - n columns
        else:
            P, lam, M, side = b_in.P.double(), b_in.lam.double(), s["m_in"].double(), "in"
            Vh = torch.linalg.svd(W, full_matrices=True)[2]
            K = Vh[m:].T                       # complement of the row space, n - m columns
        den = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
        X = skew(P @ ((P.T @ M @ P) / den) @ P.T)
        Xkk = K @ (K.T @ X @ K) @ K.T
        frac = float((Xkk.norm() / X.norm()) ** 2)
        infl = snorm(X) / snorm(X - Xkk)
        # the other side, to see which one sets sigma
        if side == "out":
            Po, lo, Mo = b_in.P.double(), b_in.lam.double(), s["m_in"].double()
        else:
            Po, lo, Mo = b_out.P.double(), b_out.lam.double(), s["m_out"].double()
        Xo = skew(Po @ ((Po.T @ Mo @ Po) / (0.5 * (lo.unsqueeze(-1) + lo.unsqueeze(-2)))) @ Po.T)
        sig_wide, sig_wide_eff, sig_other = snorm(X), snorm(X - Xkk), snorm(Xo)
        lost = max(sig_wide, sig_other) / max(sig_wide_eff, sig_other)   # factor by which c shrinks
        rows.append(dict(name=name, side=side, frac_kk=frac, sigma_inflation=infl, c_shrink=lost))
        print(f"{name:26s} {side:3s} frac_kk {frac:.3f}  sigma_inflation {infl:.3f}  c shrinks by {lost:.3f}", flush=True)
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
    out = {"checkpoint": sys.argv[1], "rows": rows, "median_frac_kk": med("frac_kk"),
           "median_sigma_inflation": med("sigma_inflation"), "median_c_shrink": med("c_shrink"),
           "max_c_shrink": max(r["c_shrink"] for r in rows)}
    print({k: v for k, v in out.items() if k != "rows"})
    (ROOT / "runs-record" / "probes" / "kernel_block.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
