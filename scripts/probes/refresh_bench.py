"""The basis refresh of Natural Pion Fast, timed and checked variant by variant on a trained model's statistics.

Loads a NaturalPionFast checkpoint (weights W, statistics A and D, momenta M_in/M_out for the 56 rotated matrices)
and builds the bases the way `NaturalPion._refactor` does (R0), then with variants:
  R1  eigh(A) once per distinct input tensor (q, k, v and gate, up share A)            exact
  R2  no `is_identity` checks (D is never the identity here; each check is a host sync)   exact
  R3  R1 + R2 with the MAGMA eigensolver instead of cuSOLVER                             exact
  R4  R1 + R2 without the eigendecomposition of C: C is used unfloored, only lam floored approximate
  R6  R1 + R2 with A^-1/2 by coupled Newton-Schulz on A + eps*lmax*I (no eigh(A))         approximate
For each: seconds per refresh (GPU-synchronised, median of 5), and the error of the natural-gradient direction
X = F^-1 M on the checkpoint's own momenta, ||X/||X|| - X0/||X0|| ||_F per side, median and max over matrices.

    python scripts/probes/refresh_bench.py <checkpoint.pt>     # writes runs-record/probes/refresh_bench.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from ngd_pion.factorization import floor_eigenvalues
from ngd_pion.linalg import exact_fp32, safe_eigh, skew

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
EPS = 1e-4


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    names = [k for k, v in sd["model"].items() if k.startswith("blocks.") and v.ndim == 2]
    st = sd["rot"]["state"]
    mats = []
    for j, name in enumerate(names):
        s = st[j]
        mats.append(dict(name=name, W=sd["model"][name].to(DEV), A=s["cov"].matrix.to(DEV), D=s["cov_backward"].matrix.to(DEV),
                         a_id=id(s["cov"]), M_in=s["m_in"].to(DEV), M_out=s["m_out"].to(DEV)))
    return mats


def inv_half_eigh(B):
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, EPS)
    return (Ub * wb.rsqrt().unsqueeze(-2)) @ Ub.transpose(-1, -2)


def inv_half_ns(B, iters=12):
    """Coupled Newton-Schulz for (B + eps*lmax I)^-1/2, normalised by the Frobenius norm (an upper bound on lmax)."""
    n = B.shape[-1]
    eye = torch.eye(n, device=B.device, dtype=B.dtype)
    lmax = torch.linalg.matrix_norm(B, ord="fro", keepdim=True)
    Bd = B + EPS * lmax * eye
    s = torch.linalg.matrix_norm(Bd, ord="fro", keepdim=True)
    Y, Z = Bd / s, eye.expand_as(B).clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y, Z = Y @ T, T @ Z
    return Z / s.sqrt()


def congruence(Bih, C, floor_c=True):
    if floor_c:
        wc, Uc = safe_eigh(C)
        C = (Uc * floor_eigenvalues(wc, EPS).unsqueeze(-2)) @ Uc.transpose(-1, -2)
    M = Bih @ C @ Bih
    lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
    return Bih @ Q, floor_eigenvalues(lam, EPS)


def is_identity(M):
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    return bool(torch.all((M - eye).abs().amax(dim=(-2, -1)) <= 1e-6))


def refresh(mats, share=False, checks=True, floor_c=True, inv_half=inv_half_eigh):
    """Bases per matrix: {k: ((P_in, lam_in), (P_out, lam_out))}, batched by shape as in _refactor."""
    out = {}
    groups = defaultdict(list)
    for k, m in enumerate(mats):
        groups[tuple(m["W"].shape)].append(k)
    cache = {}
    for ks in groups.values():
        W = torch.stack([mats[k]["W"] for k in ks]); A = torch.stack([mats[k]["A"] for k in ks])
        D = torch.stack([mats[k]["D"] for k in ks]); Wt = W.transpose(-1, -2)
        g_out, g_in = W @ A @ Wt, Wt @ D @ W
        if checks:
            is_identity(D); is_identity(g_in)              # the two host syncs of _refactor (never true here)
        P_o, l_o = congruence(inv_half(D), g_out, floor_c)
        if share:
            # inverse root of A once per distinct A tensor
            uniq = {}
            for j, k in enumerate(ks):
                uniq.setdefault(mats[k]["a_id"], []).append(j)
            need = [a for a in uniq if a not in cache]
            if need:
                roots = inv_half(torch.stack([A[uniq[a][0]] for a in need]))
                for a, r in zip(need, roots):
                    cache[a] = r
            Bih = torch.stack([cache[mats[k]["a_id"]] for k in ks])
        else:
            Bih = inv_half(A)
        P_i, l_i = congruence(Bih, g_in, floor_c)
        for j, k in enumerate(ks):
            out[k] = ((P_i[j], l_i[j]), (P_o[j], l_o[j]))
    return out


def direction(P, lam, G):
    Pt = P.transpose(-1, -2)
    den = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
    X = skew(P @ ((Pt @ G @ P) / den) @ Pt)
    return X / X.norm()


def timed(fn, reps=5):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), r


def main():
    mats = load(sys.argv[1])
    variants = {
        "R0 current": dict(),
        "R1 share eigh(A)": dict(share=True),
        "R2 no is_identity": dict(checks=False),
        "R1+R2": dict(share=True, checks=False),
        "R4 R1+R2, C unfloored": dict(share=True, checks=False, floor_c=False),
        "R6 R1+R2, A^-1/2 by Newton-Schulz": dict(share=True, checks=False, inv_half=inv_half_ns),
    }
    res = {"gpu": torch.cuda.get_device_name(0), "checkpoint": sys.argv[1]}
    with exact_fp32():
        refresh(mats)                                      # warm-up
        t0, ref = timed(lambda: refresh(mats))
        X0 = {k: (direction(*ref[k][0], m["M_in"]), direction(*ref[k][1], m["M_out"])) for k, m in enumerate(mats)}

        def report(name, t, bases):
            errs = []
            for k, m in enumerate(mats):
                xi, xo = direction(*bases[k][0], m["M_in"]), direction(*bases[k][1], m["M_out"])
                errs += [float((xi - X0[k][0]).norm()), float((xo - X0[k][1]).norm())]
            errs.sort()
            res[name] = {"s_per_refresh": t, "s_per_step_tfac25": t / 25, "dir_err_median": errs[len(errs) // 2],
                         "dir_err_max": errs[-1]}
            print(f"{name:40s} {t:.3f} s/refresh  {t/25:.4f} s/step  dir err median {errs[len(errs)//2]:.1e} max {errs[-1]:.1e}",
                  flush=True)

        for name, kw in variants.items():
            t, b = timed(lambda: refresh(mats, **kw))
            report(name, t, b)
        prev = torch.backends.cuda.preferred_linalg_library()
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
            refresh(mats, share=True, checks=False)
            t, b = timed(lambda: refresh(mats, share=True, checks=False))
            report("R3 R1+R2 on MAGMA", t, b)
        except Exception as e:
            res["R3 R1+R2 on MAGMA"] = f"failed: {type(e).__name__}: {e}"
            print("R3 failed:", e, flush=True)
        finally:
            torch.backends.cuda.preferred_linalg_library(prev)
    (ROOT / "runs-record" / "probes" / "refresh_bench.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
