"""What `exact_curv` computes, pinned against a closed form.

The quantity is `4 E_b[s_b^2]` with `s_b = <g_b, X_out W + W X_in>` for the
per-token gradient `g_b = delta_b x_b^T`. Written out,

    a_b = delta_b^T X_out W x_b        b_b = (W^T delta_b)^T X_in x_b

and under independence of `delta` and `x` the three second moments have closed
forms in `A = E[xx^T]` and `D = E[dd^T]`:

    E[a^2]  = tr(D X_out A' X_out^T)          A' = W A W^T
    E[b^2]  = tr(S' X_in A X_in^T)            S' = W^T D W
    E[ab]   = tr(D X_out W A X_in^T W^T)

The tests are **exact, not statistical**. Taking every pair of a small set of
activations with a small set of output gradients makes the empirical joint
distribution equal the product of the empirical marginals by construction, so
the closed form and the per-token average must agree to machine precision.
A sampling-based version would only agree to `O(1/sqrt(N))` and would not
distinguish a real error from noise.
"""

from __future__ import annotations

import pytest
import torch

from ngd_pion.direction import fisher_apply
from ngd_pion.exact_curv import exact_curv

DT = torch.float64
M, N = 5, 6          # d_out, d_in
NX, ND = 40, 40


def _setup(seed: int = 0):
    """`W`, skew `X`, and a token list holding every `(x, delta)` pair."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(M, N, dtype=DT, generator=g)
    Z = torch.randn(N, N, dtype=DT, generator=g)
    X_in = Z - Z.T
    Z = torch.randn(M, M, dtype=DT, generator=g)
    X_out = Z - Z.T

    x = torch.randn(NX, N, dtype=DT, generator=g)
    d = torch.randn(ND, M, dtype=DT, generator=g) * 0.3
    # every pair: xs[i*ND + j] = x[i], ds[i*ND + j] = d[j]
    xs = x.repeat_interleave(ND, dim=0)
    ds = d.repeat(NX, 1)

    A = x.T @ x / NX
    D = d.T @ d / ND
    return W, X_in, X_out, xs, ds, A, D


def _moments(W, X_in, X_out, A, D):
    A_prime = W @ A @ W.T
    S_prime = W.T @ D @ W
    ea2 = torch.trace(D @ X_out @ A_prime @ X_out.T)
    eb2 = torch.trace(S_prime @ X_in @ A @ X_in.T)
    eab = torch.trace(D @ X_out @ W @ A @ X_in.T @ W.T)
    return float(ea2), float(eb2), float(eab)


def test_matches_the_closed_form_exactly():
    W, X_in, X_out, xs, ds, A, D = _setup()
    ea2, eb2, eab = _moments(W, X_in, X_out, A, D)
    want = 4.0 * (ea2 + eb2 + 2.0 * eab)
    got = float(exact_curv(W, X_in, X_out, xs, ds))
    assert got == pytest.approx(want, rel=1e-10)


def test_kfac_curv_is_the_same_thing_without_the_cross_term():
    """`<X, F(X)>` summed over both sides is exactly `4 (E[a^2] + E[b^2])`.

    This is the identity the whole `alpha = quad/curv` argument rests on, and
    it is what makes `curv_exact` directly comparable to `curv`: same units,
    same factor of four, so the theoretical optimum stays at `eta = 2`.
    """
    W, X_in, X_out, xs, ds, A, D = _setup()
    ea2, eb2, _ = _moments(W, X_in, X_out, A, D)
    kfac = float((X_in * fisher_apply(A, W.T @ D @ W, X_in)).sum()) + float(
        (X_out * fisher_apply(D, W @ A @ W.T, X_out)).sum()
    )
    assert kfac == pytest.approx(4.0 * (ea2 + eb2), rel=1e-10)


def test_the_cross_term_is_real_and_bounded():
    """The negative control: independence alone does not make the two equal.

    `curv` and `curv_exact` differ by `8 E[ab]` even when `delta` and `x` are
    exactly independent, so a test asserting they are equal would fail for the
    right reason and be read as a bug. Cauchy-Schwarz also bounds how far apart
    they can get: `2|E[ab]| <= E[a^2] + E[b^2]`, hence
    `curv_exact / curv` lies in `[0, 2]` -- which is why the cross term cannot
    explain a measured ratio of 1e-3 and is excluded as a candidate without any
    run at all.
    """
    W, X_in, X_out, xs, ds, A, D = _setup()
    ea2, eb2, eab = _moments(W, X_in, X_out, A, D)
    kfac = 4.0 * (ea2 + eb2)
    got = float(exact_curv(W, X_in, X_out, xs, ds))

    assert abs(2.0 * eab) > 1e-3 * (ea2 + eb2)          # genuinely non-zero
    assert abs(2.0 * eab) <= (ea2 + eb2) * (1 + 1e-12)  # Cauchy-Schwarz
    assert 0.0 <= got / kfac <= 2.0 + 1e-12


def test_pairing_matters():
    """Shuffling `ds` against `xs` breaks `s_b` and must change the answer.

    The one indexing mistake this class can make is drawing the activation
    sample and the gradient sample with different indices. That would read as
    an enormous independence failure rather than as a bug, so it is pinned:
    with the pairing destroyed the value moves, and it moves toward the
    independent (Kronecker) answer, since shuffled pairs *are* independent.
    """
    W, X_in, X_out, xs, ds, A, D = _setup()
    ea2, eb2, eab = _moments(W, X_in, X_out, A, D)
    paired = float(exact_curv(W, X_in, X_out, xs, ds))

    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(ds.shape[0], generator=g)
    shuffled = float(exact_curv(W, X_in, X_out, xs, ds[perm]))

    assert shuffled != pytest.approx(paired, rel=1e-6)
    # a permutation of an all-pairs list is still all-pairs in distribution,
    # so the shuffled value stays near the closed form rather than wandering
    assert shuffled == pytest.approx(4.0 * (ea2 + eb2 + 2 * eab), rel=0.5)


def test_scales_quadratically_with_the_generator():
    """`curv_exact` is a quadratic form in `X`, so doubling `X` quadruples it."""
    W, X_in, X_out, xs, ds, A, D = _setup()
    one = float(exact_curv(W, X_in, X_out, xs, ds))
    two = float(exact_curv(W, 2 * X_in, 2 * X_out, xs, ds))
    assert two == pytest.approx(4.0 * one, rel=1e-10)
