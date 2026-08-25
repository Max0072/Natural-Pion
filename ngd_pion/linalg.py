"""Primitives shared by the rest of the package.

Nothing here knows about NGD-Pion. Every function is pure, batched over any
leading dimensions, and preserves the dtype and device it is given -- callers
choose precision, this module never converts.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

__all__ = ["skew", "floor_eigenvalues", "floor_spectrum", "cayley", "is_identity",
           "exact_fp32", "spectral_norm", "cayley_newton_schulz"]


@contextmanager
def exact_fp32():
    """Make fp32 mean fp32 for the duration, then put the settings back.

    On an Ampere-or-newer card torch performs fp32 matrix operations in TF32
    by default -- eight bits of exponent, **ten** of mantissa -- and that
    includes the solve inside `cayley`. Measured on an RTX PRO 6000 Blackwell,
    the orthogonality error of the retraction goes from `3e-06` to `4e-03`, and
    the consequence is not cosmetic: over 200 two-sided steps at an angle of
    1e-2 the singular values of `W` move by a **relative 1.0** with TF32 and by
    `2.6e-04` without. A full run is 73242 steps.

    Preserving the spectrum exactly is the whole reason this method uses Cayley
    rather than a truncated exponential, so the guarantee has to be defended
    where it is produced rather than assumed from the caller's environment. A
    CPU test suite cannot see any of this, because TF32 does not exist there.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(precision)


def skew(M: torch.Tensor) -> torch.Tensor:
    """The skew-symmetric part, `(M - M^T) / 2`.

    Used both as a projection and as hygiene: the algebra guarantees several
    intermediate results are skew, and rounding does not, so the projection is
    applied where that guarantee is relied upon.
    """
    return 0.5 * (M - M.transpose(-1, -2))


def floor_eigenvalues(w: torch.Tensor, eps: float) -> torch.Tensor:
    """Raise eigenvalues below `eps * lam_max` to that floor (ALGORITHM.md §4).

    The single regularisation primitive of the method. A floor is the identity
    on every eigenvalue above it, so well-determined directions keep their
    exact scale; a shift (`w + eps`) perturbs those too and pays accuracy
    everywhere for protection only the small end needs.

    `w` is clamped non-negative first: the spectra fed here are PSD by
    construction and any negative value is rounding.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    w = w.clamp_min(0.0)
    lam_max = w.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    return torch.maximum(w, eps * lam_max)


def floor_spectrum(M: torch.Tensor, eps: float) -> torch.Tensor:
    """`floor_eigenvalues` applied to a symmetric matrix, rebuilt in place."""
    w, U = torch.linalg.eigh(M)
    return (U * floor_eigenvalues(w, eps).unsqueeze(-2)) @ U.transpose(-1, -2)


def spectral_norm(
    X: torch.Tensor, iters: int, v: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest singular value of `X` by power iteration on `X^T X`.

    Returns `(sigma, v)`; feeding `v` back on the next call is the whole point.
    Cost is `iters` matrix-vector products, `O(n^2)`, against `O(n^3)` for a
    decomposition -- measured at 4830x cheaper than `matrix_norm(X, 2)` across
    the 56 weights of LLaMA-60M, where the exact call was 95% of the entire
    optimizer step because cusolver falls back to a slow path on 1376x1376.

    **Cold, this converges badly, and that is a property of the input rather
    than of the implementation.** A skew matrix has its singular values in
    equal pairs and the large ones bunch together, so the ratio governing
    convergence sits near 1. Measured on a random 512x512 skew: 21% relative
    error after one iteration, 6% after five, 0.5% after twenty. Warm, from a
    vector converged on the previous step's `X`, one iteration gives `1e-4`.
    So callers must cache `v` and spend iterations generously exactly twice --
    on the first step, and again whenever `X` moves discontinuously.

    The estimate is a Rayleigh quotient and therefore always a *lower* bound.
    For a diagnostic asking whether an angle stays bounded that is the less
    comfortable direction to err in, which is the reason the warm-start
    discipline above is a requirement and not an optimisation.
    """
    if iters < 0:
        raise ValueError(f"iters must be non-negative, got {iters}")
    tiny = torch.finfo(X.dtype).tiny
    if v is None:
        v = torch.ones(*X.shape[:-2], X.shape[-1], dtype=X.dtype, device=X.device)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
    Xt = X.transpose(-1, -2)
    for _ in range(iters):
        v = (Xt @ (X @ v.unsqueeze(-1))).squeeze(-1)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
    return (X @ v.unsqueeze(-1)).squeeze(-1).norm(dim=-1), v


def cayley(X: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """`(I + c/2 X)^-1 (I - c/2 X)`, i.e. `Cayley(-c X)` (ALGORITHM.md §7).

    Exactly orthogonal for skew `X` and any real `c`, and never singular: a
    real skew matrix has purely imaginary eigenvalues, so `I + c/2 X` cannot
    lose rank. Unlike a truncated exponential it also cannot inflate -- its
    spectral norm is exactly 1 at every step size, which is what keeps the
    singular values of `W` fixed rather than merely nearly fixed.

    That last sentence is only true if fp32 means fp32, so this guards itself
    rather than trusting the caller: with TF32 left on, the solve below carries
    ten bits of mantissa and the spectrum moves by a relative 1.0 over 200
    steps. See `exact_fp32`.
    """
    with exact_fp32():
        return _cayley(X, c)


def _cayley(X: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    n = X.shape[-1]
    eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
    half = 0.5 * (c if isinstance(c, torch.Tensor) else torch.as_tensor(c, dtype=X.dtype, device=X.device))
    half = half.reshape(*([1] * (X.dim() - 2)), 1, 1) if isinstance(half, torch.Tensor) and half.dim() else half
    A = half * X
    return torch.linalg.solve(eye + A, eye - A)


def cayley_newton_schulz(
    X: torch.Tensor, c: float | torch.Tensor, iters: int = 2
) -> torch.Tensor:
    """`cayley`, with the inverse by Newton-Schulz instead of a solve.

    `Cayley(-cX) = (I + A)^-1 (I - A) = 2(I + A)^-1 - I` for `A = (c/2)X`, and
    that second form is used here because it costs one matmul less.

    The iteration is `Z <- Z(2I - MZ)`, whose residual `R = I - MZ` **squares**
    every step. Starting from `Z0 = I - A` gives `M Z0 = I - A^2`, so the
    residual begins at `A^2` and runs `A^4`, `A^8`, `A^16`. Rotation angles in
    this method sit near `1e-2`, where two iterations land below the fp32
    epsilon.

    Why bother, when `torch.linalg.solve` computes the same thing exactly: the
    solve is latency-bound, not flop-bound. A 512x512 solve with 512
    right-hand sides is 0.36 GFLOP and takes 1 ms on an RTX PRO 6000
    Blackwell -- 0.36 TFLOPS, where matmuls on the same card run at 18. LU with
    pivoting is sequential and does not fill a GPU at this size, and the step
    does 112 of them. Newton-Schulz costs `4k n^3` of matmul against the
    solve's `2.7 n^3`: three times the arithmetic on hardware fifty times
    faster at it.

    **This converges only for `||A|| < 1`**, and unlike the solve it degrades
    rather than failing loudly -- at `||A|| = 0.5` a single iteration leaves an
    orthogonality error of `3.8e-2`. Callers must bound `(|c|/2) * ||X||_2`,
    which `spectral_norm` already produces for the angle diagnostic, and either
    raise `iters` or fall back to `cayley`. Measured orthogonality error in
    fp32, against `1.1e-6` for the exact solve at the same angle:

    | angle | 1 iter | 2 iters | 3 iters |
    |---|---|---|---|
    | `1e-2` | `1.1e-5` | `6.1e-6` | `6.1e-6` |
    | `1e-1` | `8.3e-6` | `3.7e-6` | `3.7e-6` |
    | `5e-1` | `2.7e-3` | `8.6e-6` | `3.1e-6` |
    | `1.0`  | `3.8e-2` | `1.2e-3` | `4.6e-6` |
    """
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    with exact_fp32():
        n = X.shape[-1]
        eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
        half = c if isinstance(c, torch.Tensor) else torch.as_tensor(
            c, dtype=X.dtype, device=X.device)
        half = 0.5 * half
        if half.dim():
            half = half.reshape(*([1] * (X.dim() - 2)), 1, 1)
        A = half * X
        M = eye + A
        Z = eye - A                       # residual A^2, and it squares
        for _ in range(iters):
            Z = Z @ (2.0 * eye - M @ Z)
        return 2.0 * Z - eye


def is_identity(M: torch.Tensor, atol: float = 1e-6) -> bool:
    """Whether `M` is the identity to `atol`, used to pick the cheap basis path."""
    n = M.shape[-1]
    if M.shape[-2] != n:
        return False
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    return bool(torch.all((M - eye).abs().amax(dim=(-2, -1)) <= atol))
