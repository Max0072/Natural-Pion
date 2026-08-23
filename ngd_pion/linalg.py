"""Primitives shared by the rest of the package.

Nothing here knows about NGD-Pion. Every function is pure, batched over any
leading dimensions, and preserves the dtype and device it is given -- callers
choose precision, this module never converts.
"""

from __future__ import annotations

import torch

__all__ = ["skew", "floor_eigenvalues", "floor_spectrum", "cayley", "is_identity"]


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


def cayley(X: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """`(I + c/2 X)^-1 (I - c/2 X)`, i.e. `Cayley(-c X)` (ALGORITHM.md §7).

    Exactly orthogonal for skew `X` and any real `c`, and never singular: a
    real skew matrix has purely imaginary eigenvalues, so `I + c/2 X` cannot
    lose rank. Unlike a truncated exponential it also cannot inflate -- its
    spectral norm is exactly 1 at every step size, which is what keeps the
    singular values of `W` fixed rather than merely nearly fixed.
    """
    n = X.shape[-1]
    eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
    half = 0.5 * (c if isinstance(c, torch.Tensor) else torch.as_tensor(c, dtype=X.dtype, device=X.device))
    half = half.reshape(*([1] * (X.dim() - 2)), 1, 1) if isinstance(half, torch.Tensor) and half.dim() else half
    A = half * X
    return torch.linalg.solve(eye + A, eye - A)


def is_identity(M: torch.Tensor, atol: float = 1e-6) -> bool:
    """Whether `M` is the identity to `atol`, used to pick the cheap basis path."""
    n = M.shape[-1]
    if M.shape[-2] != n:
        return False
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    return bool(torch.all((M - eye).abs().amax(dim=(-2, -1)) <= atol))
