"""Primitives: each property they are relied upon for, checked directly."""

import numpy as np
import pytest
import torch

from ngd_pion.linalg import cayley, floor_eigenvalues, floor_spectrum, is_identity, skew

torch.manual_seed(0)


def rand_skew(n, batch=()):
    M = torch.randn(*batch, n, n, dtype=torch.float64)
    return skew(M)


def test_skew_is_a_projection():
    M = torch.randn(7, 7, dtype=torch.float64)
    S = skew(M)
    assert torch.allclose(S, -S.transpose(-1, -2))
    assert torch.allclose(skew(S), S)


def test_floor_is_identity_above_the_threshold():
    """The defining property: a floor must not move well-determined eigenvalues."""
    w = torch.tensor([1.0, 1e-2, 1e-5, 1e-9, 0.0], dtype=torch.float64)
    out = floor_eigenvalues(w, 1e-4)
    assert out[0] == w[0] and out[1] == w[1]           # above the floor, untouched
    assert torch.all(out[2:] == 1e-4)                  # below it, raised to it


def test_floor_clamps_negative_rounding():
    w = torch.tensor([1.0, -1e-18], dtype=torch.float64)
    assert torch.all(floor_eigenvalues(w, 0.0) >= 0.0)


def test_floor_rejects_negative_eps():
    with pytest.raises(ValueError):
        floor_eigenvalues(torch.ones(3, dtype=torch.float64), -1e-8)


def test_floor_batches_per_matrix():
    """Each matrix in a batch is floored against its own lam_max, not a shared one."""
    w = torch.tensor([[1.0, 1e-9], [1e-6, 1e-15]], dtype=torch.float64)
    out = floor_eigenvalues(w, 1e-3)
    assert out[0, 1] == pytest.approx(1e-3)
    assert out[1, 1] == pytest.approx(1e-9)


def test_floor_spectrum_matches_eigenvalue_floor():
    U = torch.linalg.qr(torch.randn(9, 9, dtype=torch.float64))[0]
    w = torch.tensor([1.0, 0.5, 1e-3, 1e-7, 1e-9, 0.0, 2.0, 0.3, 1e-11], dtype=torch.float64)
    M = (U * w) @ U.T
    got = torch.linalg.eigvalsh(floor_spectrum(M, 1e-4))
    want = torch.sort(floor_eigenvalues(w, 1e-4)).values
    assert torch.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("c", [1e-6, 0.1, 1.0, 50.0])
def test_cayley_is_exactly_orthogonal_at_any_step(c):
    """The property Pion's truncated exponential only has to third order."""
    for n in (5, 16):
        R = cayley(rand_skew(n), c)
        eye = torch.eye(n, dtype=torch.float64)
        assert (R.T @ R - eye).abs().max() < 1e-12
        assert abs(torch.linalg.matrix_norm(R, 2).item() - 1.0) < 1e-12


def test_cayley_never_inflates_where_truncated_exp_does():
    """At a large angle the degree-2 exponential blows up; Cayley cannot."""
    X = rand_skew(32)
    X = X / torch.linalg.matrix_norm(X, 2)
    c = 5.0
    trunc = torch.eye(32, dtype=torch.float64) - c * X + 0.5 * (c * X) @ (c * X)
    assert torch.linalg.matrix_norm(trunc, 2) > 5.0
    assert abs(torch.linalg.matrix_norm(cayley(X, c), 2).item() - 1.0) < 1e-12


def test_cayley_preserves_singular_values():
    W = torch.randn(6, 11, dtype=torch.float64)
    s0 = torch.linalg.svdvals(W)
    W2 = cayley(rand_skew(6), 0.7) @ W @ cayley(rand_skew(11), 0.7)
    assert (torch.linalg.svdvals(W2) - s0).abs().max() < 1e-10


def test_cayley_batches():
    X = rand_skew(8, batch=(4,))
    R = cayley(X, 0.3)
    for i in range(4):
        assert torch.allclose(R[i], cayley(X[i], 0.3))


def test_cayley_agrees_with_reference():
    from ngd_pion.reference import cayley as ref_cayley

    X = rand_skew(12)
    got = cayley(X, 0.37).numpy()
    want = ref_cayley(X.numpy(), 0.37)
    assert np.abs(got - want).max() < 1e-13


def test_is_identity():
    assert is_identity(torch.eye(5, dtype=torch.float64))
    assert not is_identity(2.0 * torch.eye(5, dtype=torch.float64))
    assert not is_identity(torch.randn(4, 6, dtype=torch.float64))
