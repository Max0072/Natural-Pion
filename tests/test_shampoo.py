"""What makes Shampoo close on `so(n)`, pinned exactly rather than statistically.

Four of these are machine-precision identities, not tolerance-tuned
approximations, and that is the point: each one is a structural claim from the
module docstring, so a wrong implementation has to violate an identity to pass
rather than merely drift within a tolerance.

The fifth is the oracle check that this package requires of every optimised
path -- torch against a naive numpy transcription.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngd_pion.direction import generators
from ngd_pion.shampoo import ShampooPion, gram, inverse_root
from ngd_pion.shampoo_reference import ShampooPionReference

DT = torch.float64
M, N = 8, 6          # d_out, d_in; both even so a generic skew is full rank


def _skew(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, n, dtype=DT, generator=g)
    return Z - Z.T


def _weights(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(M, N, dtype=DT, generator=g)
    G = torch.randn(M, N, dtype=DT, generator=g) * 0.1
    return W, G


# --- consequence 1: the two Shampoo factors coincide -------------------------


def test_gram_forms_agree_for_skew():
    """`G G^T`, `G^T G` and `-G^2` are the same matrix when `G` is skew.

    This is what halves the state and the eigendecompositions against ordinary
    Shampoo, so it is pinned rather than asserted in prose.
    """
    G = _skew(N, 0)
    assert torch.allclose(gram(G), G.T @ G, atol=1e-12)
    assert torch.allclose(gram(G), -(G @ G), atol=1e-12)


def test_gram_is_psd_symmetric():
    G = _skew(N, 1)
    P = gram(G)
    assert torch.allclose(P, P.T, atol=1e-14)
    assert torch.linalg.eigvalsh(P).min() >= -1e-12


# --- consequence 2: the sandwich stays in the algebra ------------------------


def test_sandwich_is_skew():
    """`P^-p G P^-p` is skew for symmetric `P` and skew `G`.

    Without this the retraction is not a rotation and the singular values of
    `W` are not preserved, which is the whole premise of the method.
    """
    G = _skew(N, 2)
    Q = inverse_root(gram(G), 0.25, 1e-6, "floor")
    X = Q @ G @ Q
    assert torch.allclose(X, -X.T, atol=1e-12)


def test_one_sided_preconditioning_is_not_skew():
    """The negative control for the above: `P^-2p G` leaves the algebra.

    It does so only once the accumulator has mixed directions, and that
    caveat is the whole content of the test. For a `P` built from a *single*
    gradient, `P = -G^2` commutes with `G`, so `(P^-1/2 G)^T = G^T P^-1/2 =
    -P^-1/2 G` and the one-sided form is skew too -- a degenerate coincidence
    of the first step, which an implementation checked only there would pass
    while being wrong from the second step onward.

    So `P` is accumulated over two different gradients, which is the regime the
    optimizer actually runs in, and there the two-sided sandwich stays in
    `so(n)` while the one-sided product does not.
    """
    G1, G2 = _skew(N, 3), _skew(N, 13)
    P = gram(G1) + gram(G2)

    two_sided = inverse_root(P, 0.25, 1e-6, "floor")
    X2 = two_sided @ G1 @ two_sided
    assert torch.allclose(X2, -X2.T, atol=1e-12)

    one_sided = inverse_root(P, 0.5, 1e-6, "floor")
    X1 = one_sided @ G1
    assert not torch.allclose(X1, -X1.T, atol=1e-6)


def test_one_sided_is_skew_only_because_it_commutes_on_the_first_step():
    """Pins the coincidence named above, so it cannot be mistaken for the rule."""
    G = _skew(N, 3)
    one_sided = inverse_root(gram(G), 0.5, 1e-6, "floor") @ G
    assert torch.allclose(one_sided, -one_sided.T, atol=1e-10)


# --- consequence 3: one gradient in, and the generator is orthogonalised ------


def test_single_gradient_orthogonalises():
    """With `P = G G^T`, `power = 1/4` and no damping, every plane turns equally.

    In the real Schur basis the blocks come out `[[0, 1], [-1, 0]]`, so
    `X X^T = I` exactly. Requires an even dimension: a skew matrix has even
    rank, so for odd `n` one plane is always degenerate and the inverse root of
    a zero eigenvalue is not defined.
    """
    G = _skew(N, 4)
    Q = inverse_root(gram(G), 0.25, 0.0, "floor")
    X = Q @ G @ Q
    eye = torch.eye(N, dtype=DT)
    assert torch.allclose(X @ X.T, eye, atol=1e-9)
    # and the singular values, which are the plane angles, are all 1
    s = torch.linalg.svdvals(X)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-9)


# --- the scale invariance the method is being adopted for --------------------


def test_floor_damping_is_scale_invariant():
    """`G -> cG` leaves the step unchanged, which is what fixes the cross-layer
    spread of rotation angle."""
    G = _skew(N, 5)
    for c in (1e-3, 1e3):
        Q1 = inverse_root(gram(G), 0.25, 1e-6, "floor")
        Q2 = inverse_root(gram(c * G), 0.25, 1e-6, "floor")
        X1 = Q1 @ G @ Q1
        X2 = Q2 @ (c * G) @ Q2
        assert torch.allclose(X1, X2, rtol=1e-9, atol=1e-11)


def test_shift_damping_breaks_scale_invariance():
    """The documented cost of `damping="shift"`, pinned as a negative control.

    The original's `eps I` is not homogeneous, so it forfeits exactly the
    property the previous test establishes. Measuring it here is what keeps the
    default an argued choice rather than a silent departure from the original.
    """
    G = _skew(N, 5)
    c = 1e-3
    Q1 = inverse_root(gram(G), 0.25, 1e-2, "shift")
    Q2 = inverse_root(gram(c * G), 0.25, 1e-2, "shift")
    X1 = Q1 @ G @ Q1
    X2 = Q2 @ (c * G) @ Q2
    assert not torch.allclose(X1, X2, rtol=1e-3, atol=1e-6)


# --- the optimizer ------------------------------------------------------------


def test_spectrum_is_preserved():
    """Cayley on both sides freezes the singular values, as it does for Pion."""
    W, G = _weights()
    p = torch.nn.Parameter(W.clone())
    opt = ShampooPion([p], lr=0.1, eps=1e-8, compute_dtype=DT)
    before = torch.linalg.svdvals(p.detach())
    for _ in range(5):
        p.grad = G.clone()
        opt.step()
    after = torch.linalg.svdvals(p.detach())
    assert torch.allclose(before, after, rtol=1e-9, atol=1e-11)


def test_power_zero_is_ablated_pion():
    """`power = 0` leaves the raw generator, which is the natural control arm."""
    W, G = _weights()
    p = torch.nn.Parameter(W.clone())
    opt = ShampooPion([p], lr=0.05, power=0.0, compute_dtype=DT)
    p.grad = G.clone()
    opt.step()

    from ngd_pion.linalg import cayley
    G_in, G_out = generators(W, G)
    expected = cayley(G_out, 0.05) @ W @ cayley(G_in, 0.05)
    assert torch.allclose(p.detach(), expected, atol=1e-12)


@pytest.mark.parametrize("damping", ["floor", "shift"])
@pytest.mark.parametrize("beta", [0.0, 0.9])
def test_matches_the_numpy_oracle(damping, beta):
    """The torch path is correct exactly insofar as it reproduces the oracle."""
    W, G = _weights(seed=7)
    p = torch.nn.Parameter(W.clone())
    opt = ShampooPion(
        [p], lr=0.05, power=0.25, beta=beta, eps=1e-6,
        damping=damping, t_fac=3, compute_dtype=DT,
    )
    ref = ShampooPionReference(
        W.numpy(), lr=0.05, power=0.25, beta=beta, eps=1e-6,
        damping=damping, t_fac=3,
    )
    for k in range(7):
        Gk = (G * (1.0 + 0.3 * k)).contiguous()
        p.grad = Gk.clone()
        opt.step()
        ref.step(Gk.numpy())
        assert np.allclose(p.detach().numpy(), ref.W, rtol=1e-8, atol=1e-10), f"step {k}"


def test_rejects_non_2d_and_bad_damping():
    p = torch.nn.Parameter(torch.zeros(4, dtype=DT))
    with pytest.raises(ValueError):
        ShampooPion([p])
    q = torch.nn.Parameter(torch.zeros(4, 4, dtype=DT))
    with pytest.raises(ValueError):
        ShampooPion([q], damping="tikhonov")


# --- the wiring, which is where this project's last two config bugs lived -----


def test_every_config_field_reaches_the_optimizer():
    """`shampoo_*` must arrive in the optimizer, not merely in the run hash.

    `ngd_power` reaches the manifest, the configuration hash and the directory
    name without reaching the optimizer, so it reads as a controlled variable
    in six runs' worth of results and is not one. That is a wiring bug a unit
    test can catch and a training run cannot, so it is caught here for the one
    optimizer added after the bug was understood.
    """
    from dataclasses import replace

    from harness.config import RunConfig
    from harness.model import ModelConfig, Transformer
    from harness.train import build_optimizers

    small = ModelConfig(vocab_size=256, hidden=64, layers=2, heads=2, ffn_hidden=176, seq_len=32)
    cfg = RunConfig(optimizer="shampoo-pion", model=small)
    cfg = replace(
        cfg,
        lr=0.037,
        shampoo_power=0.3,
        shampoo_beta=0.7,
        shampoo_eps=3e-5,
        shampoo_damping="shift",
        ngd_t_fac=11,
    )
    rot, adamw, recorder = build_optimizers(Transformer(small), cfg)

    assert isinstance(rot, ShampooPion)
    assert recorder is None, "Shampoo needs no activations and must not attach hooks"
    group = rot.param_groups[0]
    assert group["lr"] == 0.037
    assert group["power"] == 0.3
    assert group["beta"] == 0.7
    assert group["eps"] == 3e-5
    assert group["damping"] == "shift"
    assert group["t_fac"] == 11


def test_changing_a_field_changes_the_step():
    """The stronger form: a field that reaches the group must also move the run.

    Reaching `param_groups` is necessary and not sufficient -- `Basis.power`
    reached its dataclass too. This asserts the weights actually differ.
    """
    W, G = _weights(seed=3)
    out = []
    for power in (0.25, 0.4):
        p = torch.nn.Parameter(W.clone())
        opt = ShampooPion([p], lr=0.05, power=power, eps=1e-6, compute_dtype=DT)
        for _ in range(3):
            p.grad = G.clone()
            opt.step()
        out.append(p.detach().clone())
    assert not torch.allclose(out[0], out[1], atol=1e-8)


def test_plane_angles_via_eigvalsh_match_svdvals():
    """The diagnostic's cheaper, better-conditioned route gives the same numbers.

    `X X^T` for skew `X` has eigenvalues `th^2`, the squared plane angles, each
    twice. Going through `eigvalsh` avoids handing an iterative SVD a spectrum
    of exactly repeated singular values -- which is what made cusolver fall
    back to an exact method on the first real run.
    """
    X = _skew(N, 11)
    via_eigh = torch.linalg.eigvalsh(gram(X)).clamp_min(0.0).sqrt()
    via_svd = torch.linalg.svdvals(X)
    assert torch.allclose(via_eigh.sort().values, via_svd.sort().values, atol=1e-10)
