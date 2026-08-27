"""Why is quad/curv 1e32 in the real run but only 1.18 on a square toy?

Hypothesis: the real weights are wide (n_out < n_in), so gram_in = W^T W is a
projector with an EXACT null space of dimension n_in - n_out. basis_congruence
floors that null space; fisher_apply uses the raw W^T W, whose kernel is exact.
X therefore lives partly in a subspace the raw operator annihilates, and curv
-- mathematically 4||B^0.5 X C^0.5||_F^2, but summed as n^2 mixed-sign terms --
loses it to cancellation in fp32.

Run the identical computation at both precisions and compare.
"""
import sys
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.factorization import build_bases
from ngd_pion.direction import generators, natural_gradient, fisher_apply
from ngd_pion.linalg import is_identity

torch.manual_seed(0)
n_in, B, EPS = 768, 4096, 1e-4

for n_out in (768, 192):          # square (W^T W = I) vs wide (rank-deficient)
    tag = "square" if n_out == n_in else "WIDE"
    # semi-orthogonal W, exactly as the harness initialises it
    W64 = torch.linalg.qr(torch.randn(n_in, n_out, dtype=torch.float64))[0].T.contiguous()
    lam = torch.logspace(0, -6, n_in, dtype=torch.float64)
    Qa = torch.linalg.qr(torch.randn(n_in, n_in, dtype=torch.float64))[0]
    Ah = (Qa * lam.sqrt()) @ Qa.T
    Xs = torch.randn(B, n_in, dtype=torch.float64) @ Ah
    A64 = (Xs.T @ Xs) / B                       # finite-sample covariance
    delta = torch.randn(B, n_out, dtype=torch.float64)
    G64 = (delta.T @ Xs) / B

    gram = W64.T @ W64
    rank = int((torch.linalg.eigvalsh(gram) > 1e-8).sum())
    print(f"\n===== W {n_out}x{n_in} ({tag}) =====")
    print(f"  is_identity(W^T W) = {bool(is_identity(gram))}   rank(W^T W) = {rank}/{n_in}"
          f"   -> exact nulls: {n_in - rank}")

    for dt in (torch.float64, torch.float32):
        W, A, G = W64.to(dt), A64.to(dt), G64.to(dt)
        b_in, b_out = build_bases(W, A, EPS)
        G_in, G_out = generators(W, G)
        X_in, X_out = natural_gradient(G_in, b_in), natural_gradient(G_out, b_out)
        eye = torch.eye(n_out, dtype=dt)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        c_in = (X_in * fisher_apply(A, W.T @ W, X_in)).sum()
        c_out = (X_out * fisher_apply(eye, W @ A @ W.T, X_out)).sum()
        curv = c_in + c_out
        # the same number, computed as a sum of squares instead (no cancellation)
        Bh = torch.linalg.eigvalsh(A).clamp_min(0)
        term = torch.linalg.matrix_norm(
            _s(A) @ X_in.to(torch.float64) @ _s(W64.T @ W64), "fro") ** 2 * 4 \
            if False else None
        print(f"  {str(dt).split('.')[-1]:>7}: quad={float(quad):+.6e}"
              f"  curv_in={float(c_in):+.6e}  curv_out={float(c_out):+.6e}"
              f"  curv={float(curv):+.6e}  q/c={float(quad/curv):+.4e}")

    # ground truth without cancellation: 4||B^1/2 X C^1/2||_F^2, fp64
    def half(M):
        w, U = torch.linalg.eigh(M)
        return (U * w.clamp_min(0).sqrt()) @ U.T
    b_in, b_out = build_bases(W64, A64, EPS)
    G_in, G_out = generators(W64, G64)
    X_in, X_out = natural_gradient(G_in, b_in), natural_gradient(G_out, b_out)
    t_in = 4 * float(torch.linalg.matrix_norm(half(A64) @ X_in @ half(gram), "fro") ** 2)
    t_out = 4 * float(torch.linalg.matrix_norm(X_out @ half(W64 @ A64 @ W64.T), "fro") ** 2)
    print(f"  fp64 sum-of-squares (no cancellation): curv_in={t_in:.6e}  curv_out={t_out:.6e}")
