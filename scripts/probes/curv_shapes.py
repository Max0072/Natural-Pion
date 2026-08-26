"""quad/curv across all three weight shapes LLaMA-60M actually contains.

square  n_out == n_in : W^T W = I, W A W^T full rank        -- no exact kernel
wide    n_out <  n_in : W^T W is a rank-n_out projector     -- kernel on the IN side
tall    n_out >  n_in : W^T W = I, W A W^T has rank n_in    -- kernel on the OUT side

The tall case is the one not covered by the previous probe.
"""
import sys
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.factorization import build_bases
from ngd_pion.direction import generators, natural_gradient, fisher_apply
from ngd_pion.linalg import is_identity

torch.manual_seed(0)
B, EPS = 4096, 1e-4

def probe(n_out, n_in, tag):
    W64 = torch.linalg.qr(torch.randn(max(n_in, n_out), min(n_in, n_out),
                                      dtype=torch.float64))[0]
    W64 = W64.T.contiguous() if n_out < n_in else W64
    assert W64.shape == (n_out, n_in), W64.shape
    lam = torch.logspace(0, -6, n_in, dtype=torch.float64)
    Qa = torch.linalg.qr(torch.randn(n_in, n_in, dtype=torch.float64))[0]
    Xs = torch.randn(B, n_in, dtype=torch.float64) @ ((Qa * lam.sqrt()) @ Qa.T)
    A64 = (Xs.T @ Xs) / B
    G64 = (torch.randn(B, n_out, dtype=torch.float64).T @ Xs) / B

    gram_in, gram_out = W64.T @ W64, W64 @ A64 @ W64.T
    r_in = int((torch.linalg.eigvalsh(gram_in) > 1e-10).sum())
    r_out = int((torch.linalg.eigvalsh(gram_out) > 1e-10 * gram_out.diag().max()).sum())
    print(f"\n===== W {n_out}x{n_in} ({tag}) =====")
    print(f"  is_identity(W^T W)={bool(is_identity(gram_in))}"
          f"  rank(W^T W)={r_in}/{n_in}  rank(W A W^T)={r_out}/{n_out}")

    for dt in (torch.float64, torch.float32):
        W, A, G = W64.to(dt), A64.to(dt), G64.to(dt)
        b_in, b_out = build_bases(W, A, EPS)
        G_in, G_out = generators(W, G)
        X_in, X_out = natural_gradient(G_in, b_in), natural_gradient(G_out, b_out)
        q_in, q_out = (G_in * X_in).sum(), (G_out * X_out).sum()
        c_in = (X_in * fisher_apply(A, W.T @ W, X_in)).sum()
        c_out = (X_out * fisher_apply(torch.eye(n_out, dtype=dt), W @ A @ W.T, X_out)).sum()
        print(f"  {str(dt).split('.')[-1]:>7}: "
              f"in q={float(q_in):+.4e} c={float(c_in):+.4e} r={float(q_in/c_in):.3e} | "
              f"out q={float(q_out):+.4e} c={float(c_out):+.4e} r={float(q_out/c_out):.3e} | "
              f"total r={float((q_in+q_out)/(c_in+c_out)):.4e}")

probe(768, 768, "square")
probe(192, 768, "wide  -> kernel on IN side")
probe(768, 192, "tall  -> kernel on OUT side")
