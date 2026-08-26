"""Does the eigenvalue floor actually matter? Two questions, fp64, CPU only.

Q1. The natural gradient in degenerate directions: the claim is that because
    A = E[x x^T], the gradient's own component along a low-variance direction
    carries sqrt(lam), so ||X|| diverges as eps^(-1/2) rather than eps^(-1).
    Control: the same solve with a random skew G of equal norm, uncorrelated
    with A, which should show the full eps^(-1).

Q2. quad vs curv: quad is measured against the FLOORED operator (through
    basis.denominator), curv against the RAW one (through fisher_apply).
    Prediction: curv <= quad termwise, so alpha >= 1 by construction.
"""
import sys, math
sys.path.insert(0, "/onyx/data/p330/Natural-Pion")
import torch
from ngd_pion.factorization import basis_identity_anchor
from ngd_pion.direction import generators, natural_gradient, fisher_apply

torch.manual_seed(0)
dt = torch.float64
n, B = 256, 512
ORDERS = 8

# A with a designed spectrum spanning ORDERS decades
lam_true = torch.logspace(0, -ORDERS, n, dtype=dt)
Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt))
A = (Q * lam_true) @ Q.T
A = 0.5 * (A + A.T)

# real samples from N(0, A), and a real batch gradient G = sum_b delta_b x_b^T
Ah = (Q * lam_true.sqrt()) @ Q.T
X_samp = torch.randn(B, n, dtype=dt) @ Ah          # x ~ N(0, A)
W, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt))  # square, W^T W = I
delta = torch.randn(B, n, dtype=dt)
G = (delta.T @ X_samp) / B

G_in, _ = generators(W, G)
# control: random skew, same Frobenius norm, no relation to A
R = torch.randn(n, n, dtype=dt)
G_rand = R - R.T
G_rand *= G_in.norm() / G_rand.norm()

print(f"A: n={n} cond={lam_true[0]/lam_true[-1]:.2e}  ||G_in||={G_in.norm():.4e}\n")
print(f"{'eps':>9} {'#floored':>9} | {'|X| real':>11} {'slope':>7} |"
      f" {'|X| rand':>11} {'slope':>7} | {'quad':>11} {'curv':>11} {'q/c':>10}")
print("-" * 104)

prev = {}
for e in [10.0**-k for k in range(2, 13)]:
    b = basis_identity_anchor(A, e)
    n_floored = int((lam_true < e * lam_true[0]).sum())
    row = {}
    for tag, Gx in (("real", G_in), ("rand", G_rand)):
        Xs = natural_gradient(Gx, b)
        row[tag] = float(Xs.norm())
        if tag == "real":
            quad = float((Gx * Xs).sum())
            curv = float((Xs * fisher_apply(A, W.T @ W, Xs)).sum())
    slopes = {}
    for tag in ("real", "rand"):
        # d log||X|| / d log(1/eps)  between consecutive decades
        slopes[tag] = math.log10(row[tag] / prev[tag]) if prev else float("nan")
    prev = dict(row)
    print(f"{e:9.0e} {n_floored:9d} | {row['real']:11.4e} {slopes['real']:7.3f} |"
          f" {row['rand']:11.4e} {slopes['rand']:7.3f} |"
          f" {quad:11.4e} {curv:11.4e} {quad/curv:10.3e}")

print("\nslope = decades of ||X|| gained per decade of eps lowered."
      "\n  0.5 => eps^(-1/2)  (gradient smallness cancels half the divergence)"
      "\n  1.0 => eps^(-1)    (no cancellation at all)")
