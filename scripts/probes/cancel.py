import torch
torch.manual_seed(0)

def sym(M):  return 0.5*(M + M.transpose(-1,-2))
def skew(M): return 0.5*(M - M.transpose(-1,-2))

def gen_two_matmuls(W, G):          # what direction.py does now
    return W.T @ G - G.T @ W

def gen_one_matmul(W, G):           # M - M^T, exactly antisymmetric
    M = W.T @ G
    return M - M.T

n = 512
# W semi-orthogonal as the method keeps it; G with a controllable skew fraction
W = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]

print(f"{'skew/sym':>10}{'two-matmul err':>17}{'one-matmul err':>17}{'ratio':>9}")
print("-"*54)
for frac in (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    # build G so that skew(W^T G) / sym(W^T G) has the target ratio
    S = sym(torch.randn(n, n, dtype=torch.float64))
    K = skew(torch.randn(n, n, dtype=torch.float64))
    K = K * frac * S.norm() / K.norm()
    M_true = S + K
    G = W @ M_true                       # so W^T G = M_true exactly (W orthogonal)
    truth = (M_true - M_true.T)          # = 2*skew, in fp64

    # bf16-level noise on G, as an autocast backward would deliver
    Gn = (G.to(torch.bfloat16).to(torch.float64))
    a = gen_two_matmuls(W.float(), Gn.float()).double()
    b = gen_one_matmul(W.float(), Gn.float()).double()
    ea = (a - truth).norm() / truth.norm()
    eb = (b - truth).norm() / truth.norm()
    print(f"{frac:>10.0e}{ea:>17.3e}{eb:>17.3e}{ea/eb:>9.2f}")

print("\nsame, but with G kept in fp32 (no bf16 round-trip):")
print(f"{'skew/sym':>10}{'two-matmul err':>17}{'one-matmul err':>17}{'ratio':>9}")
print("-"*54)
for frac in (1e-2, 1e-3, 1e-4, 1e-5):
    S = sym(torch.randn(n, n, dtype=torch.float64))
    K = skew(torch.randn(n, n, dtype=torch.float64))
    K = K * frac * S.norm() / K.norm()
    M_true = S + K
    G = W @ M_true
    truth = (M_true - M_true.T)
    a = gen_two_matmuls(W.float(), G.float()).double()
    b = gen_one_matmul(W.float(), G.float()).double()
    ea = (a - truth).norm() / truth.norm()
    eb = (b - truth).norm() / truth.norm()
    print(f"{frac:>10.0e}{ea:>17.3e}{eb:>17.3e}{ea/eb:>9.2f}")
