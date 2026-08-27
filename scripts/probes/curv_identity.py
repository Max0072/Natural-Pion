import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

def skew(M): return 0.5*(M - M.transpose(-1,-2))
def fisher_apply(B,C,X): return 2.0*(B@X@C + C@X@B)

for trial in range(5):
    n = 64
    B = torch.randn(n,n); B = B@B.T          # symmetric PSD
    C = torch.randn(n,n); C = C@C.T          # symmetric PSD
    X = skew(torch.randn(n,n))               # skew

    current = (X * fisher_apply(B,C,X)).sum()
    cheap   = 4.0 * ((B@X) * (X@C)).sum()
    print(f"trial {trial}: current={current:.12e} cheap={cheap:.12e} "
          f"rel={abs(current-cheap)/abs(current):.3e}")

# and check it still holds when B is the identity (the out-side case)
n = 64
B = torch.eye(n)
C = torch.randn(n,n); C = C@C.T
X = skew(torch.randn(n,n))
print("identity-B:", float((X*fisher_apply(B,C,X)).sum()), float(4.0*((B@X)*(X@C)).sum()))
