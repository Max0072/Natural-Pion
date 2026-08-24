"""Is the GPU's fp32 linear algebra actually fp32, and does it matter for the spectrum?"""
import time
import torch

dev = torch.device("cuda")


def flags(label):
    print(f"--- {label}")
    print(f"    matmul.allow_tf32        {torch.backends.cuda.matmul.allow_tf32}")
    print(f"    cudnn.allow_tf32         {torch.backends.cudnn.allow_tf32}")
    try:
        print(f"    float32_matmul_precision {torch.get_float32_matmul_precision()}")
    except Exception:
        pass
    for name in ("matmul", "linalg"):
        try:
            print(f"    cuda.{name}.fp32_precision   {getattr(torch.backends.cuda, name).fp32_precision}")
        except Exception:
            pass


def skew(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n, generator=g).to(dev)
    return 0.5 * (x - x.T)


def cayley(X, c):
    I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(I + 0.5 * c * X, I - 0.5 * c * X)


def orth_error(n, c, dtype):
    X = skew(n, 0).to(dtype)
    R = cayley(X, c)
    I = torch.eye(n, device=dev, dtype=dtype)
    return (R.T @ R - I).abs().max().item()


def spectrum_drift(steps, angle, dtype):
    """What the paper actually claims: the singular values do not move."""
    torch.manual_seed(1)
    W = (torch.randn(512, 512, device=dev) * 0.02).to(dtype)
    s0 = torch.linalg.svdvals(W.double())
    for k in range(steps):
        A, B = skew(512, 100 + k).to(dtype), skew(512, 900 + k).to(dtype)
        c = angle / torch.linalg.matrix_norm(A, 2)
        W = cayley(A, c) @ W @ cayley(B, c)
    s1 = torch.linalg.svdvals(W.double())
    return ((s1 - s0).abs() / s0).max().item()


def bench_eigh(n, repeats=10):
    m = torch.randn(n, n, device=dev)
    m = m @ m.T / n + torch.eye(n, device=dev)
    for _ in range(3):                      # warm up: the first call pays cuSOLVER's setup
        torch.linalg.eigh(m)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        torch.linalg.eigh(m)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")

flags("as the harness finds them")
print(f"\n{'c':>8} {'orth err fp32':>15} {'orth err fp64':>15}")
for c in (1.0, 1e-2, 1e-3):
    print(f"{c:>8.0e} {orth_error(512, c, torch.float32):>15.1e} {orth_error(512, c, torch.float64):>15.1e}")

print("\nnow forcing true fp32")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception as exc:
    print("   (set_float32_matmul_precision:", exc, ")")
for name in ("matmul", "linalg"):
    try:
        getattr(torch.backends.cuda, name).fp32_precision = "ieee"
    except Exception as exc:
        print(f"   (cuda.{name}.fp32_precision: {exc})")
flags("after forcing")
print(f"\n{'c':>8} {'orth err fp32':>15}")
for c in (1.0, 1e-2, 1e-3):
    print(f"{c:>8.0e} {orth_error(512, c, torch.float32):>15.1e}")

print("\nspectrum after 200 two-sided steps at angle 1e-2 (relative, max over 512 values)")
print(f"    forced fp32   {spectrum_drift(200, 1e-2, torch.float32):.2e}")
print(f"    fp64          {spectrum_drift(200, 1e-2, torch.float64):.2e}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
print(f"    tf32 allowed  {spectrum_drift(200, 1e-2, torch.float32):.2e}")

print("\neigh, warmed up, mean of 10")
torch.backends.cuda.matmul.allow_tf32 = False
for n in (512, 1376):
    print(f"    {n:>5}  {bench_eigh(n):7.1f} ms")
