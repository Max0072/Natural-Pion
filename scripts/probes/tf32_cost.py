"""What honest fp32 costs on this card, for the matmul shapes the model uses."""
import time
import torch

dev = torch.device("cuda")


def bench(m, k, n, dtype, repeats=50):
    a = torch.randn(m, k, device=dev, dtype=dtype)
    b = torch.randn(k, n, device=dev, dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        a @ b
    torch.cuda.synchronize()
    return repeats * 2 * m * k * n / (time.perf_counter() - t) / 1e12


shapes = [(4096, 4096, 4096), (131072, 512, 512), (131072, 512, 1376), (131072, 1376, 512)]
print(f"{'shape':>26} {'tf32 on':>10} {'tf32 off':>10} {'ratio':>7}")
for m, k, n in shapes:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    on = bench(m, k, n, torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    off = bench(m, k, n, torch.float32)
    print(f"{f'{m}x{k}x{n}':>26} {on:>10.1f} {off:>10.1f} {on/off:>7.1f}x")
