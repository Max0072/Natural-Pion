"""How far is K-FAC from the empirical Fisher, along the direction we step?

`F(X) = 2(S X A + A X S)` comes from replacing `E[(d x^T) (x) (d x^T)]` with
`E[dd^T] (x) E[xx^T]`, which is exact only if the backward signal and the
activations are independent. That assumption is never tested here, and two
approximations sit between our `F` and the curvature `eta* = 2` is derived
from: this one, and the empirical Fisher standing in for the true one. They
have different remedies, so it is worth knowing which dominates.

The comparison needs no full covariance. For a skew `X`,

    <g_b, X> = 2 u_b^T X x_b     with  u_b = W^T d_b

so the exact quadratic form is `E_b[(2 u_b^T X x_b)^2]`, one `(tokens x n)` by
`(n x n)` matmul and a contraction. The K-FAC form is `4 tr(A X^T S X)`. Under
independence they are equal identically; their ratio is the error, measured
where it matters rather than in general.

Run:
    apptainer exec --nv $SIF python scripts/probes/kfac_error.py [checkpoint]
"""

from __future__ import annotations

import os
import sys

import torch

from harness.config import RunConfig
from harness.data import TokenCorpus
from harness.model import Transformer
from ngd_pion.direction import fisher_apply, generators, natural_gradient
from ngd_pion.factorization import basis_congruence, basis_identity_anchor
from ngd_pion.linalg import exact_fp32, is_identity

SEQUENCES = 128          # smaller than a training step; this measures a correlation


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    dev = "cuda"
    cfg = RunConfig()
    torch.manual_seed(cfg.seed)
    model = Transformer(cfg.model).to(dev)
    if len(sys.argv) > 1:
        sd = torch.load(sys.argv[1], map_location=dev, weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"  weights from step {sd.get('step')}")
    else:
        print("  freshly initialised weights")

    linears, _ = model.parameter_split()
    names = {id(m.weight): n for n, m in model.named_modules() if hasattr(m, "weight")}
    chosen = [linears[0], linears[3], linears[5]]        # square, tall, wide

    acts, grads = {}, {}
    handles = []
    for m in chosen:
        def pre(mod, inputs, _m=m):
            acts[id(_m)] = inputs[0].detach()
            return None

        def fwd(mod, inputs, output, _m=m):
            if output.requires_grad:
                output.register_hook(lambda g, k=id(_m): grads.__setitem__(k, g.detach()))
            return None

        handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(fwd)]

    # RunConfig's default is relative; the sbatch scripts pass the real one
    path = os.environ.get("DATA_BIN", f"{os.environ['DATA_p330']}/c4/c4_train.bin")
    data = TokenCorpus(path, cfg.model.seq_len, seed=cfg.seed)
    x, y = data.batch(SEQUENCES, dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    for h in handles:
        h.remove()

    print(f"\n  loss {float(loss):.4f}, {SEQUENCES} sequences = "
          f"{SEQUENCES * cfg.model.seq_len} tokens\n")
    print(f"  {'layer':<24}{'shape':>13}{'K-FAC':>12}{'exact':>12}{'kfac/exact':>12}")
    print("  " + "-" * 73)

    for m in chosen:
        W = m.weight.detach().float()
        G = m.weight.grad.detach().float()
        xs = acts[id(m)].reshape(-1, W.shape[1]).float()
        n_tok = xs.shape[0]
        # autograd hands back dL/dout with the mean's 1/N already in it
        ds = grads[id(m)].reshape(-1, W.shape[0]).float() * n_tok
        us = ds @ W                                     # u_b^T = d_b^T W

        with exact_fp32():
            A = (xs.transpose(0, 1) @ xs) / n_tok
            S = (us.transpose(0, 1) @ us) / n_tok
            gram_in = W.transpose(0, 1) @ W
            basis_in = (
                basis_identity_anchor(A, cfg.ngd_eps)
                if is_identity(gram_in)
                else basis_congruence(A, gram_in, cfg.ngd_eps)
            )
            G_in, _ = generators(W, G)
            X = natural_gradient(G_in, basis_in)

            kfac = float((X * fisher_apply(A, S, X)).sum())
            # <g_b, X> = 2 u_b^T X x_b, one row at a time and never materialised
            per = 2.0 * ((us @ X) * xs).sum(dim=-1)
            exact = float((per * per).mean())

        print(f"  {names.get(id(m.weight), '?'):<24}{str(tuple(W.shape)):>13}"
              f"{kfac:12.4e}{exact:12.4e}{kfac / exact if exact else float('nan'):12.3e}")

    print("\n  a ratio far below 1 means K-FAC underestimates the curvature along X,")
    print("  which inflates F^-1 G by the same factor and pushes eta* down.")


if __name__ == "__main__":
    main()
