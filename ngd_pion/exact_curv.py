"""`FastNGDPionS` that also measures the curvature it is *not* using.

Why
---

`alpha = quad / curv` is identically 1 on a fresh basis, because `curv` is
evaluated with the same Kronecker operator that built `X`: `X = F^-1 G` gives
`<X, F(X)> = <X, G>` as algebra, not as an approximation. So the trust region
reads out basis staleness and nothing else, and the step length falls entirely
to `eta`.

K-FAC does not do this. Martens and Grosse build the direction with the
approximation and choose the length with matrix-vector products against the
*exact* curvature, precisely because judging a step with the operator that
produced it is circular.

Measured on this model (docs/JOURNAL.md, 2026-08-27), the circularity is not
harmless. Two independent routes -- a fit of the reduction ratio `rho` against
`lr` at step 0, and `scripts/probes/kfac_error.py` contracting per-token
quantities directly -- put the Kronecker form 1e2 to 1e3 times below the true
curvature along `X`. That factor is the whole of the 500x between the derived
`eta* = 2` and the swept optimum of 1e-2.

What this class does, and deliberately does not do
--------------------------------------------------

It **measures**, it does not act. `_apply` calls `super()._apply` and then
recomputes `X` to form the diagnostic, so the trajectory is bit-identical to
`FastNGDPionS` and the measurement cannot be blamed for a change in the run.
Wiring the exact curvature into `alpha` is the next step and belongs in its own
class, after this one says what the number is.

The quantity
------------

For one token, `g_b = delta_b x_b^T`, and the first-order weight motion is
`-c V` with `V = X_out W + W X_in`. So

    s_b = <g_b, V> = delta_b^T X_out (W x_b) + (W^T delta_b)^T X_in x_b

`quad = 2 E_b[s_b]` by the lemma in `direction`, and the true second-order
coefficient along the ray is `E_b[s_b^2]`. To keep it in the same units as
`curv` -- so that `alpha_exact` is directly comparable to `alpha`, and the
theoretical optimum stays at `eta = 2` -- the diagnostic reports

    curv_exact = 4 E_b[s_b^2]

Under exact independence of `delta` and `x` this equals `curv` identically,
which is what `tests/test_exact_curv.py` pins.

Three implementation choices, each of which is a way to get this wrong
----------------------------------------------------------------------

* **Paired subsampling.** Retaining every token for every layer is about 15 GB
  against roughly 12 GB of headroom at this configuration. A random `n_tokens`
  of them is enough -- the quantity is an expectation, and 4096 samples give a
  relative error of order 1.6% against an effect of 140x. The index is drawn in
  the forward hook and **reused** in the backward one: `x_b` and `delta_b` must
  be the same tokens or `s_b` is a product of unrelated pairs, which would read
  as a spectacular independence failure and be an indexing bug.
* **`delta` is rescaled.** Autograd hands back `dl_b/dout_b / N` because the
  loss is a mean. `observe_backward` handles this for the covariance with a
  `scale`; here the per-token vector itself is needed, so it is multiplied by
  the token count to recover the per-sample quantity.
* **Only on logged steps.** Nothing is retained on the other steps, so the
  memory and the two extra solves are paid once every `exact_every` steps.

Gradient clipping does not disturb the comparison. The clip scales `G` by `s`,
`X` is linear in `G`, so `quad`, `curv` and `curv_exact` all scale as `s^2` and
both ratios are clip-invariant.
"""

from __future__ import annotations

import torch

from .direction import generators, natural_gradient
from .with_s_fast import FastNGDPionS

__all__ = ["ExactCurvNGDPionS", "exact_curv"]


def exact_curv(
    W: torch.Tensor,
    X_in: torch.Tensor,
    X_out: torch.Tensor,
    xs: torch.Tensor,
    ds: torch.Tensor,
) -> torch.Tensor:
    """`4 E_b[s_b^2]` from paired per-token activations and output gradients.

    Args:
        W: the weight, `(m, n)`.
        X_in: skew generator on the input side, `(n, n)`.
        X_out: skew generator on the output side, `(m, m)`.
        xs: activations entering the layer, `(tokens, n)`.
        ds: **per-sample** output gradients, `(tokens, m)` -- already multiplied
            back up by the token count, not the `1/N` autograd returns.

    Never materialises anything `tokens x tokens`: both terms are a matmul into
    `(tokens, d)` followed by a row-wise contraction.
    """
    ys = xs @ W.transpose(-1, -2)                     # (tokens, m)
    us = ds @ W                                       # (tokens, n)
    s = ((ds @ X_out) * ys).sum(-1) + ((us @ X_in) * xs).sum(-1)
    return 4.0 * (s * s).mean()


class ExactCurvNGDPionS(FastNGDPionS):
    """`FastNGDPionS` plus `curv_exact` and `alpha_exact`, changing nothing.

    Args:
        exact_every: measure once every this many steps. `0` disables it and
            leaves the class behaviourally identical to its parent.
        exact_tokens: how many tokens to retain per layer for the estimate.
    """

    def __init__(self, params, *, exact_every: int = 30, exact_tokens: int = 4096, **kwargs):
        if exact_every < 0:
            raise ValueError(f"exact_every must be non-negative, got {exact_every}")
        if exact_tokens < 1:
            raise ValueError(f"exact_tokens must be at least 1, got {exact_tokens}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["exact_every"] = exact_every
            group["exact_tokens"] = exact_tokens

    # -- retaining a paired sample -------------------------------------------

    def _sampler(self, device: torch.device) -> torch.Generator:
        """One generator per device, seeded once, never the global RNG."""
        cache = self.__dict__.setdefault("_samplers", {})
        key = (device.type, device.index)
        if key not in cache:
            g = torch.Generator(device=device)
            g.manual_seed(0x5EED)
            cache[key] = g
        return cache[key]

    def _sampling(self, param: torch.Tensor) -> bool:
        group = self._group_of(param)
        if not group["exact_every"]:
            return False
        return self.state[param].get("step", 0) % group["exact_every"] == 0

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        super().observe(param, x)
        state = self.state[param]
        if not self._sampling(param):
            for key in ("exact_x", "exact_d", "exact_idx"):
                state.pop(key, None)
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        k = min(self._group_of(param)["exact_tokens"], flat.shape[0])
        # The index is drawn here and kept for the backward hook: the two
        # samples have to be the same tokens.
        #
        # Its own generator, not the global RNG, so that this class really is
        # behaviourally identical to its parent -- a run with the diagnostic on
        # and one with it off must produce the same weights, and sharing the
        # RNG would quietly break that. Sampled with replacement: this is an
        # expectation over tokens, and `randint` is O(k) where `randperm` is
        # O(tokens).
        idx = torch.randint(0, flat.shape[0], (k,), device=flat.device,
                            generator=self._sampler(flat.device))
        state["exact_idx"] = idx
        state["exact_x"] = flat[idx].to(torch.float32)

    def observe_backward(self, param: torch.Tensor, delta: torch.Tensor) -> None:
        super().observe_backward(param, delta)
        state = self.state[param]
        idx = state.get("exact_idx")
        if idx is None or "exact_x" not in state:
            return
        flat = delta.detach().reshape(-1, delta.shape[-1])
        if flat.shape[0] < idx.shape[0]:
            state.pop("exact_x", None)
            return
        # `* n` undoes the `1/N` autograd applies because the loss is a mean.
        state["exact_d"] = flat[idx].to(torch.float32) * float(flat.shape[0])
        state.pop("exact_idx", None)

    # -- the measurement ------------------------------------------------------

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        # The parent takes the step first, so nothing below can alter it. `X`
        # is then rebuilt from the same basis and the same gradient, which
        # costs two extra solves on one step in `exact_every`.
        xs = self.state[p].pop("exact_x", None)
        ds = self.state[p].pop("exact_d", None)
        W = p.detach().to(group["compute_dtype"])
        G = None if p.grad is None else p.grad.detach().to(group["compute_dtype"])
        super()._apply(p, group)

        state = self.state[p]
        if xs is None or ds is None or G is None:
            state.pop("curv_exact", None)
            state.pop("alpha_exact", None)
            return

        G_in, G_out = generators(W, G)
        basis_in, basis_out = state["bases"]
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        ce = exact_curv(W.float(), X_in.float(), X_out.float(), xs, ds)
        state["curv_exact"] = ce
        quad = state["quad"]
        state["alpha_exact"] = (
            float(quad) / float(ce) if float(ce) > 0.0 else float("nan")
        )
