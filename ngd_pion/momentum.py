"""ALGORITHM.md §9 -- first-moment averaging, the half of Adam we did not have.

Why this exists
---------------

`A` and `D` are EMAs, so the method already carries memory across steps, and
the natural objection is that this is already a kind of momentum. It is not,
and the distinction is which side of the quotient the memory sits on:

    here        X = EMA(F)^-1 G      the *metric* is averaged
    momentum    X = F^-1 EMA(G)      the *direction* is averaged

Averaging the denominator stabilises the geometry. It cannot cancel zero-mean
noise in the numerator, because it enters multiplicatively.

The identification with Adam is exact rather than by analogy. `powered.py`
establishes that in the basis, `F` acts elementwise with eigenvalues `d_ij`,
and `d_ij` **is** the second moment of that generator component, since `F` is
by construction the covariance of the generators. So `F` is Adam's `v`, in
Kronecker-factored matrix form, and NGD-Pion-S has been RMSProp all along:
second-moment memory, no first-moment memory. This class supplies the `m`.

Three measurements point the same way, which is why the knob is worth adding
rather than merely available:

* `ALGORITHM.md` records a split-half in which the step disagrees by 47%
  between two independent samples, with the note that the Fisher reweights
  that variance but does not average it. (Measured at `d_out = 256` on ~80k
  tokens; `scripts/probes/split_half_step.py` re-measures it on the real model.)
* Per-component `|E[g]| / sqrt(E[g^2])` peaks at 0.035 against `1/sqrt(N)` of
  2.8e-3, so the step is mostly sampling noise, and `docs/JOURNAL.md` derives
  `eta* ~ 2 SNR^2` from it.
* Within our own runs the treatment is asymmetric: AdamW carries `beta1 = 0.9`
  over the embedding, the head and the norm gains -- 56.5% of the model -- while
  the 43.5% that Pion owns get no first-moment averaging at all.

What is taken from Pion, and what is not
----------------------------------------

`momentum="lie"` keeps separate buffers on the two generators; `"ambient"`
smooths the raw gradient before the generators are extracted. Both are Pion's,
and `"lie"` is the default here because it is the variant their only published
60M figures come from. The buffers stay skew because the generators are, so the
smoothed result is still a Lie-algebra element and needs no projection.

**Their second moment is deliberately not taken.** `Pion._smooth_lie` can divide
`m` by `sqrt(v)`; doing that here would divide by the second moment twice, once
through `v` and once through `F`. This class supplies `m` and nothing else.

**No bias correction, as in theirs.** The buffers start at zero, so the first
few steps take a step scaled by roughly `1 - beta1^t`. At `beta1 = 0.9` that is
ten steps out of 73242 and it is stated rather than corrected, because matching
their mechanism is the point of the comparison.

The prediction, before the run
------------------------------

An EMA with factor `beta` averages `k = (1 + beta) / (1 - beta)` independent
draws and so divides the sampling variance by `k`. If the step's noise is what
holds `eta` down -- the `eta* ~ 2 SNR^2` account -- then `eta*` should rise by
roughly `k` when momentum is switched on. `beta1 = 0.9` gives `k = 19`. If
`eta*` does not move, that account of `eta` is wrong, which is worth knowing
separately from whether the loss improves.
"""

from __future__ import annotations

import torch

from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .linalg import cayley, spectral_norm
from .with_s_fast import FastNGDPionS

__all__ = ["MomentumNGDPionS"]


class MomentumNGDPionS(FastNGDPionS):
    """`FastNGDPionS` with a first moment on the direction.

    Args:
        momentum: `"lie"` buffers the two generators separately, `"ambient"`
            buffers the raw gradient before they are extracted, `"none"`
            reproduces `FastNGDPionS` exactly -- which `tests/test_momentum.py`
            pins as a bit-identical trajectory, so the copy of `_apply` below
            cannot silently drift from its parent.
        beta1: EMA factor. `0.9` is Pion's and Adam's.
    """

    def __init__(self, params, *, momentum: str = "lie", beta1: float = 0.9, **kwargs):
        if momentum not in ("lie", "ambient", "none"):
            raise ValueError(
                f"momentum must be 'lie', 'ambient' or 'none', got {momentum!r}"
            )
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["momentum"] = momentum
            group["beta1"] = beta1

    @staticmethod
    def _ema(state: dict, key: str, g: torch.Tensor, beta: float) -> torch.Tensor:
        if key not in state:
            state[key] = torch.zeros_like(g)
        state[key].mul_(beta).add_(g, alpha=1.0 - beta)
        return state[key]

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]
        mode, beta1 = group["momentum"], group["beta1"]

        if mode == "ambient":
            G = self._ema(state, "m_ambient", G, beta1)
        G_in, G_out = generators(W, G)
        if mode == "lie":
            G_in = self._ema(state, "m_in", G_in, beta1)
            G_out = self._ema(state, "m_out", G_out, beta1)

        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        D = state["cov_backward"].matrix.to(device=p.device, dtype=dt)
        Wt = W.transpose(-1, -2)
        # `quad` and `curv` are formed from the *smoothed* generators, which is
        # the consistent choice: `alpha` is the ratio of the predicted decrease
        # to the curvature along the step actually taken, and the step is built
        # from `G_in`, `G_out` as they stand here. Using the raw gradient for
        # `quad` and the smoothed one for the step would make `alpha` a ratio
        # between two different directions.
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, Wt @ D @ W, X_in)).sum() + (
            X_out * fisher_apply(D, W @ A @ Wt, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)
        state["quad"] = quad
        state["curv"] = curv

        c = group["lr"] * float(alpha)
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        sigma_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        sigma_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        state["angle"] = c * torch.maximum(sigma_in, sigma_out)
        state["pred_drop"] = c * quad

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1

    def load_state_dict(self, state_dict) -> None:
        """Put the momentum buffers back on the parameter's device.

        Same failure `NGDPion.load_state_dict` guards against: a resumed run
        maps to CPU and torch relocates only what it recognises.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                for key in ("m_ambient", "m_in", "m_out"):
                    t = st.get(key)
                    if isinstance(t, torch.Tensor):
                        st[key] = t.to(p.device)
