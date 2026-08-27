"""ALGORITHM.md §8 -- Shampoo's preconditioner, on the skew-symmetric space.

Shampoo (Gupta, Koren, Singer 2018) preconditions a matrix gradient from both
sides with running Gram matrices of the gradient itself:

    L = sum_t G_t G_t^T      R = sum_t G_t^T G_t      step = L^-1/4 G R^-1/4

That is a different object from everything else in this package. `optimizer.py`
and `with_s.py` build their Kronecker factors from *network statistics* -- the
input covariance `A = E[x x^T]` and the backward covariance `D = E[dd^T]` --
and invert an approximation to the Fisher. Shampoo never looks at an activation
and never assumes anything about the model: its factors come from the gradients
it has already seen. In particular there is no `delta ⊥ x` independence
assumption, which is the approximation measured on this model to be the whole
of the missing factor (`kfac/exact = 0.0128`, docs/JOURNAL.md 2026-08-27).

`powered.py` raises the *Fisher's* eigenvalues to a power and its docstring
calls `power = 1/2` "Adam, Shampoo". That identification is loose and this
module is not a special case of it: the exponents agree arithmetically, the
matrices being raised to them do not.

Why this fits so(n) particularly well
-------------------------------------

The object being preconditioned here is not a weight-space gradient. It is a
rotation generator: `generators` returns `G_in = W^T G - G^T W` in `so(n)` and
`G_out = G W^T - W G^T` in `so(m)`, both skew by construction. Three
consequences, and all three are in our favour.

**1. The two factors coincide.** For skew `G`, `G^T = -G`, so

    L = G G^T = -G^2 = G^T G = R

exactly, not approximately. One accumulator per side instead of two, one
`eigh` instead of two, half the state. `P` is formed as `G G^T` rather than as
`-G G` because the Gram form is manifestly symmetric PSD under rounding; that
the two agree for skew input is pinned by a test rather than assumed.

**2. The two-sided form is what preserves the algebra.** With `P` symmetric and
`G` skew,

    (P^-p G P^-p)^T = P^-p G^T P^-p = -P^-p G P^-p

so the preconditioned step is still skew, still a valid rotation generator, and
Cayley still returns an exactly orthogonal matrix -- which is what freezes the
singular values of `W`. So Shampoo's characteristic two-sided sandwich is not
one option among many here; it is the one that closes on `so(n)`.

The one-sided `P^-2p G` does **not**, and the exception is worth stating
because it is a trap. On the very first step `P = G G^T = -G^2` commutes with
`G`, so `(P^-1/2 G)^T = G^T P^-1/2 = -P^-1/2 G` and the one-sided form is skew
as well. That is a coincidence of a single accumulated gradient and it dies as
soon as the accumulator mixes directions. An implementation validated only at
step 0 would look correct and leave the algebra from step 1 onward, so both
halves are pinned in `test_shampoo.py`.

**3. One gradient in, and the step is the orthogonalised generator.** In the
real Schur basis a skew `G` is block diagonal with blocks `[[0, th],[-th, 0]]`.
Then `P = G G^T` has eigenvalues `th^2` (twice each), `P^-1/4` has `th^-1/2`,
and the sandwich gives blocks

    th^-1/2 * th * th^-1/2 = 1

-- *every rotation plane turns by the same angle*. Single-step Shampoo on
`so(n)` is exactly the orthogonalisation of the generator, which is the `so(n)`
analogue of what Muon does to a weight-space gradient, and accumulation over
`t` interpolates from there back toward the raw gradient. `test_shampoo.py`
pins this to machine precision at `eps = 0`.

What it is expected to fix, stated before it was run
----------------------------------------------------

The full-length run's diagnosed defect is that per-layer rotation angle spans
4658x to 36496x *within one step*: the Fisher is block diagonal per layer, so
it equalises curvature inside a block and says nothing about scale between
blocks, while one scalar `eta` is applied to all of them. Pion buys that
calibration by fiat with `scaling="rms"`; NGD-Pion has nothing.

Shampoo gets it structurally, because the sandwich is scale free. Under
`G -> cG` for a whole layer, `P -> c^2 P`, `P^-1/4 -> c^-1/2`, and

    c^-1/2 * c * c^-1/2 = 1

so the step is invariant. Both the spread *within* a layer (consequence 3) and
the scale *between* layers (this) collapse, with no hooks and no `rms`.

**The damping choice can destroy exactly this, which is why it is a field.**
The original adds `eps I` to the accumulator. An absolute shift is not
homogeneous -- `(c^2 P + eps I)^-1/4` does not scale as `c^-1/2` -- so it
breaks the invariance the method is being adopted for, most severely early on
when `P` is still small and low rank. The relative floor `max(lam, eps lam_max)`
this package already uses everywhere is homogeneous and preserves it exactly,
and was separately measured 134x more accurate than a shift on a wide layer.
`damping="floor"` is therefore the default and `damping="shift"` reproduces the
original; the two are one flag apart and the difference is worth measuring
rather than arguing.

What is given up
----------------

The trust region. `alpha = quad / curv` is derived from `X = F^-1 G` holding
exactly, which is what makes it dimensionless; with any other preconditioner
the ratio carries units and is not a multiplier at all (`powered.py` established
this for the powered family). It is switched off here explicitly rather than
left silently meaningless, and `eta` is the only thing setting the step length.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .direction import generators
from .linalg import cayley, exact_fp32, floor_eigenvalues, safe_eigh, skew, spectral_norm

__all__ = ["ShampooPion", "inverse_root", "gram"]


def gram(G: torch.Tensor) -> torch.Tensor:
    """`G G^T`, which for skew `G` is also `G^T G` and `-G^2`.

    Written as the Gram product rather than as `-G @ G` so the result is
    symmetric PSD under rounding as well as in exact arithmetic; the identity
    between the three forms is pinned in `test_shampoo.py`.
    """
    return G @ G.transpose(-1, -2)


def inverse_root(
    P: torch.Tensor, power: float, eps: float, damping: str
) -> torch.Tensor:
    """`P^-power` for symmetric PSD `P`, damped one of two ways.

    `damping="floor"` raises the spectrum to `eps * lam_max` -- homogeneous, so
    `inverse_root(c^2 P) = c^-2power inverse_root(P)` exactly, which is the
    scale invariance the whole construction rests on.

    `damping="shift"` adds `eps` to every eigenvalue, as the original Shampoo
    adds `eps I` to its accumulators. Faithful, and not homogeneous.
    """
    w, U = safe_eigh(P)
    if damping == "floor":
        w = floor_eigenvalues(w, eps)
    elif damping == "shift":
        w = w.clamp_min(0.0) + eps
    else:
        raise ValueError(f"damping must be 'floor' or 'shift', got {damping!r}")
    return (U * w.pow(-power).unsqueeze(-2)) @ U.transpose(-1, -2)


class ShampooPion(torch.optim.Optimizer):
    """Pion's rotation, driven by Shampoo's preconditioner on `so(n)`.

    Takes no activations and needs no hooks: unlike every other optimizer in
    this package it is a pure function of the gradients it has been shown.

    Args:
        params: 2-D weight tensors, as `NGDPion`.
        lr: `eta`, and here it is the *only* thing setting step length -- the
            trust region does not survive the change of preconditioner. It has
            its own optimum, unrelated to any `eta` measured for the Fisher
            variants, and must be swept rather than carried over.
        power: exponent per side. `0.25` is the original, so the two sides
            compose to the `-1/2` of Adagrad. `0` disables preconditioning and
            recovers ablated Pion, which makes it the natural control.
        beta: `0` accumulates a plain sum, as the original does; the resulting
            implicit `t^-1/2` decay then compounds with the cosine schedule,
            which is a real interaction on a 73242-step run and not an
            artefact. A positive value makes it an EMA instead.
        eps: damping, read according to `damping`.
        damping: `"floor"` (relative, homogeneous, this package's standard) or
            `"shift"` (absolute, the original's `eps I`). See the module
            docstring: the choice decides whether scale invariance survives.
        t_fac: recompute the inverse roots every `t_fac` steps. The
            accumulators are updated every step regardless; only the
            eigendecomposition is amortised, which is what distributed Shampoo
            does for the same reason.
        plane_every: record the plane-angle spread every `plane_every` steps,
            `0` to disable, which is the default. It is the direct test of the
            orthogonalisation property and the reason is also why it is opt in:
            it costs an `svdvals` per side per layer, and on the real model 24
            of the 56 weights carry a 1376-dimensional side. `angle` is
            recorded every step regardless -- that one is a warm power
            iteration and effectively free.
        alternate: update one side per step, as Pion does, instead of both.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        power: float = 0.25,
        beta: float = 0.0,
        eps: float = 1e-4,
        damping: str = "floor",
        t_fac: int = 25,
        plane_every: int = 0,
        compute_dtype: torch.dtype = torch.float32,
        alternate: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if power < 0.0:
            raise ValueError(f"power must be non-negative, got {power}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if damping not in ("floor", "shift"):
            raise ValueError(f"damping must be 'floor' or 'shift', got {damping!r}")
        if t_fac < 1:
            raise ValueError(f"t_fac must be at least 1, got {t_fac}")
        params = list(params)
        for p in params:
            if p.dim() != 2:
                raise ValueError(
                    f"ShampooPion takes 2-D weights, got shape {tuple(p.shape)}; "
                    "give embeddings, norms and biases to another optimizer"
                )
        super().__init__(
            params,
            dict(
                lr=lr,
                power=power,
                beta=beta,
                eps=eps,
                damping=damping,
                t_fac=t_fac,
                plane_every=plane_every,
                compute_dtype=compute_dtype,
                alternate=alternate,
            ),
        )

    def load_state_dict(self, state_dict) -> None:
        """Reload, then put the accumulators and roots back on the parameter's device.

        Same failure as `NGDPion.load_state_dict` guards against: a resumed run
        maps to CPU, and torch relocates only what it recognises. These tensors
        sit in a plain dict and are not relocated, so the first matmul of the
        resumed step would fail on mismatched devices -- and no CPU test can
        see it.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for key in ("P_in", "P_out", "Q_in", "Q_out", "v_in", "v_out"):
                    t = state.get(key)
                    if isinstance(t, torch.Tensor):
                        state[key] = t.to(p.device)

    # --- the accumulators ---------------------------------------------------

    def _state_of(self, p: torch.Tensor, group: dict) -> dict:
        state = self.state[p]
        if "step" not in state:
            dt = group["compute_dtype"]
            m, n = p.shape
            state["step"] = 0
            state["since_refactor"] = 0
            state["P_in"] = torch.zeros(n, n, dtype=dt, device=p.device)
            state["P_out"] = torch.zeros(m, m, dtype=dt, device=p.device)
            state["v_in"] = None
            state["v_out"] = None
        return state

    @staticmethod
    def _accumulate(P: torch.Tensor, G: torch.Tensor, beta: float) -> None:
        L = gram(G)
        if beta == 0.0:
            P.add_(L)
        else:
            P.mul_(beta).add_(L, alpha=1.0 - beta)

    def _rebuild(self, params: list[torch.Tensor], group: dict) -> None:
        """Recompute the inverse roots, fusing equal shapes into one `eigh`."""
        power, eps, damping = group["power"], group["eps"], group["damping"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        for members in by_shape.values():
            for side in ("in", "out"):
                P = torch.stack([self.state[p][f"P_{side}"] for p in members])
                Q = (
                    inverse_root(P, power, eps, damping)
                    if power > 0.0
                    else torch.eye(P.shape[-1], dtype=P.dtype, device=P.device).expand_as(P)
                )
                for i, p in enumerate(members):
                    self.state[p][f"Q_{side}"] = Q[i].clone()
            for p in members:
                self.state[p]["since_refactor"] = 0

    # --- the step -----------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # The retraction's exactness is the reason this method preserves the
        # spectrum at all, and TF32 destroys it. See `linalg.exact_fp32`.
        with exact_fp32():
            self._step()
        return loss

    def _step(self) -> None:
        for group in self.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                state = self._state_of(p, group)
                dt = group["compute_dtype"]
                W = p.detach().to(dt)
                G = p.grad.detach().to(dt)
                G_in, G_out = generators(W, G)
                self._accumulate(state["P_in"], G_in, group["beta"])
                self._accumulate(state["P_out"], G_out, group["beta"])
                state["_G"] = (G_in, G_out)

            # The roots are rebuilt *after* accumulation and before the step, so
            # the first step already preconditions with the gradient it is about
            # to take -- Adagrad's convention, and the one that makes the
            # single-gradient orthogonalisation above exact.
            due = [
                p for p in active
                if "Q_in" not in self.state[p]
                or self.state[p]["since_refactor"] >= group["t_fac"]
            ]
            if due:
                self._rebuild(due, group)

            for p in active:
                self._apply(p, group)

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G_in, G_out = state.pop("_G")
        Q_in, Q_out = state["Q_in"], state["Q_out"]

        # `skew` is hygiene, not a projection: the sandwich is skew by algebra
        # and rounding does not know that.
        X_in = skew(Q_in @ G_in @ Q_in)
        X_out = skew(Q_out @ G_out @ Q_out)

        c = group["lr"]
        sig_in, state["v_in"] = spectral_norm(X_in, 20 if state["v_in"] is None else 2, state["v_in"])
        sig_out, state["v_out"] = spectral_norm(X_out, 20 if state["v_out"] is None else 2, state["v_out"])
        state["angle"] = c * float(torch.maximum(sig_in, sig_out))

        plane_every = group["plane_every"]
        if plane_every and state["step"] % plane_every == 0:
            self._diagnose(state, X_in, X_out)

        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1

    @staticmethod
    def _diagnose(state: dict, X_in: torch.Tensor, X_out: torch.Tensor) -> None:
        """The plane-angle spread, which is the direct test of the mechanism.

        A skew matrix's singular values come in equal pairs, one pair per
        rotation plane. Under the orthogonalisation of consequence 3 they are
        all equal, so `plane_ratio` is 1; under the raw gradient it is the
        condition number of the generator. It reads out how far the
        preconditioner has actually got, per layer, and it is the quantity the
        4658x-36496x cross-layer spread was a symptom of.
        """
        for side, X in (("in", X_in), ("out", X_out)):
            s = torch.linalg.svdvals(X.float())
            s = s[s > 0]
            state[f"plane_ratio_{side}"] = (
                float(s.max() / s.min()) if s.numel() else float("nan")
            )
            state[f"plane_max_{side}"] = float(s.max()) if s.numel() else float("nan")
