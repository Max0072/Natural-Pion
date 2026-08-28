"""One step, with the variants as flags instead of as a tower of subclasses.

Why
---

`_apply` had been copied three times -- `optimizer.py`, `with_s.py`,
`with_s_fast.py` -- and `momentum.py` made a fourth. Copies drift, so each one
needed a test whose only job was to police the copy. Worse, the tower is
*linear*: momentum exists only in combination with the measured `S`, because it
was written as a subclass of the `S` variant, and there is no way to ask for
momentum without `S` or for any combination nobody happened to subclass.

The variants are not a hierarchy. They are independent choices on one pipeline
-- statistics, basis, solve, retraction -- so they belong as flags.

Two kinds of flag, kept apart on purpose
----------------------------------------

**Algorithmic flags change the trajectory** and each one has to reproduce the
class it replaces *bit for bit*: `use_s`, `momentum`, `beta1`.

**Implementation flags should not change it**, or should change it by a stated
and bounded amount: `retraction`, `angle`. These are tested differently --
`angle` feeds nothing in the step and must leave the weights untouched
whichever way it is computed, while `retraction="ns"` is an approximation whose
error is a function of the rotation angle and is bounded rather than zero.

Mixing the two kinds in one list is how a knob that was supposed to be free
quietly becomes a variable in a comparison. `tests/test_unified.py` separates
them.

What is deliberately not folded in
----------------------------------

`powered.py` (`F^-p`) and `damped.py` (additive Tikhonov with a
Levenberg-Marquardt rule) stay where they are. Both are dead ends -- with the
caveat recorded in `ngd_pion/README.md` that their verdicts were reached at 150
steps, a horizon since measured to invert -- and folding them in would multiply
the flag surface for no present benefit. `shampoo.py` and `pion_baseline.py`
stay separate because they are different algorithms, not settings of this one:
Shampoo builds its preconditioner from the generators and has no covariance, no
Fisher and no basis at all.

`reference.py` and the `-ref` classes stay untouched. They are the oracle, and
`AGENTS.md` requires that the reference is never optimised in place.

The reduction that makes this exact
-----------------------------------

With `D = I` the `S` path *is* the `S = I` path: the pairs `(A, W^T D W)` and
`(D, W A W^T)` become `(A, W^T W)` and `(I, W A W^T)`, and `is_identity(D)`
sends the out-side down the identity anchor, which is what `build_bases` does
unconditionally. So `use_s` is a genuine branch and not an approximation -- but
it *is* written as a branch rather than by passing an identity matrix, because
`W^T I W` and `W^T W` are only equal to whatever BLAS guarantees, and this file
claims bit-identity.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .exact_curv import exact_curv
from .factorization import basis_congruence, basis_identity_anchor
from .linalg import cayley, cayley_newton_schulz, is_identity, spectral_norm
from .with_s import NGDPionS

__all__ = ["NGDPionUnified"]


class NGDPionUnified(NGDPionS):
    """NGD-Pion with the variant choices exposed as flags.

    Args:
        use_s: `True` measures `S` from the backward signal (`ngd-pion-s`),
            `False` fixes `S = I` (`ngd-pion`). When `False` no backward
            statistic is required and none is asked for.
        momentum: `"none"`, `"lie"` (buffers on the two generators, the variant
            Pion's published figures use) or `"ambient"` (the raw gradient
            smoothed before the generators are taken).
        beta1: the momentum EMA factor. No bias correction, as in Pion.
        retraction: `"cayley"` is the exact solve; `"ns"` is Newton-Schulz,
            measured at 0.18x of Cayley's cost on the real shapes, which is 54%
            of the step. **It is an approximation and its error grows with the
            angle**: at two iterations, `8.6e-6` at 0.5 rad but `1.2e-3` at 1.0,
            and measured angles on this model reach 1.3 to 3.5. `ns_guard`
            falls back to the exact solve per layer wherever the angle is too
            large, using the angle this step already computes.
        ns_iters: Newton-Schulz iterations where it is used.
        ns_guard: rotation angle above which a layer takes the exact solve
            regardless. `0` disables the fallback, which is how to measure what
            the guard is worth rather than assume it.
        trust: which ratio sets `alpha`.

            `"quad_curv"` is the historical one, `quad / curv`. **It is not a
            trust region.** Both halves are formed from `X = F^-1 G` with the
            same operator, so on a fresh basis it is 1 by algebra and it can
            only ever detect that the basis has gone stale. **It does detect it**:
            measured over 24304 layer-steps of the full-length run, `alpha` is
            below 1 on **37.1%** of them and below 0.9 on 13.6%, with a minimum
            of 0.0262. So it is not inert -- it is active, and what it is active
            about is staleness rather than the model being wrong.

            `"exact"` is `quad / curv_exact`, with `curv_exact = 4 E_b[s_b^2]`
            measured against per-token quantities rather than against the
            operator that built the step -- what K-FAC does, and what makes the
            ratio capable of noticing the model is wrong. Measured
            `alpha_exact/alpha` on this model is 4.7e-3 at step 1 and 0.04-0.12
            in steady state, so it shortens by 8 to 25x, **per layer and per
            step**, which is the pair of freedoms a swept `eta` does not have.

            `"none"` fixes `alpha = alpha_max`.

            **The risk to check first**: `curv_exact` is an estimate from
            `exact_tokens` tokens, so `alpha` becomes a noisy per-step
            multiplier. A trust region that jitters is a variance source, not a
            control. If it hurts, smoothing `curv_exact` is the next thing to
            try, not abandoning the ratio.
        exact_beta: EMA factor on `curv_exact`. `0` uses the instantaneous
            estimate, which is what the first attempt did and why it failed.

            Measured on job 297594: the median `alpha` across layers spans
            **8.4x**, and within a single layer across steps it spans
            **109.5x**. The per-layer differences are real -- 8.4x agrees with
            the 2-5x between attention and FFN in the journal -- but they sit
            under thirteen times as much estimation noise, and a multiplier
            that jitters by 100x is a variance source rather than a control.

            **More tokens cannot fix it.** The noise falls as `1/sqrt(N)`, the
            subsample is 4096 and the whole batch is 131072, so even spending
            the entire batch buys `sqrt(32) = 5.7x` against the 13x needed.
            Smoothing in time is the only route, which is unsurprising: `A` and
            `D` are EMAs of exactly the same kind and only this one was taken
            instantaneously.

            **The average is geometric.** `curv_exact` is heavy-tailed --
            mean/median 10.1 across layers, 6362 at worst, single draws at 187
            medians -- so an arithmetic EMA converges to a mean far above the
            typical value and `alpha` collapses. Measured at `beta = 0.99` with
            an arithmetic mean: median `alpha` reached 0.0000 by step 101 and
            the rotation stopped. Smoothing `log(curv_exact)` estimates the
            geometric mean, which for this tail sits at the median.

            The buffer is seeded with its first observation rather than with
            zero, which in the log is the same as starting the geometric mean
            at the first draw.
        exact_tokens: tokens subsampled per layer for `curv_exact`. Paired
            between the forward and the backward hook by index, drawn from a
            generator of this class's own so that turning the flag on does not
            move the global RNG.
        angle: `"power"` is the warm power iteration, `"svd"` the exact
            `matrix_norm`, `"off"` skips it. The angle is read by
            `harness.instrument` and by the Newton-Schulz guard, and by nothing
            in the step itself, so `"off"` is safe only with
            `retraction="cayley"` and is rejected otherwise.
    """

    def __init__(
        self,
        params,
        *,
        use_s: bool = True,
        momentum: str = "none",
        beta1: float = 0.9,
        trust_lr: bool = False,
        trust_lr_cap: float = 100.0,
        trust: str = "quad_curv",
        exact_beta: float = 0.0,
        exact_tokens: int = 4096,
        retraction: str = "cayley",
        ns_iters: int = 2,
        ns_guard: float = 0.5,
        angle: str = "power",
        **kwargs,
    ) -> None:
        if momentum not in ("none", "lie", "ambient"):
            raise ValueError(f"momentum must be none/lie/ambient, got {momentum!r}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if trust_lr_cap < 1.0:
            raise ValueError(f"trust_lr_cap must be at least 1, got {trust_lr_cap}")
        if trust not in ("quad_curv", "exact", "none"):
            raise ValueError(f"trust must be quad_curv/exact/none, got {trust!r}")
        if trust == "exact" and not use_s:
            raise ValueError(
                "trust='exact' needs the per-token output gradients, which only "
                "the measured-S path collects; set use_s=True"
            )
        if not 0.0 <= exact_beta < 1.0:
            raise ValueError(f"exact_beta must be in [0, 1), got {exact_beta}")
        if exact_tokens < 1:
            raise ValueError(f"exact_tokens must be at least 1, got {exact_tokens}")
        if retraction not in ("cayley", "ns"):
            raise ValueError(f"retraction must be cayley/ns, got {retraction!r}")
        if angle not in ("power", "svd", "off"):
            raise ValueError(f"angle must be power/svd/off, got {angle!r}")
        if angle == "off" and retraction == "ns" and ns_guard > 0:
            raise ValueError(
                "the Newton-Schulz guard reads the angle, so angle='off' would "
                "disable the guard silently; set ns_guard=0 to mean it"
            )
        if ns_iters < 1:
            raise ValueError(f"ns_iters must be at least 1, got {ns_iters}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group.update(
                use_s=use_s, momentum=momentum, beta1=beta1,
                retraction=retraction, ns_iters=ns_iters, ns_guard=ns_guard,
                angle=angle, trust=trust, exact_tokens=exact_tokens,
                exact_beta=exact_beta, trust_lr=trust_lr,
                trust_lr_cap=trust_lr_cap, lr_scale=1.0,
            )

    # --- statistics ---------------------------------------------------------

    def _ready(self, param: torch.Tensor) -> bool:
        """`S = I` needs no backward statistic, so it must not be demanded."""
        if not self._group_of(param)["use_s"]:
            state = self.state.get(param, {})
            return "cov" in state and state["cov"].ready
        return super()._ready(param)

    def adapt_damping(self, rho: float) -> None:
        """The classical trust-region radius update, on `eta`.

        The harness calls this hook `adapt_damping` because `damped.py` was its
        only implementer and damping was that variant's radius. **What this
        class adapts is the step scale**, which is this one's radius: `eta`
        multiplies the step directly, and the floor stays where it is.

        Why `eta` and not the floor. `damped.py` replaced the spectral floor
        with Tikhonov damping and adapted that, and job 297755 measured the
        result: the adaptation is worth 0.39, and the substitution costs 0.72.
        The rule was never the problem; the thing it was bolted to was.

        Why adapt at all. Measured `rho` on the full-length run runs **0.47 at
        the start and 1.32 by step 33000**, crossing 1 around 15000: the step is
        too long early and too short late, in that order. One scalar cannot be
        both, and a sweep can only pick the compromise.

        Thresholds and factor are K-FAC's, and the cadence comes from
        `rho_every`, which kfac-jax sets to 5.
        """
        if rho is None or rho != rho or rho in (float("inf"), float("-inf")):
            return
        for group in self.param_groups:
            if not group.get("trust_lr"):
                continue
            cap = group["trust_lr_cap"]
            if rho > 0.75:
                group["lr_scale"] = min(group["lr_scale"] * 1.5, cap)
            elif rho < 0.25:
                group["lr_scale"] = max(group["lr_scale"] / 1.5, 1.0 / cap)

    def _sampler(self, device: torch.device) -> torch.Generator:
        """One generator per device, seeded once, never the global RNG."""
        cache = self.__dict__.setdefault("_samplers", {})
        key = (device.type, device.index)
        if key not in cache:
            g = torch.Generator(device=device)
            g.manual_seed(0x5EED)
            cache[key] = g
        return cache[key]

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        super().observe(param, x)
        if self._group_of(param)["trust"] != "exact":
            return
        state = self.state[param]
        flat = x.detach().reshape(-1, x.shape[-1])
        k = min(self._group_of(param)["exact_tokens"], flat.shape[0])
        # Drawn here and kept for the backward hook: the two samples have to be
        # the same tokens, or the estimate reads as a spectacular independence
        # failure rather than as a curvature.
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
            state.pop("exact_idx", None)
            return
        # `* n` undoes the `1/N` autograd applies because the loss is a mean.
        state["exact_d"] = flat[idx].to(torch.float32) * float(flat.shape[0])
        state.pop("exact_idx", None)

    def _D(self, p: torch.Tensor, group: dict) -> torch.Tensor | None:
        if not group["use_s"]:
            return None
        return self.state[p]["cov_backward"].matrix.to(
            device=p.device, dtype=group["compute_dtype"]
        )

    # --- the basis ----------------------------------------------------------

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        dt, eps = group["compute_dtype"], group["eps"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) for p in members])
            A = torch.stack(
                [self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p in members]
            )
            Wt = W.transpose(-1, -2)
            gram_out = W @ A @ Wt

            if group["use_s"]:
                D = torch.stack([self._D(p, group) for p in members])
                gram_in = Wt @ D @ W
                basis_out = (
                    basis_identity_anchor(gram_out, eps)
                    if is_identity(D)
                    else basis_congruence(D, gram_out, eps)
                )
            else:
                gram_in = Wt @ W
                basis_out = basis_identity_anchor(gram_out, eps)

            basis_in = (
                basis_identity_anchor(A, eps)
                if is_identity(gram_in)
                else basis_congruence(A, gram_in, eps)
            )
            for i, p in enumerate(members):
                self.state[p]["bases"] = (
                    type(basis_in)(basis_in.P[i], basis_in.lam[i], basis_in.orthogonal),
                    type(basis_out)(basis_out.P[i], basis_out.lam[i], basis_out.orthogonal),
                )
                self.state[p]["since_refactor"] = 0

    # --- the step -----------------------------------------------------------

    @staticmethod
    def _ema(state: dict, key: str, g: torch.Tensor, beta: float) -> torch.Tensor:
        if key not in state:
            state[key] = torch.zeros_like(g)
        state[key].mul_(beta).add_(g, alpha=1.0 - beta)
        return state[key]

    def _measure_angle(self, state, group, X_in, X_out, c):
        """`(angle, sigma_max)`; `sigma_max` is what the retraction guard reads."""
        mode = group["angle"]
        if mode == "off":
            return float("nan"), None
        if mode == "svd":
            s_in = torch.linalg.matrix_norm(X_in, 2)
            s_out = torch.linalg.matrix_norm(X_out, 2)
        else:
            fresh = "angle_v_in" not in state or state["since_refactor"] == 0
            iters = group["angle_warmup"] if fresh else group["angle_iters"]
            s_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
            s_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        sigma = torch.maximum(s_in, s_out)
        return c * sigma, sigma

    def _retract(self, X: torch.Tensor, c: float, group: dict, sigma) -> torch.Tensor:
        """Cayley, or Newton-Schulz where the angle is small enough for it."""
        if group["retraction"] == "cayley":
            return cayley(X, c)
        guard = group["ns_guard"]
        if guard > 0 and sigma is not None and float(c * sigma) > guard:
            return cayley(X, c)
        return cayley_newton_schulz(X, c, group["ns_iters"])

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
        Wt = W.transpose(-1, -2)
        if group["use_s"]:
            D = self._D(p, group)
            B_in, C_in, B_out = A, Wt @ D @ W, D
        else:
            B_in, C_in = A, Wt @ W
            B_out = torch.eye(W.shape[0], dtype=dt, device=W.device)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(B_in, C_in, X_in)).sum() + (
            X_out * fisher_apply(B_out, W @ A @ Wt, X_out)
        ).sum()
        state["quad"] = quad
        state["curv"] = curv
        mode = group["trust"]
        if mode == "none":
            alpha = torch.as_tensor(group["alpha_max"], dtype=quad.dtype, device=quad.device)
        elif mode == "exact":
            xs = state.pop("exact_x", None)
            ds = state.pop("exact_d", None)
            if xs is None or ds is None:
                raise RuntimeError(
                    "trust='exact' needs paired per-token samples; observe() and "
                    "observe_backward() must both run every step for it"
                )
            ce = exact_curv(W.float(), X_in.float(), X_out.float(), xs, ds)
            # The raw estimate is kept whatever the smoothing does, so that the
            # noise this flag exists to remove stays measurable.
            state["curv_exact"] = ce
            beta = group["exact_beta"]
            if beta > 0.0:
                # **A geometric mean, not an arithmetic one, and the difference
                # decides whether this works at all.** `curv_exact` is heavily
                # heavy-tailed: measured across 56 layers, mean/median is 10.1
                # typically and 6362 at worst, and a single draw reaches 187
                # medians typically. An EMA estimates the *mean*, so a long one
                # converges to a value ten times the typical and `alpha`
                # collapses -- measured at `beta = 0.99`, the median `alpha`
                # went to 0.0000 from step 101 and the rotation died.
                #
                # Averaging in the log estimates the geometric mean, which for
                # a tail of this shape sits at the median, which is the typical
                # value the ratio wants. `kfac_error.py` already reported its
                # cross-layer figure as a geometric mean, for the same reason.
                lg = torch.log(ce.clamp_min(torch.finfo(ce.dtype).tiny))
                if "curv_exact_log" not in state:
                    state["curv_exact_log"] = lg.clone()
                else:
                    state["curv_exact_log"] = beta * state["curv_exact_log"] + (1.0 - beta) * lg
                ce = torch.exp(state["curv_exact_log"])
                state["curv_exact_smooth"] = ce
            alpha = trust_region_alpha(quad, ce.to(quad.dtype), group["alpha_max"])
            state["alpha_exact"] = float(alpha)
        else:
            alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)

        c = group["lr"] * group.get("lr_scale", 1.0) * float(alpha)
        state["angle"], sigma = self._measure_angle(state, group, X_in, X_out, c)
        state["pred_drop"] = c * quad

        if group["alternate"]:
            W = (W @ self._retract(X_in, c, group, sigma) if state["step"] % 2
                 else self._retract(X_out, c, group, sigma) @ W)
        else:
            W = self._retract(X_out, c, group, sigma) @ W @ self._retract(X_in, c, group, sigma)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1

    def load_state_dict(self, state_dict) -> None:
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
