"""ALGORITHM.md §7 -- the optimizer, which only orchestrates.

Every piece of mathematics lives in `covariance`, `factorization`, `direction`
and `linalg`; this module decides *when* each runs and applies the retraction.

It takes parameters, not modules. The input covariance has to be supplied from
outside -- by `observe` during the forward pass, or by `set_covariance`
directly -- because the layer types differ across frameworks: `nn.Linear`,
Megatron's `ColumnParallelLinear`, a fused QKV projection. `hooks.attach`
covers the `nn.Linear` case; anything else writes its own five-line adapter
without this module needing to know.

Precision: the step runs in `compute_dtype`, fp32 by default. fp64 buys
nothing here -- measured end-to-end error in fp32 is 1e-5 to 1e-3 on realistic
spectra -- and costs a factor of 30 to 60 on consumer GPUs, where it is also
simply unavailable on some backends.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .covariance import CovarianceAccumulator
from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .factorization import build_bases
from .linalg import cayley

__all__ = ["NGDPion"]


class NGDPion(torch.optim.Optimizer):
    """Curvature-preconditioned Pion.

    Args:
        params: 2-D weight tensors. Everything else -- embeddings, the output
            head, norm gains, biases -- belongs to another optimizer, which is
            how Pion splits parameters too.
        lr: `eta`. The one hyperparameter that genuinely needs tuning.
        beta: EMA rate for the input covariance.
        eps: spectral floor relative to `lam_max`. Its lower bound is set by
            `compute_dtype`, not by the problem: below the working machine
            epsilon the floor is meaningless and the small end of the pencil
            rounds to noise. Measured on a wide layer with `cond(A) = 1e4`,
            fp32 against fp64 gives `2e-1` error at `eps = 1e-8`, `5e-3` at
            `1e-6` and `8e-5` at `1e-4`. So `1e-4` for fp32 -- which also sits
            at the held-out optimum -- and anything down to `1e-10` is safe in
            fp64. `0` is valid only when every `W` is square and every `A`
            full rank; otherwise the dead block is a true `0/0`.
        alpha_max: cap on the trust-region ratio. `1` keeps it one-sided, so a
            stale basis can only shorten the step.
        t_fac: refactor every `t_fac` steps. Fixed on purpose -- a data
            dependent schedule makes the run irreproducible across hardware.
        compute_dtype: precision of the factorisation and the step.
        alternate: update one side per step, as Pion does, instead of both.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.95,
        eps: float = 1e-4,
        alpha_max: float = 1.0,
        t_fac: int = 100,
        compute_dtype: torch.dtype = torch.float32,
        alternate: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if t_fac < 1:
            raise ValueError(f"t_fac must be at least 1, got {t_fac}")
        params = list(params)
        for p in params:
            if p.dim() != 2:
                raise ValueError(
                    f"NGDPion takes 2-D weights, got shape {tuple(p.shape)}; "
                    "give embeddings, norms and biases to another optimizer"
                )
        super().__init__(
            params,
            dict(
                lr=lr,
                beta=beta,
                eps=eps,
                alpha_max=alpha_max,
                t_fac=t_fac,
                compute_dtype=compute_dtype,
                alternate=alternate,
            ),
        )

    # --- §3: statistics, supplied from outside ------------------------------

    def _accumulator(self, param: torch.Tensor) -> CovarianceAccumulator:
        state = self.state[param]
        if "cov" not in state:
            group = self._group_of(param)
            state["cov"] = CovarianceAccumulator(
                beta=group["beta"],
                dtype=torch.float64 if group["compute_dtype"] is torch.float64 else torch.float32,
            )
            state["step"] = 0
            state["since_refactor"] = 0
        return state["cov"]

    def _group_of(self, param: torch.Tensor) -> dict:
        for group in self.param_groups:
            for p in group["params"]:
                if p is param:
                    return group
        raise KeyError("parameter is not held by this optimizer")

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        """Record the activations entering the layer that owns `param`."""
        self._accumulator(param).observe(x)

    def set_covariance(self, param: torch.Tensor, A: torch.Tensor) -> None:
        """Install `A` directly, bypassing accumulation. For tests and adapters."""
        acc = self._accumulator(param)
        acc.load_state_dict({"matrix": A.to(acc.dtype), "count": acc.count, "beta": acc.beta})
        self.state[param].pop("bases", None)

    # --- §4: refactorisation, batched over equal shapes ---------------------

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        """Rebuild bases, fusing layers of equal shape into one `eigh` call."""
        dt = group["compute_dtype"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) for p in members])
            A = torch.stack([self.state[p]["cov"].matrix.to(dt) for p in members])
            basis_in, basis_out = build_bases(W, A, group["eps"])
            for i, p in enumerate(members):
                self.state[p]["bases"] = (
                    type(basis_in)(basis_in.P[i], basis_in.lam[i], basis_in.orthogonal),
                    type(basis_out)(basis_out.P[i], basis_out.lam[i], basis_out.orthogonal),
                )
                self.state[p]["since_refactor"] = 0

    # --- §§5-7: the step ----------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if not self._accumulator(p).ready:
                    raise RuntimeError(
                        "no activations recorded for a parameter -- call observe() "
                        "during the forward pass, or set_covariance() directly"
                    )
            stale = [p for p in active if "bases" not in self.state[p]]
            if stale:
                self._refactor(stale, group)
            for p in active:
                self._apply(p, group)

            due = [p for p in active if self.state[p]["since_refactor"] >= group["t_fac"]]
            if due:
                self._refactor(due, group)
        return loss

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]

        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(dt)
        eye_out = torch.eye(W.shape[0], dtype=dt, device=W.device)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, W.T @ W, X_in)).sum() + (
            X_out * fisher_apply(eye_out, W @ A @ W.T, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)

        c = group["lr"] * float(alpha)
        # recorded for diagnostics: dropping Pion's RMS scaling removed what
        # pinned the rotation angle, so whether it stays bounded is measured
        # rather than assumed
        state["angle"] = c * float(
            torch.maximum(
                torch.linalg.matrix_norm(X_in, 2), torch.linalg.matrix_norm(X_out, 2)
            )
        )
        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1
