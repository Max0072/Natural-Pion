"""NGD-Pion with `S` measured instead of assumed.

`optimizer.py` takes `S = E[delta delta^T]` to be the identity. That is not a
mild simplification: it distributes the step across layers with the **wrong
sign**.

    with the true S = W^T E[dd^T] W :   ||X|| ~ 1 / (||W|| ||delta|| ||x||)
    with S = I  (so S = W^T W)      :   ||X|| ~ ||delta|| / (||W|| ||x||)

Curvature goes as `delta^2`, so the natural gradient's step should be *larger*
where the backward signal is weaker. Under `S = I` it is larger where the
signal is stronger, and the two differ by `||delta||^2` per layer.

Measured on LLaMA-60M at `eta = 1.0`: the per-layer rotation angle correlates
with `||delta||` at 0.92 to 0.98 and spans three orders of magnitude within a
single step, the output projections `attn.wo` and `ffn.down` taking rotations
of a radian while `attn.wq` and `attn.wk` take a thousandth of one. Against
Pion that showed up as a flat 0.28 deficit in loss from step 500 to step 25 000
-- the shape of a systematically misallocated step rather than of an
accumulating one.

**Two accumulators, two rates.** The backward signal depends on the current
loss surface and on everything downstream of the layer, so it moves faster than
the activation statistics do. `beta_backward` is separate from `beta` for that
reason and should be shorter.

**What it costs.** An `E[delta delta^T]` accumulation is another
`tokens x m^2` matmul per step, comparable to the activation one; and the
out-side pair becomes `(D, W A W^T)` with neither factor the identity, so it
leaves `basis_identity_anchor` for the congruence path -- an extra `eigh` and
an inverse square root per refactorisation.

**`eta` does not carry over.** `F` now scales with `||D||`, which is `1e-18` to
`1e-12` on this model, so `X` scales inversely and the useful range of `eta`
moves by many orders. It has to be swept again from scratch.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .covariance import CovarianceAccumulator
from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .factorization import basis_congruence, basis_identity_anchor
from .linalg import cayley, is_identity
from .optimizer import NGDPion

__all__ = ["NGDPionS", "BackwardRecorder", "attach_backward"]


class NGDPionS(NGDPion):
    """`NGDPion` with the backward covariance estimated rather than assumed.

    Args:
        beta_backward: EMA rate for `D = E[delta delta^T]`. Shorter than
            `beta` on purpose; see the module docstring.

    Everything else is inherited. `observe_backward` has to be fed during the
    backward pass, the way `observe` is fed during the forward -- use
    `attach_backward`, or write the five-line adapter a different framework
    needs.
    """

    def __init__(self, params, *, beta_backward: float = 0.5, **kwargs) -> None:
        if not 0.0 <= beta_backward < 1.0:
            raise ValueError(f"beta_backward must lie in [0, 1), got {beta_backward}")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["beta_backward"] = beta_backward

    # --- the second statistic ------------------------------------------------

    def _backward_accumulator(self, param: torch.Tensor) -> CovarianceAccumulator:
        state = self.state[param]
        if "cov_backward" not in state:
            group = self._group_of(param)
            state["cov_backward"] = CovarianceAccumulator(
                beta=group["beta_backward"],
                dtype=torch.float64 if group["compute_dtype"] is torch.float64 else torch.float32,
            )
        return state["cov_backward"]

    def observe_backward(self, param: torch.Tensor, delta: torch.Tensor) -> None:
        """Record the gradient arriving at the output of the layer owning `param`."""
        self._backward_accumulator(param).observe(delta)

    def _ready(self, param: torch.Tensor) -> bool:
        state = self.state[param]
        return (
            "cov_backward" in state
            and state["cov_backward"].ready
            and self._accumulator(param).ready
        )

    # --- §4 with both factors measured --------------------------------------

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        """As the reference, but the pairs are `(A, W^T D W)` and `(D, W A W^T)`."""
        dt = group["compute_dtype"]
        by_shape: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            by_shape[tuple(p.shape)].append(p)

        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) for p in members])
            A = torch.stack(
                [self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p in members]
            )
            D = torch.stack(
                [
                    self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt)
                    for p in members
                ]
            )
            Wt = W.transpose(-1, -2)
            gram_in = Wt @ D @ W
            gram_out = W @ A @ Wt
            eps = group["eps"]

            basis_in = (
                basis_identity_anchor(A, eps)
                if is_identity(gram_in)
                else basis_congruence(A, gram_in, eps)
            )
            basis_out = (
                basis_identity_anchor(gram_out, eps)
                if is_identity(D)
                else basis_congruence(D, gram_out, eps)
            )
            for i, p in enumerate(members):
                self.state[p]["bases"] = (
                    type(basis_in)(basis_in.P[i], basis_in.lam[i], basis_in.orthogonal),
                    type(basis_out)(basis_out.P[i], basis_out.lam[i], basis_out.orthogonal),
                )
                self.state[p]["since_refactor"] = 0

    def _step(self, closure_loss=None):
        for group in self.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if not self._ready(p):
                    raise RuntimeError(
                        "this variant needs both statistics -- call observe() during "
                        "the forward pass and observe_backward() during the backward, "
                        "or use attach_backward()"
                    )
            stale = [p for p in active if "bases" not in self.state[p]]
            if stale:
                self._refactor(stale, group)
            for p in active:
                self._apply(p, group)
            due = [p for p in active if self.state[p]["since_refactor"] >= group["t_fac"]]
            if due:
                self._refactor(due, group)

    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]

        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        D = state["cov_backward"].matrix.to(device=p.device, dtype=dt)
        Wt = W.transpose(-1, -2)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, Wt @ D @ W, X_in)).sum() + (
            X_out * fisher_apply(D, W @ A @ Wt, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)

        c = group["lr"] * float(alpha)
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


class BackwardRecorder:
    """Feeds each layer's output gradient to `observe_backward`.

    A forward hook that attaches a hook to the output tensor, rather than
    `register_full_backward_hook`: the latter wraps the module in autograd
    functions and materialises `grad_input`, which is not needed here.

    `enabled` mirrors `ActivationRecorder`'s: the EMA is defined per optimizer
    step, so under gradient accumulation only one micro-batch may feed it.
    """

    def __init__(self, modules, optimizer: NGDPionS) -> None:
        self.enabled = True
        self._handles = [
            m.register_forward_hook(self._make(m, optimizer)) for m in list(modules)
        ]

    def _make(self, module, optimizer):
        weight = module.weight

        def forward_hook(mod, inputs, output):
            if not self.enabled or not torch.is_tensor(output) or not output.requires_grad:
                return None

            def grad_hook(grad):
                optimizer.observe_backward(weight, grad.detach())
                return None

            output.register_hook(grad_hook)
            return None

        return forward_hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __len__(self) -> int:
        return len(self._handles)


def attach_backward(modules, optimizer: NGDPionS) -> BackwardRecorder:
    """Start recording output gradients for `modules` into `optimizer`."""
    return BackwardRecorder(modules, optimizer)
