"""Natural Pion: spectrum-preserving rotations along the natural gradient (docs/paper/FINAL.md).

Two classes, one algorithm:

* `NaturalPion` -- the method in its exact form (FINAL.md section 3, Algorithm 1): the Fisher basis is
  rebuilt every step and the statistics are collected every step.
* `NaturalPionFast` -- the form we run (Algorithm 2): the same direction, length rule and retraction,
  with the basis rebuilt every 25 steps and the statistics collected only in the 6 steps before each
  rebuild. Validated against the exact form in FINAL.md section 6.4.

For each rotated weight `W` (m x n) with gradient `G`, one step is

    G_in  = skew(W^T G),  G_out = skew(G W^T)                 generators (Riemannian gradients)
    M     = beta1 M + (1 - beta1) G                           Lie momentum on each side
    X     = F^-1(M)                                           F(X) = (B X C + C X B) / 2 on each side:
                                                              in (A, W^T D W), out (D, W A W^T)
    sigma = max(||X_in||_2, ||X_out||_2)                      one scale for both sides
    c     = theta_t / sigma                                   the largest plane turns by theta_t
    W    <- Cay(X_out, c) W Cay(X_in, c)                      Cayley: W's singular values are kept

`lr` is `theta_t`, the rotation angle in radians; the caller schedules it (cosine in our runs).
`A = E[x x^T]` (layer input) and `D = E[d d^T]` (gradient at the layer output) are fed by hooks:
`ngd_pion.hooks.attach` for `observe` and `ngd_pion.with_s.attach_backward` for `observe_backward`.

`NaturalPionFast` reproduces `scripts/probes/step_norm.py` with `STEP_NORM_FAST=1 STEP_NORM_BATCH=1
STEP_NORM_A_LAST=6 STEP_NORM_D_LAST=6` bit for bit (tests/test_natural_pion.py). It is written from
the same primitives (`covariance`, `direction`, `factorization`, `linalg`); the reference classes
(`optimizer.py`, `with_s.py`, `unified.py`) are untouched.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .covariance import CovarianceAccumulator
from .direction import generators
from .factorization import basis_congruence, basis_identity_anchor
from .linalg import exact_fp32, is_identity, skew, spectral_norm

__all__ = ["NaturalPion", "NaturalPionFast"]


class NaturalPion(torch.optim.Optimizer):
    """Natural Pion, exact form: fresh Fisher basis and statistics every step.

    Args:
        params: 2-D weights to rotate. Embeddings, head, norms and biases belong to another optimizer.
        lr: rotation angle `theta` in radians for the largest plane of every matrix. Scheduled by the caller.
        beta1: Lie momentum on the generators (as in Pion). 0 turns it off.
        beta_a: EMA factor of `A = E[x x^T]`.
        beta_d: EMA factor of `D = E[d d^T]`.
        t_fac: rebuild the Fisher basis every `t_fac` steps.
        stats_window: collect `A` and `D` only in the last `stats_window` steps before a rebuild;
            `None` collects every step. Between rebuilds nothing reads them.
        eps: spectral floor, relative to the largest eigenvalue, on every spectrum that is inverted.
        angle_iters, angle_warmup: power-iteration steps for `||X||_2`, warm-started from the previous
            step's vector; `angle_warmup` right after a rebuild, where `X` jumps.
        damping: Tikhonov-style damping relative to each matrix's largest Fisher eigenvalue: the denominator
            (lam_i + lam_j)/2 becomes (lam_i + lam_j)/2 + damping * max. In the basis this is F + delta B X B,
            damping in the metric of the anchor factor. 0 (default) leaves only the floor `eps`.
        retraction: "cayley" solves (I + c/2 X) R = (I - c/2 X) exactly; "ns" computes the inverse by
            `ns_iters` Newton-Schulz iterations (matrix products only), which is exact to fp32 round-off for
            angles well below 1 and is used only while the scheduled angle is at most `ns_guard`; "trunc" is
            Pion's retraction, the degree-2 truncated exponential I + A + A^2/2 with A = -c X, which is not
            orthogonal: (T^T T = I + A^4/4), so it lets the spectrum grow.
        row_blocks: `{id(weight): k}` splits that weight's rows into `k` equal blocks, each rotated as a
            separate matrix with its own Fisher (input moment `A` shared, output moment the block's
            diagonal block of `D`). Pion rotates the query projection per attention head this way.
        share_inputs: weights fed the very same input tensor (q, k, v; gate, up) share one accumulator
            for `A` instead of computing the identical gram once each. Bit-identical to not sharing.
        stats_dtype: "fp32" forms the grams of `A` and `D` from fp32 inputs; "bf16" rounds the inputs
            (activations, output gradients) to bf16 and still accumulates and returns the gram in fp32, so
            it stays a PSD gram (see `CovarianceAccumulator.observe`).
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        *,
        beta1: float = 0.9,
        beta_a: float = 0.95,
        beta_d: float = 0.5,
        t_fac: int = 1,
        stats_window: int | None = None,
        eps: float = 1e-4,
        angle_iters: int = 2,
        angle_warmup: int = 50,
        row_blocks: dict[int, int] | None = None,
        damping: float = 0.0,
        retraction: str = "cayley",
        ns_iters: int = 2,
        ns_guard: float = 0.5,
        share_inputs: bool = False,
        stats_dtype: str = "fp32",
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must lie in [0, 1), got {beta1}")
        for name, b in (("beta_a", beta_a), ("beta_d", beta_d)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {b}")
        if t_fac < 1:
            raise ValueError(f"t_fac must be at least 1, got {t_fac}")
        if retraction not in ("cayley", "ns", "trunc"):
            raise ValueError(f"retraction must be cayley, ns or trunc, got {retraction!r}")
        if stats_dtype not in ("fp32", "bf16"):
            raise ValueError(f"stats_dtype must be fp32 or bf16, got {stats_dtype!r}")
        if stats_window is not None and not 1 <= stats_window <= t_fac:
            raise ValueError(f"stats_window must lie in [1, t_fac], got {stats_window}")
        params = list(params)
        for p in params:
            if p.dim() != 2:
                raise ValueError(f"NaturalPion takes 2-D weights, got shape {tuple(p.shape)}")
        self.row_blocks = dict(row_blocks or {})
        for p in params:
            k = self.row_blocks.get(id(p), 1)
            if k < 1 or p.shape[0] % k:
                raise ValueError(f"row_blocks {k} does not divide {p.shape[0]} rows")
        self.share_inputs = share_inputs
        self._input_dtype = torch.bfloat16 if stats_dtype == "bf16" else None
        self._last_input = None                    # (input tensor, its accumulator) for share_inputs
        super().__init__(params, dict(
            lr=lr, beta1=beta1, beta_a=beta_a, beta_d=beta_d, t_fac=t_fac, stats_window=stats_window,
            eps=eps, angle_iters=angle_iters, angle_warmup=angle_warmup, compute_dtype=torch.float32,
            damping=damping, retraction=retraction, ns_iters=ns_iters, ns_guard=ns_guard,
        ))

    # --- statistics, fed by hooks -----------------------------------------------------------------

    def _group_of(self, param: torch.Tensor) -> dict:
        for group in self.param_groups:
            for p in group["params"]:
                if p is param:
                    return group
        raise KeyError("parameter is not held by this optimizer")

    def _collecting(self, param: torch.Tensor) -> bool:
        """Whether this step's statistics are ever read: always before the first basis, then only inside
        the window that precedes a rebuild."""
        group = self._group_of(param)
        window = group["stats_window"]
        state = self.state[param]
        return window is None or "bases" not in state or state["since_refactor"] >= group["t_fac"] - window

    def _stat(self, param: torch.Tensor, key: str, beta: float) -> CovarianceAccumulator:
        state = self.state[param]
        if key not in state:
            state[key] = CovarianceAccumulator(beta=beta, dtype=torch.float32, input_dtype=self._input_dtype)
            state.setdefault("step", 0)
            state.setdefault("since_refactor", 0)
        return state[key]

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        """Activations entering the layer that owns `param` (forward pre-hook)."""
        if not self._collecting(param):
            return
        beta = self._group_of(param)["beta_a"]
        last = self._last_input
        if self.share_inputs and last is not None and last[0] is x and last[1].beta == beta:
            self.state[param]["cov"] = last[1]     # same tensor, already folded in: same gram, computed once
            self.state[param].setdefault("step", 0)
            self.state[param].setdefault("since_refactor", 0)
            return
        acc = self._stat(param, "cov", beta)
        acc.observe(x)
        if self.share_inputs:
            self._last_input = (x, acc)

    def observe_backward(self, param: torch.Tensor, delta: torch.Tensor) -> None:
        """Gradient at the layer's output (backward hook). Scaled by the token count squared: the loss is a
        mean, so autograd's `delta` carries `1/N`, and the Fisher of a mean loss is the mean of per-token
        outer products."""
        if not self._collecting(param):
            return
        flat = delta.detach().reshape(-1, delta.shape[-1])
        self._stat(param, "cov_backward", self._group_of(param)["beta_d"]).observe(
            flat, scale=float(flat.shape[0]) ** 2)

    def _ready(self, param: torch.Tensor) -> bool:
        state = self.state[param]
        return "cov" in state and state["cov"].ready and "cov_backward" in state and state["cov_backward"].ready

    def _units(self, p: torch.Tensor) -> list[slice]:
        """The row blocks of `p` that rotate as separate matrices: one block unless `row_blocks` splits it."""
        k = self.row_blocks.get(id(p), 1)
        h = p.shape[0] // k
        return [slice(i * h, (i + 1) * h) for i in range(k)]

    @staticmethod
    def _unit_state(state: dict, i: int) -> dict:
        """Per-block state. Block 0 lives in the parameter's own state, so an unsplit weight keeps the layout
        it always had; further blocks get their own dicts."""
        if i == 0:
            return state
        return state.setdefault("blocks", {}).setdefault(i, {})

    # --- the Fisher basis ---------------------------------------------------------------------------

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        """Bases with `P^T B P = I`, `P^T C P = diag(lam)` for in (A, W^T D W) and out (D, W A W^T),
        batched over matrices of equal shape."""
        dt, eps = group["compute_dtype"], group["eps"]
        by_shape: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for p in params:
            units = self._units(p)
            for i, rows in enumerate(units):
                shape = (rows.stop - rows.start, p.shape[1])
                by_shape[shape].append((p, i, rows, len(units) == 1))
        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) if whole else p.detach().to(dt)[rows] for p, _, rows, whole in members])
            A = torch.stack([self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p, *_ in members])
            D = torch.stack([
                self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt) if whole
                else self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt)[rows, rows]
                for p, _, rows, whole in members])
            Wt = W.transpose(-1, -2)
            gram_out = W @ A @ Wt
            gram_in = Wt @ D @ W
            basis_out = basis_identity_anchor(gram_out, eps) if is_identity(D) else basis_congruence(D, gram_out, eps)
            basis_in = basis_identity_anchor(A, eps) if is_identity(gram_in) else basis_congruence(A, gram_in, eps)
            for j, (p, i, _, _) in enumerate(members):
                self._unit_state(self.state[p], i)["bases"] = (
                    type(basis_in)(basis_in.P[j], basis_in.lam[j], basis_in.orthogonal, None, 1.0),
                    type(basis_out)(basis_out.P[j], basis_out.lam[j], basis_out.orthogonal, None, 1.0),
                )
                self.state[p]["since_refactor"] = 0

    # --- the step -------------------------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # fp32 must mean fp32: with TF32 the Cayley solve stops being orthogonal and W's spectrum drifts.
        with exact_fp32():
            for group in self.param_groups:
                active = [p for p in group["params"] if p.grad is not None]
                for p in active:
                    if not self._ready(p):
                        raise RuntimeError("no statistics for a parameter: attach the forward and backward hooks")
                stale = [p for p in active if "bases" not in self.state[p]]
                if stale:
                    self._refactor(stale, group)
                self._rotate(active, group)
                due = [p for p in active if self.state[p]["since_refactor"] >= group["t_fac"]]
                self._last_input = None
                if due:
                    self._refactor(due, group)
        return loss

    @staticmethod
    def _ema(state: dict, key: str, g: torch.Tensor, beta: float) -> torch.Tensor:
        if key not in state:
            state[key] = torch.zeros_like(g)
        state[key].mul_(beta).add_(g, alpha=1.0 - beta)
        return state[key]

    def _sigma(self, state: dict, group: dict, X_in: torch.Tensor, X_out: torch.Tensor) -> float:
        """`max(||X_in||_2, ||X_out||_2)` by warm power iteration."""
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        s_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        s_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        return float(torch.maximum(s_in, s_out))

    def _rotate(self, active: list[torch.Tensor], group: dict) -> None:
        dt, beta1 = group["compute_dtype"], group["beta1"]
        rows = []                                  # [p, W, X_in, X_out, block index, block rows, whole]
        by_size: dict[int, list] = {}              # side size -> [(row, slot, generator, basis)]
        for p in active:
            units = self._units(p)
            Wp, Gp = p.detach().to(dt), p.grad.detach().to(dt)
            for i, r in enumerate(units):
                whole = len(units) == 1
                state = self._unit_state(self.state[p], i)
                W, G = (Wp, Gp) if whole else (Wp[r], Gp[r])
                G_in, G_out = generators(W, G)
                if beta1 > 0.0:
                    G_in = self._ema(state, "m_in", G_in, beta1)
                    G_out = self._ema(state, "m_out", G_out, beta1)
                b_in, b_out = state["bases"]
                rows.append([p, W, None, None, i, r, whole])
                by_size.setdefault(G_in.shape[0], []).append((len(rows) - 1, 2, G_in, b_in))
                by_size.setdefault(G_out.shape[0], []).append((len(rows) - 1, 3, G_out, b_out))

        # X = F^-1 G: in the basis F is elementwise division by (lam_i + lam_j)/2
        for items in by_size.values():
            P = torch.stack([b.P for *_, b in items])
            den = torch.stack([b.denominator for *_, b in items])     # (lam_i + lam_j) / 2
            if group["damping"]:
                den = den + group["damping"] * den.amax(dim=(-2, -1), keepdim=True)
            G = torch.stack([g for _, _, g, _ in items])
            Pt = P.transpose(-1, -2)
            X = skew(P @ ((Pt @ G @ P) / den) @ Pt)
            for k, (i, slot, _, _) in enumerate(items):
                rows[i][slot] = X[k]

        # c = theta / sigma, shared by both sides of a matrix
        solves: dict[int, list] = {}
        for i, (p, W, X_in, X_out, b, _, _) in enumerate(rows):
            state = self._unit_state(self.state[p], b)
            state["since_refactor"] = self.state[p]["since_refactor"]
            c = group["lr"] / max(self._sigma(state, group, X_in, X_out), 1e-30)
            solves.setdefault(X_in.shape[0], []).append((i, 2, c, X_in))
            solves.setdefault(X_out.shape[0], []).append((i, 3, c, X_out))

        # Cayley: R = (I + c/2 X)^-1 (I - c/2 X), exactly orthogonal for skew X
        for n, items in solves.items():
            X = torch.stack([x for *_, x in items])
            half = torch.as_tensor([0.5 * c for _, _, c, _ in items], dtype=X.dtype, device=X.device).reshape(-1, 1, 1)
            eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
            if group["retraction"] == "trunc":
                A = -(half + half) * X                       # -c X: Cay(cX) and exp(-cX) agree to second order
                R = eye + A + 0.5 * (A @ A)
            elif group["retraction"] == "ns" and group["lr"] <= group["ns_guard"]:
                # Cay = 2 (I + A)^-1 - I; Newton-Schulz Z <- Z (2I - M Z) from Z0 = I - A, residual A^2 squaring
                A = half * X
                M, Z = eye + A, eye - A
                for _ in range(group["ns_iters"]):
                    Z = Z @ (2.0 * eye - M @ Z)
                R = 2.0 * Z - eye
            else:
                R = torch.linalg.solve(eye + half * X, eye - half * X)
            for k, (i, slot, _, _) in enumerate(items):
                rows[i][slot] = R[k]

        new: dict[int, list] = {}
        for p, W, R_in, R_out, b, r, whole in rows:
            if whole:
                p.copy_((R_out @ W @ R_in).to(p.dtype))
            else:
                new.setdefault(id(p), [p, []])[1].append(R_out @ W @ R_in)
        for p, blocks in new.values():
            p.copy_(torch.cat(blocks, dim=0).to(p.dtype))
        for p in active:
            state = self.state[p]
            state["step"] += 1
            state["since_refactor"] += 1

    def load_state_dict(self, state_dict) -> None:
        """torch moves plain tensors in the state; the accumulators and bases have to be moved by hand."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for key in ("cov", "cov_backward"):
                    if key in state:
                        state[key].to(p.device)
                for unit in [state, *state.get("blocks", {}).values()]:
                    if "bases" in unit:
                        unit["bases"] = tuple(
                            type(b)(b.P.to(p.device), b.lam.to(p.device), b.orthogonal, None, 1.0) for b in unit["bases"])
                    for key in ("m_in", "m_out", "angle_v_in", "angle_v_out"):
                        if isinstance(unit.get(key), torch.Tensor):
                            unit[key] = unit[key].to(p.device)


class NaturalPionFast(NaturalPion):
    """Natural Pion, efficient form: basis every 25 steps, statistics in the 6 steps before each rebuild,
    EMA 0.5 on both factors, one `A` per distinct input tensor. Everything else as `NaturalPion`."""

    def __init__(self, params, lr: float = 0.05, *, t_fac: int = 25, stats_window: int | None = 6,
                 beta_a: float = 0.5, beta_d: float = 0.5, share_inputs: bool = True, **kwargs) -> None:
        super().__init__(params, lr, t_fac=t_fac, stats_window=stats_window, beta_a=beta_a, beta_d=beta_d,
                         share_inputs=share_inputs, **kwargs)
