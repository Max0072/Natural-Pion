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
`ngd_pion.hooks.attach` for `observe` and `ngd_pion.hooks.attach_backward` for `observe_backward`.
Built from the primitives in `covariance`, `direction`, `factorization` and `linalg`.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .covariance import CovarianceAccumulator, SketchAccumulator
from .direction import generators
from .factorization import (Basis, basis_congruence, basis_congruence_from_root, basis_identity_anchor,
                            floored_inverse_root)
from .linalg import floor_eigenvalues, safe_eigh
from .linalg import exact_fp32, is_identity, skew, spectral_norm, tf32_matmul
from contextlib import nullcontext

__all__ = ["NaturalPion", "NaturalPionFast", "NaturalPionTurbo", "NaturalPionStiefel", "BY_NAME", "default_of"]


def ns_iterations(angle: float, minimum: int) -> int | None:
    """Newton-Schulz iterations for (I + Psi)^-1, ||Psi||_2 = angle / 2, that make the Cayley step exact to fp32
    round-off: after k iterations from Z0 = I - Psi the error is about (angle/2)^(2^(k+1)), so the smallest k >= minimum
    with that below 2^-24 (2 up to angle 0.2, 3 up to 0.4, 4 up to 0.8). None where the iteration converges too
    slowly to be worth it (angle / 2 >= 0.9): the exact solve then."""
    x = float(angle) / 2.0
    if x >= 0.9:
        return None
    k = minimum
    while x ** (2 ** (k + 1)) > 2.0 ** -24:
        k += 1
    return k


def _lmax(C: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Largest eigenvalue of symmetric PSD `C` (batched) by power iteration, shape (..., 1)."""
    v = torch.ones(*C.shape[:-1], 1, dtype=C.dtype, device=C.device)
    for _ in range(iters):
        v = C @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    return (v.transpose(-1, -2) @ C @ v).squeeze(-1)


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
            Newton-Schulz iterations (matrix products only), at least `ns_iters` and as many as the step's largest
            angle needs to be exact to fp32 round-off (`ns_iterations`); the exact solve above `ns_guard`. It rounds
            next to 1 with a positive bias in fp32 (kept to reproduce that measurement); "ns_delta" is the same
            iteration carried in Z - I, which removes the bias; "trunc" is
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
        ns_guard: float = 1.8,
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
        if retraction not in ("cayley", "ns", "ns_delta", "trunc"):
            raise ValueError(f"retraction must be cayley, ns, ns_delta or trunc, got {retraction!r}")
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

    def _collecting(self, param: torch.Tensor, kind: str = "a") -> bool:
        """Whether this step's statistic `kind` ("a": input moment, "d": output-gradient moment) is ever read:
        always before the first basis, then only inside the window that precedes a rebuild."""
        group = self._group_of(param)
        window = group["stats_window"]
        state = self.state[param]
        return window is None or "bases" not in state or state["since_refactor"] >= group["t_fac"] - window

    def _stat(self, param: torch.Tensor, key: str, beta: float) -> CovarianceAccumulator:
        state = self.state[param]
        if key not in state:
            k = getattr(self, "_stats_sketch", 0)
            if k:
                state[key] = SketchAccumulator(beta=beta, rank=k, stride=getattr(self, "_stats_stride", 1))
            else:
                state[key] = CovarianceAccumulator(beta=beta, dtype=torch.float32, input_dtype=self._input_dtype,
                                                   stride=getattr(self, "_stats_stride", 1))
            state.setdefault("step", 0)
            state.setdefault("since_refactor", 0)
        return state[key]

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        """Activations entering the layer that owns `param` (forward pre-hook)."""
        if not self._collecting(param, "a"):
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
        if not self._collecting(param, "d"):
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
            # D is never the identity for a trained layer, so no cheap-path check (each was a host sync)
            basis_out = self._basis_out(D, gram_out, eps, members)
            basis_in = self._basis_in(A, gram_in, eps, members)
            for j, (p, i, _, _) in enumerate(members):
                self._unit_state(self.state[p], i)["bases"] = (
                    type(basis_in)(basis_in.P[j], basis_in.lam[j], basis_in.orthogonal, None, 1.0),
                    type(basis_out)(basis_out.P[j], basis_out.lam[j], basis_out.orthogonal, None, 1.0),
                )
                self.state[p]["since_refactor"] = 0

    def _basis_in(self, A: torch.Tensor, gram_in: torch.Tensor, eps: float, members: list):
        """In-side basis of the pencil (A, W^T D W) for a batch of matrices of one shape."""
        return basis_congruence(A, gram_in, eps)

    def _basis_out(self, D: torch.Tensor, gram_out: torch.Tensor, eps: float, members: list):
        """Out-side basis of the pencil (D, W A W^T)."""
        return basis_congruence(D, gram_out, eps)

    # --- optional per-part timing (off by default; one attribute check when off) ---------------------

    profile: dict | None = None          # set to {} to accumulate GPU-synchronised seconds per part

    def _tick(self, part: str | None) -> None:
        """Close the running part and open `part`. Synchronises the GPU only when profiling is on."""
        if self.profile is None:
            return
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        last = getattr(self, "_tick_open", None)
        if last is not None:
            self.profile[last[0]] = self.profile.get(last[0], 0.0) + now - last[1]
        self._tick_open = (part, now) if part else None

    # --- the step -------------------------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # fp32 must mean fp32: with TF32 the Cayley solve stops being orthogonal and W's spectrum drifts.
        with self._step_precision():
            for group in self.param_groups:
                active = [p for p in group["params"] if p.grad is not None]
                for p in active:
                    if not self._ready(p):
                        raise RuntimeError("no statistics for a parameter: attach the forward and backward hooks")
                stale = [p for p in active if "bases" not in self.state[p]]
                if stale:
                    self._await_refresh()
                    self._tick("refresh"); self._refactor(stale, group); self._tick(None)
                self._await_refresh()
                self._rotate(active, group)
                self._tick(None)
                due = [p for p in active if self.state[p]["since_refactor"] >= group["t_fac"]]
                self._last_input = None
                if due:
                    self._refresh(due, group)
        return loss

    def _refresh(self, params: list[torch.Tensor], group: dict) -> None:
        """Rebuild the bases of `params` now (the reference path)."""
        self._tick("refresh"); self._refactor(params, group); self._tick(None)

    def _await_refresh(self) -> None:
        """Wait for a refresh running elsewhere; nothing to wait for on the reference path."""

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

    def _solve(self, P, Pt, G, den):
        """`X = F^-1 G` in the basis: elementwise division by (lam_i + lam_j)/2, made exactly skew."""
        return skew(P @ ((Pt @ G @ P) / den) @ Pt)

    def _correct(self, X, G, P, den, items, rows, group):
        """Hook after `X = F^-1 G` in the basis; the reference path uses the basis solution as it is."""
        return X

    def _step_precision(self):
        """The arithmetic of the whole step: true fp32 (TF32 off), so the retraction stays orthogonal."""
        return exact_fp32()

    def _retraction_precision(self):
        """The arithmetic of the Cayley step inside the step's own; the reference path changes nothing."""
        return nullcontext()

    def _largest_angle(self, group: dict) -> float:
        """The largest rotation angle of this step, which decides whether Newton-Schulz may stand in for the exact
        Cayley solve: theta itself, since c = theta / sigma turns every matrix by theta."""
        return group["lr"]

    def _scales(self, rows: list, group: dict) -> list:
        """`c = theta / sigma` per rotated unit, one power iteration per matrix (the reference path)."""
        cs = []
        for p, W, X_in, X_out, b, _, _ in rows:
            state = self._unit_state(self.state[p], b)
            state["since_refactor"] = self.state[p]["since_refactor"]
            cs.append(group["lr"] / max(self._sigma(state, group, X_in, X_out), 1e-30))
        return cs

    def _rotate(self, active: list[torch.Tensor], group: dict) -> None:
        self._tick("generators+momentum")
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
        self._tick("F^-1")
        for items in by_size.values():
            P = torch.stack([b.P for *_, b in items])
            den = torch.stack([b.denominator for *_, b in items])     # (lam_i + lam_j) / 2
            if group["damping"]:
                den = den + group["damping"] * den.amax(dim=(-2, -1), keepdim=True)
            G = torch.stack([g for _, _, g, _ in items])
            Pt = P.transpose(-1, -2)
            X = self._solve(P, Pt, G, den)
            X = self._correct(X, G, P, den, items, rows, group)
            for k, (i, slot, _, _) in enumerate(items):
                rows[i][slot] = X[k]

        # c = theta / sigma, shared by both sides of a matrix
        self._tick("sigma")
        cs = self._scales(rows, group)
        solves: dict[int, list] = {}
        for i, (p, W, X_in, X_out, b, _, _) in enumerate(rows):
            solves.setdefault(X_in.shape[0], []).append((i, 2, cs[i], X_in))
            solves.setdefault(X_out.shape[0], []).append((i, 3, cs[i], X_out))

        # Cayley: R = (I + c/2 X)^-1 (I - c/2 X), exactly orthogonal for skew X
        self._tick("cayley")
        angle = self._largest_angle(group)
        iters = ns_iterations(angle, group["ns_iters"]) if angle <= group["ns_guard"] else None
        with self._retraction_precision():
            for n, items in solves.items():
                X = torch.stack([x for *_, x in items])
                if isinstance(items[0][2], torch.Tensor):
                    half = (0.5 * torch.stack([c for _, _, c, _ in items])).to(X.dtype).reshape(-1, 1, 1)
                else:
                    half = torch.as_tensor([0.5 * c for _, _, c, _ in items], dtype=X.dtype, device=X.device).reshape(-1, 1, 1)
                eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
                if group["retraction"] == "trunc":
                    A = -(half + half) * X                       # -c X: Cay(cX) and exp(-cX) agree to second order
                    R = eye + A + 0.5 * (A @ A)
                elif group["retraction"] == "ns_delta" and iters is not None:
                    # The same Newton-Schulz iteration carried in the correction z = Z - I: residual r = M Z - I =
                    # A + z + A z, and Z (I - r) gives z <- z - r - z r. Nothing is rounded next to 1, which in "ns"
                    # biases R^T R - I positive by ~1e-6 per side per step in fp32 (the trained spectrum grew 1.1%
                    # over 3000 steps). R = I + 2 z.
                    A = half * X
                    z = -A
                    for _ in range(iters):
                        r = A + z + A @ z
                        z = z - r - z @ r
                    R = eye + 2.0 * z
                elif group["retraction"] == "ns" and iters is not None:
                    # Cay = 2 (I + A)^-1 - I; Newton-Schulz Z <- Z (2I - M Z) from Z0 = I - A, residual A^2 squaring
                    A = half * X
                    M, Z = eye + A, eye - A
                    for _ in range(iters):
                        Z = Z @ (2.0 * eye - M @ Z)
                    R = 2.0 * Z - eye
                else:
                    R = torch.linalg.solve(eye + half * X, eye - half * X)
                for k, (i, slot, _, _) in enumerate(items):
                    rows[i][slot] = R[k]

        self._tick("apply W")
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
    """Natural Pion, **conservative** tier: every speed-up standard and explainable, loss within 0.010 of
    `NaturalPion` at 3000 and 15000 steps, and only in Natural Pion's own parts (never those shared with Pion).
    Basis every 25 steps, statistics in the 6 steps before each rebuild, EMA 0.5 on both factors, one `A` per
    distinct input tensor, the Cayley inverse by Newton-Schulz in the correction form (`ns_delta`; the exact
    solve above `ns_guard`). Everything else as `NaturalPion`."""

    def __init__(self, params, lr: float = 0.05, *, t_fac: int = 25, stats_window: int | None = 6,
                 beta_a: float = 0.5, beta_d: float = 0.5, share_inputs: bool = True,
                 retraction: str = "ns_delta", stats_dtype: str = "bf16", batch_sigma: bool = True,
                 share_root: bool = True, direction: str = "natural", length: str = "spectral", sides: str = "joint", planes: str = "scaled",
                 attn_mult: float = 1.0, angle_mult: dict[int, float] | None = None, a_window: int | None = None, d_window: int | None = None, a_every: int = 1,
                 tf32: str = "off", fisher_dtype: str = "fp32", **kwargs) -> None:
        super().__init__(params, lr, t_fac=t_fac, stats_window=stats_window, beta_a=beta_a, beta_d=beta_d,
                         share_inputs=share_inputs, retraction=retraction, stats_dtype=stats_dtype, **kwargs)
        # Ablation switches (Section 5.4); every default is the method itself.
        #   direction  "natural": X = F^-1 G | "raw": X = G
        #   length     "spectral": c = theta / sigma(X) | "before": theta / sigma(G) | "rms": the RMS of W X_in + X_out W
        #              is theta, as in Pion | "fisher": theta / sqrt(<X, F X>) | "quad": theta min(1, <G, X> / <X, F X>)
        #              | "fixed": c = theta.  F here is the Fisher at the current W and statistics.
        #   sides      "joint": one c for both sides, from the larger sigma | "each": each side by its own
        #   planes     "scaled": X as it is | "equal": every plane of X turned by theta (polar factor, one SVD per side)
        #   attn_mult  the angle of the square (attention) matrices times this
        #   angle_mult {id(param): m}: the angle of those matrices times m (a diagnostic: which matrices want more)
        #   a_window, d_window   steps before a rebuild in which A, D are fed (default: stats_window); a_every: A every
        #              n-th step only
        if direction not in ("natural", "raw"):
            raise ValueError(f"direction must be natural or raw, got {direction!r}")
        if length not in ("spectral", "before", "rms", "fisher", "quad", "fixed"):
            raise ValueError(f"unknown length rule {length!r}")
        if sides not in ("joint", "each") or planes not in ("scaled", "equal"):
            raise ValueError("sides must be joint or each, planes scaled or equal")
        if sides == "each" and length not in ("spectral", "before"):
            raise ValueError("sides='each' is defined for the spectral rules only")
        if planes == "equal" and (length != "spectral" or sides != "joint"):
            raise ValueError("planes='equal' sets the length itself: use it with length='spectral', sides='joint'")
        if attn_mult <= 0.0 or a_every < 1:
            raise ValueError("attn_mult must be positive and a_every at least 1")
        self.direction, self.length, self.sides, self.planes = direction, length, sides, planes
        self.attn_mult, self.a_window, self.d_window, self.a_every = attn_mult, a_window, d_window, a_every
        self.angle_mult = dict(angle_mult or {})
        self._ablated = (length, sides, planes, attn_mult) != ("spectral", "joint", "scaled", 1.0) or bool(self.angle_mult)
        self.angles: dict[int, float] | None = None    # realised angle per matrix, on the ablated length path
        self._gens: dict[tuple, torch.Tensor] = {}
        self.batch_sigma = batch_sigma
        self.share_root = share_root
        # tf32: "off" -- the whole step in true fp32 (the method); "upstream" -- fp32 products in TF32 everywhere except
        # the Cayley step, so only the direction is affected (the rotation stays orthogonal to fp32 round-off);
        # "all" -- the Cayley step in TF32 too (measured, not assumed: scripts/probes/tf32_check.py)
        if tf32 not in ("off", "upstream", "all"):
            raise ValueError(f"tf32 must be off, upstream or all, got {tf32!r}")
        self.tf32 = tf32
        # fisher_dtype "bf16": the four products of F^-1 in bf16 (as Turbo), the result made exactly skew in fp32, so the
        # Cayley step still rotates; only the direction is approximated
        if fisher_dtype not in ("fp32", "bf16"):
            raise ValueError(f"fisher_dtype must be fp32 or bf16, got {fisher_dtype!r}")
        self.fisher_dtype = fisher_dtype

    def _step_precision(self):
        return exact_fp32() if self.tf32 == "off" else tf32_matmul()

    def _solve(self, P, Pt, G, den):
        if self.fisher_dtype != "bf16":
            return super()._solve(P, Pt, G, den)
        Pb, Ptb = P.to(torch.bfloat16), Pt.to(torch.bfloat16)
        Y = ((Ptb @ G.to(torch.bfloat16) @ Pb).float() / den).to(torch.bfloat16)
        return skew((Pb @ Y @ Ptb).float())

    def _retraction_precision(self):
        return exact_fp32() if self.tf32 == "upstream" else nullcontext()

    def _collecting(self, param: torch.Tensor, kind: str = "a") -> bool:
        window = self.a_window if kind == "a" else self.d_window
        if window is None and (kind == "d" or self.a_every == 1):
            return super()._collecting(param, kind)
        group, state = self._group_of(param), self.state[param]
        if "bases" not in state:
            return True
        if window is None:
            window = group["stats_window"]
        inside = window is None or state["since_refactor"] >= group["t_fac"] - window
        return inside and (kind == "d" or state.get("step", 0) % self.a_every == 0)

    def _correct(self, X, G, P, den, items, rows, group):
        if self.direction == "raw":
            X = G
        if self.length in ("before", "quad"):            # the generators, for the rules that read them
            self._gens.update({(i, slot): G[k] for k, (i, slot, _, _) in enumerate(items)})
        return X

    def _basis_in(self, A: torch.Tensor, gram_in: torch.Tensor, eps: float, members: list):
        """`A^{-1/2}` once per distinct input moment (q, k, v and gate, up share one), then the rest per matrix."""
        if not self.share_root:
            return super()._basis_in(A, gram_in, eps, members)
        slot: dict[int, int] = {}
        order = []
        for j, (p, *_) in enumerate(members):
            key = id(self.state[p]["cov"])
            if key not in slot:
                slot[key] = len(order); order.append(j)
        roots = floored_inverse_root(A[order], eps)
        root = roots[[slot[id(self.state[p]["cov"])] for p, *_ in members]]
        return basis_congruence_from_root(root, gram_in, eps)

    def _scales(self, rows: list, group: dict) -> list:
        """The power iteration batched by (side size, iterations); `c` stays on the device, so the step makes no
        host synchronisation for it. Same estimate as the reference path up to the round-off of batched products."""
        if self._ablated:
            return self._ablated_scales(rows, group)
        if not self.batch_sigma:
            return super()._scales(rows, group)
        n = len(rows)
        sig = torch.zeros(2, n, dtype=rows[0][2].dtype, device=rows[0][2].device)
        groups: dict[tuple, list] = defaultdict(list)
        states = []
        for k, (p, W, X_in, X_out, b, _, _) in enumerate(rows):
            st = self._unit_state(self.state[p], b)
            st["since_refactor"] = self.state[p]["since_refactor"]
            states.append(st)
            fresh = "angle_v_in" not in st or st["since_refactor"] == 0
            it = group["angle_warmup"] if fresh else group["angle_iters"]
            groups[(X_in.shape[0], it)].append((0, k, X_in, "angle_v_in"))
            groups[(X_out.shape[0], it)].append((1, k, X_out, "angle_v_out"))
        for (_, it), items in groups.items():
            X = torch.stack([x for _, _, x, _ in items])
            vs = [states[k].get(key) for _, k, _, key in items]
            v = None if any(x is None for x in vs) else torch.stack(vs)
            s_, v = spectral_norm(X, it, v)
            for j, (side, k, _, key) in enumerate(items):
                states[k][key] = v[j]
                sig[side, k] = s_[j]
        c = group["lr"] / sig.amax(dim=0).clamp_min(1e-30)
        return list(c.unbind(0))


    def _largest_angle(self, group: dict) -> float:
        if self._ablated and self.angles:                  # the length rules set the angle themselves
            return max(self.angles.values())
        return super()._largest_angle(group)

    def _fisher_quadratic(self, p, W, r, whole, X_in, X_out):
        """<X, F X> summed over both sides, F at the current W with the statistics as last collected."""
        st = self.state[p]
        A = st["cov"].matrix.to(W.dtype)
        D = st["cov_backward"].matrix.to(W.dtype)
        if not whole:
            D = D[r, r]
        Wt = W.transpose(-1, -2)
        S, V = Wt @ D @ W, W @ A @ Wt
        F_in = 0.5 * (S @ X_in @ A + A @ X_in @ S)
        F_out = 0.5 * (V @ X_out @ D + D @ X_out @ V)
        return (X_in * F_in).sum() + (X_out * F_out).sum()

    def _ablated_scales(self, rows: list, group: dict) -> list:
        """The length rules of Section 5.4, one matrix at a time. Each side's X is scaled in place to the step it
        takes (c_in X_in, c_out X_out) and the returned c is 1. Records the realised angle per matrix."""
        gens, self._gens = self._gens, {}
        self.angles = {}
        cs = []
        for i, (p, W, X_in, X_out, b, r, whole) in enumerate(rows):
            st = self._unit_state(self.state[p], b)
            st["since_refactor"] = self.state[p]["since_refactor"]
            theta = group["lr"] * (self.attn_mult if p.shape[0] == p.shape[1] else 1.0) * self.angle_mult.get(id(p), 1.0)
            if self.planes == "equal":
                X_in, X_out = _polar_skew(X_in), _polar_skew(X_out)
                c_in = c_out = theta
                angle = theta
            else:
                fresh = "angle_v_in" not in st or st["since_refactor"] == 0
                it = group["angle_warmup"] if fresh else group["angle_iters"]
                s_in, st["angle_v_in"] = spectral_norm(X_in, it, st.get("angle_v_in"))
                s_out, st["angle_v_out"] = spectral_norm(X_out, it, st.get("angle_v_out"))
                s_in, s_out = float(s_in), float(s_out)
                if self.length == "before":
                    g_in, g_out = gens[(i, 2)], gens[(i, 3)]
                    s_in_g, st["g_v_in"] = spectral_norm(g_in, it, st.get("g_v_in"))
                    s_out_g, st["g_v_out"] = spectral_norm(g_out, it, st.get("g_v_out"))
                    ref_in, ref_out = float(s_in_g), float(s_out_g)
                else:
                    ref_in, ref_out = s_in, s_out
                if self.length in ("spectral", "before"):
                    if self.sides == "each":
                        c_in, c_out = theta / max(ref_in, 1e-30), theta / max(ref_out, 1e-30)
                    else:
                        c_in = c_out = theta / max(ref_in, ref_out, 1e-30)
                elif self.length == "rms":
                    m, n = W.shape
                    rms = float(torch.linalg.matrix_norm(W @ X_in + X_out @ W)) / (m * n) ** 0.5
                    c_in = c_out = theta / max(rms, 1e-30)
                elif self.length == "fixed":
                    c_in = c_out = theta
                else:
                    curv = float(self._fisher_quadratic(p, W, r, whole, X_in, X_out))
                    if self.length == "fisher":
                        c_in = c_out = theta / max(curv, 1e-30) ** 0.5
                    else:
                        quad = float((gens[(i, 2)] * X_in).sum() + (gens[(i, 3)] * X_out).sum())
                        c_in = c_out = theta * min(1.0, quad / max(curv, 1e-30))
                angle = max(c_in * s_in, c_out * s_out)
            rows[i][2], rows[i][3] = c_in * X_in, c_out * X_out
            self.angles[id(p)] = max(self.angles.get(id(p), 0.0), angle)
            cs.append(1.0)
        return cs


def _polar_skew(X: torch.Tensor) -> torch.Tensor:
    """The polar factor of skew X, kept skew: X = U S V^T goes to U V^T on the planes above round-off (1e-6 of the
    largest), so every plane of X turns by the same angle. Computed as X (X^T X)^{-1/2} from the symmetric eigenproblem
    of X^T X = -X^2 (the same factor as from an SVD, which on these shapes runs tens of seconds per step on a GPU)."""
    lam, V = safe_eigh(-(X @ X))
    s = lam.clamp_min(0).sqrt()
    keep = s > 1e-6 * s.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    inv = torch.where(keep, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
    Y = X @ (V * inv.unsqueeze(-2)) @ V.transpose(-1, -2)
    return 0.5 * (Y - Y.transpose(-1, -2))


class NaturalPionTurbo(NaturalPionFast):
    """Natural Pion, **aggressive** tier: whatever is faster, no guarantees; its loss cost is measured and
    recorded, and it is used in no comparison. Now: everything batched, including the parts Pion shares
    (generators, momentum, the two-sided product), sigma without host syncs, bf16 statistics. Same mathematics
    as `NaturalPionFast` with the same options, up to the round-off of batched products."""

    def __init__(self, params, lr: float = 0.05, *, stats_dtype: str = "bf16", fisher_dtype: str = "bf16",
                 stats_stride: int = 1, c_floor: str = "shift", inv_root: str = "ns", stats_window: int | None = 1,
                 warm_refresh: int = 0, stats_sketch: int = 0, async_refresh: bool = False, **kwargs) -> None:
        """Defaults (2026-11-05): every approximation that cost nothing in loss at 3000 steps -- F^-1 in bf16, C
        shifted, B^-1/2 by Newton-Schulz, statistics in the one step before a refresh. T_fac stays 25 (50 cost 0.023).
        Approximations, all upstream of the Cayley gate (X is made exactly skew before it, so the rotation stays
        orthogonal whatever they do; only the direction changes):
          fisher_dtype "bf16": the four products of F^-1 in bf16
          stats_stride k:      A and D from every k-th token
          c_floor "shift":     C + eps * lmax(C) I in the refresh instead of flooring C's eigenvalues (no eigh of C)
          inv_root "ns":       B^-1/2 by coupled Newton-Schulz on B + eps * lmax I instead of an eigendecomposition
          warm_refresh K:      only every K-th refresh is a full one (K = 1 or 0: all full); in between the previous
                               vectors P are kept, each column rescaled to (P^T B~ P)_ii = 1 and lam_i set to the
                               Rayleigh quotient (P^T C~ P)_ii / (P^T B~ P)_ii, floored (B~, C~ shifted by eps*lmax);
                               no eigendecomposition, two products per side
          stats_sketch k:      A and D kept as Nystrom sketches of rank k (cost T n k instead of T n^2, memory n k);
                               the refresh rebuilds the rank-k approximation, the floor fills the rest
        Scheduling:
          async_refresh:       the refresh on a side stream, overlapping the next forward and backward pass (it is
                               read only by the next step's rotation); only with a statistics window, so the hooks
                               never write A and D while it reads them. Bit-identical weights; no gain measured."""
        if fisher_dtype not in ("fp32", "bf16") or c_floor not in ("eigh", "shift") or inv_root not in ("eigh", "ns"):
            raise ValueError("fisher_dtype fp32|bf16, c_floor eigh|shift, inv_root eigh|ns")
        if stats_stride < 1:
            raise ValueError("stats_stride must be at least 1")
        self._stats_stride = stats_stride
        if stats_sketch < 0:
            raise ValueError("stats_sketch must be >= 0")
        self._stats_sketch = stats_sketch
        super().__init__(params, lr, stats_dtype=stats_dtype, stats_window=stats_window, **kwargs)
        self.fisher_dtype, self.c_floor, self.inv_root = fisher_dtype, c_floor, inv_root
        if warm_refresh < 0:
            raise ValueError("warm_refresh must be >= 0")
        self.warm_refresh = warm_refresh
        if self.tf32 != "off":
            raise ValueError("tf32 is an option of NaturalPionFast's own step; Turbo computes its Cayley step itself")
        self.async_refresh = async_refresh
        self._side = None
        self._pending = None

    def _refresh(self, params: list[torch.Tensor], group: dict) -> None:
        dev = params[0].device
        if not (self.async_refresh and dev.type == "cuda" and group["stats_window"] is not None
                and group["stats_window"] < group["t_fac"] and self.profile is None):
            return super()._refresh(params, group)
        self._await_refresh()
        if self._side is None:
            self._side = torch.cuda.Stream(device=dev)
        main = torch.cuda.current_stream(dev)
        self._side.wait_stream(main)                       # W as rotated this step, A and D as collected
        with torch.cuda.stream(self._side):
            self._refactor(params, group)
            for p in params:                               # bases made here are read on the main stream later
                st = self.state[p]
                for unit in [st, *st.get("blocks", {}).values()]:
                    for b in unit.get("bases", ()):
                        b.P.record_stream(main); b.lam.record_stream(main)
        ev = torch.cuda.Event()
        ev.record(self._side)
        self._pending = (ev, main)

    def _await_refresh(self) -> None:
        if self._pending is not None:
            ev, main = self._pending
            main.wait_event(ev)
            self._pending = None

    def state_dict(self):
        if self._pending is not None:
            self._pending[0].synchronize()
            self._pending = None
        return super().state_dict()

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        K = self.warm_refresh
        if K <= 1:
            return super()._refactor(params, group)
        full, warm = [], []
        for p in params:
            st = self.state[p]
            n = st.get("refreshes", 0)
            (warm if "bases" in st and n % K != 0 else full).append(p)
            st["refreshes"] = n + 1
        if full:
            super()._refactor(full, group)
        if warm:
            self._warm_refactor(warm, group)

    def _warm_refactor(self, params: list[torch.Tensor], group: dict) -> None:
        dt, eps = group["compute_dtype"], group["eps"]
        by_shape: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for p in params:
            units = self._units(p)
            for i, rows in enumerate(units):
                by_shape[(rows.stop - rows.start, p.shape[1])].append((p, i, rows, len(units) == 1))

        def rescale(P, B, C):
            n = B.shape[-1]
            eye = torch.eye(n, dtype=B.dtype, device=B.device)
            Bs = B + eps * _lmax(B).unsqueeze(-1) * eye
            Cs = C + eps * _lmax(C).unsqueeze(-1) * eye
            g = (P * (Bs @ P)).sum(-2)                         # diag(P^T B~ P)
            h = (P * (Cs @ P)).sum(-2)                         # diag(P^T C~ P)
            g = g.clamp_min(torch.finfo(g.dtype).tiny)
            return P / g.sqrt().unsqueeze(-2), floor_eigenvalues(h / g, eps)

        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) if whole else p.detach().to(dt)[rows] for p, _, rows, whole in members])
            A = torch.stack([self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p, *_ in members])
            D = torch.stack([
                self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt) if whole
                else self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt)[rows, rows]
                for p, _, rows, whole in members])
            Wt = W.transpose(-1, -2)
            old = [self._unit_state(self.state[p], i)["bases"] for p, i, _, _ in members]
            P_in, l_in = rescale(torch.stack([b[0].P for b in old]), A, Wt @ D @ W)
            P_out, l_out = rescale(torch.stack([b[1].P for b in old]), D, W @ A @ Wt)
            for j, (p, i, _, _) in enumerate(members):
                self._unit_state(self.state[p], i)["bases"] = (
                    Basis(P=P_in[j], lam=l_in[j], orthogonal=False), Basis(P=P_out[j], lam=l_out[j], orthogonal=False))
                self.state[p]["since_refactor"] = 0

    def _root(self, B: torch.Tensor, eps: float) -> torch.Tensor:
        if self.inv_root == "eigh":
            return floored_inverse_root(B, eps)
        eye = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
        Bd = B + eps * _lmax(B).unsqueeze(-1) * eye
        s = torch.linalg.matrix_norm(Bd, ord="fro", keepdim=True)
        Y, Z = Bd / s, eye.expand_as(B).clone()
        for _ in range(12):
            T = 0.5 * (3.0 * eye - Z @ Y)
            Y, Z = Y @ T, T @ Z
        return Z / s.sqrt()

    def _from_root(self, root: torch.Tensor, C: torch.Tensor, eps: float):
        if self.c_floor == "eigh":
            return basis_congruence_from_root(root, C, eps)
        eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
        C = C + eps * _lmax(C).unsqueeze(-1) * eye
        M = root @ C @ root
        lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
        return Basis(P=root @ Q, lam=floor_eigenvalues(lam, eps), orthogonal=False)

    def _basis_out(self, D, gram_out, eps, members):
        if self.c_floor == "eigh" and self.inv_root == "eigh":
            return super()._basis_out(D, gram_out, eps, members)
        return self._from_root(self._root(D, eps), gram_out, eps)

    def _basis_in(self, A, gram_in, eps, members):
        if self.c_floor == "eigh" and self.inv_root == "eigh":
            return super()._basis_in(A, gram_in, eps, members)
        slot, order = {}, []
        for j, (p, *_) in enumerate(members):
            key = id(self.state[p]["cov"])
            if key not in slot:
                slot[key] = len(order); order.append(j)
        roots = self._root(A[order], eps)
        root = roots[[slot[id(self.state[p]["cov"])] for p, *_ in members]]
        return self._from_root(root, gram_in, eps)

    @torch.no_grad()
    def _rotate(self, active: list[torch.Tensor], group: dict) -> None:
        dt, beta1, lr = group["compute_dtype"], group["beta1"], group["lr"]
        units = []                                   # (param, block, rows, whole, state, W, G)
        for p in active:
            blocks = self._units(p)
            Wp, Gp = p.detach().to(dt), p.grad.detach().to(dt)
            for i, r in enumerate(blocks):
                whole = len(blocks) == 1
                st = self._unit_state(self.state[p], i)
                st["since_refactor"] = self.state[p]["since_refactor"]
                units.append((p, i, r, whole, st, Wp if whole else Wp[r], Gp if whole else Gp[r]))
        n_u = len(units)
        by_shape: dict[tuple, list[int]] = defaultdict(list)
        for k, u in enumerate(units):
            by_shape[tuple(u[5].shape)].append(k)

        # generators, batched by shape
        self._tick("generators+momentum")
        g_in, g_out = [None] * n_u, [None] * n_u
        for ks in by_shape.values():
            W = torch.stack([units[k][5] for k in ks]); G = torch.stack([units[k][6] for k in ks])
            Wt = W.transpose(-1, -2)
            a, b = Wt @ G, G @ Wt
            a = 0.5 * (a - a.transpose(-1, -2)); b = 0.5 * (b - b.transpose(-1, -2))
            for j, k in enumerate(ks):
                g_in[k], g_out[k] = a[j], b[j]
        # Lie momentum, one multi-tensor call
        if beta1 > 0.0:
            for k, u in enumerate(units):
                u[4].setdefault("m_in", torch.zeros_like(g_in[k]))
                u[4].setdefault("m_out", torch.zeros_like(g_out[k]))
            ms = [u[4]["m_in"] for u in units] + [u[4]["m_out"] for u in units]
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, g_in + g_out, alpha=1.0 - beta1)
            g_in, g_out = ms[:n_u], ms[n_u:]

        # X = F^-1 G, batched by side size
        self._tick("F^-1")
        x_in, x_out = [None] * n_u, [None] * n_u
        sides: dict[int, list] = defaultdict(list)
        for k, u in enumerate(units):
            b_in, b_out = u[4]["bases"]
            sides[g_in[k].shape[0]].append((x_in, k, g_in[k], b_in))
            sides[g_out[k].shape[0]].append((x_out, k, g_out[k], b_out))
        for items in sides.values():
            P = torch.stack([b.P for *_, b in items]); Pt = P.transpose(-1, -2)
            den = torch.stack([b.denominator for *_, b in items])
            if group["damping"]:
                den = den + group["damping"] * den.amax(dim=(-2, -1), keepdim=True)
            G = torch.stack([g for _, _, g, _ in items])
            if self.fisher_dtype == "bf16":
                Pb, Ptb = P.to(torch.bfloat16), Pt.to(torch.bfloat16)
                Y = ((Ptb @ G.to(torch.bfloat16) @ Pb).float() / den).to(torch.bfloat16)
                X = skew((Pb @ Y @ Ptb).float())
            else:
                X = skew(P @ ((Pt @ G @ P) / den) @ Pt)
            for j, (dst, k, _, _) in enumerate(items):
                dst[k] = X[j]

        # sigma = max(||X_in||, ||X_out||) by warm power iteration, batched by (size, iterations); no host sync
        self._tick("sigma")
        sig = torch.zeros(2, n_u, dtype=dt, device=units[0][5].device)
        pw: dict[tuple, list] = defaultdict(list)
        for k, u in enumerate(units):
            st = u[4]
            fresh = "angle_v_in" not in st or st["since_refactor"] == 0
            it = group["angle_warmup"] if fresh else group["angle_iters"]
            pw[(x_in[k].shape[0], it)].append((0, k, x_in[k], "angle_v_in"))
            pw[(x_out[k].shape[0], it)].append((1, k, x_out[k], "angle_v_out"))
        for (n, it), items in pw.items():
            X = torch.stack([x for _, _, x, _ in items])
            vs = [units[k][4].get(key) for _, k, _, key in items]
            v = None if any(x is None for x in vs) else torch.stack(vs)
            s, v = spectral_norm(X, it, v)
            for j, (side, k, _, key) in enumerate(items):
                units[k][4][key] = v[j]
                sig[side, k] = s[j]
        c = lr / sig.amax(dim=0).clamp_min(1e-30)              # (n_u,)

        # Cayley, batched by size: Newton-Schulz in Z - I, or the exact solve above the guard
        self._tick("cayley")
        r_in, r_out = [None] * n_u, [None] * n_u
        cay: dict[int, list] = defaultdict(list)
        for k in range(n_u):
            cay[x_in[k].shape[0]].append((r_in, k, x_in[k]))
            cay[x_out[k].shape[0]].append((r_out, k, x_out[k]))
        iters = ns_iterations(lr, group["ns_iters"]) if lr <= group["ns_guard"] else None
        use_ns = group["retraction"] == "ns_delta" and iters is not None
        for n, items in cay.items():
            X = torch.stack([x for _, _, x in items])
            idx = torch.tensor([k for _, k, _ in items], device=X.device)
            A = (0.5 * c[idx]).reshape(-1, 1, 1) * X
            eye = torch.eye(n, dtype=X.dtype, device=X.device)
            if use_ns:
                z = -A
                for _ in range(iters):
                    r = A + z + A @ z
                    z = z - r - z @ r
                R = eye + 2.0 * z
            else:
                E = eye.expand_as(X)
                R = torch.linalg.solve(E + A, E - A)
            for j, (dst, k, _) in enumerate(items):
                dst[k] = R[j]

        # W <- R_out W R_in, batched by shape
        self._tick("apply W")
        new = [None] * n_u
        for ks in by_shape.values():
            W = torch.stack([units[k][5] for k in ks])
            Ro = torch.stack([r_out[k] for k in ks]); Ri = torch.stack([r_in[k] for k in ks])
            Y = Ro @ W @ Ri
            for j, k in enumerate(ks):
                new[k] = Y[j]
        by_param: dict[int, list] = {}
        for k, u in enumerate(units):
            p, whole = u[0], u[3]
            if whole:
                p.copy_(new[k].to(p.dtype))
            else:
                by_param.setdefault(id(p), [p, []])[1].append(new[k])
        for p, blocks in by_param.values():
            p.copy_(torch.cat(blocks, dim=0).to(p.dtype))
        for p in active:
            self.state[p]["step"] += 1
            self.state[p]["since_refactor"] += 1


def stiefel_basis(B: torch.Tensor, Cq: torch.Tensor, Q: torch.Tensor, eps: float):
    """The wide side of a rectangular matrix without its invisible block (batched).

    The side has size s, the other side k < s; B (s x s) is this side's moment, Cq = Q^T C Q (k x k) the gram of
    rank k (W A W^T on the out side, W^T D W on the in side) seen in Q (s x k), an orthonormal basis of the k
    directions W occupies on this side. The pencil (B_f, C) with C unfloored has k eigenvalues Lam in the range of C
    and s - k zeros (floored to f = eps * max Lam); the zero-zero block of F^-1 is the one that cannot move W.
    Returns (P_r, lam_r, f, H): P_r (s x k) the range part of the basis, lam_r floored, H = B_f^-1 - P_r P_r^T.
    One eigendecomposition of size s; the rest is k x k."""
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, eps)
    E = (Ub * (1.0 / wb).unsqueeze(-2)) @ Ub.transpose(-1, -2)            # B_f^-1
    wc, Uc = safe_eigh(0.5 * (Cq + Cq.transpose(-1, -2)))
    Ch = (Uc * wc.clamp_min(0).sqrt().unsqueeze(-2)) @ Uc.transpose(-1, -2)  # Cq^1/2
    EQ = E @ Q
    S = Ch @ (Q.transpose(-1, -2) @ EQ) @ Ch
    lam, U = safe_eigh(0.5 * (S + S.transpose(-1, -2)))
    tiny = torch.finfo(lam.dtype).tiny
    Pr = EQ @ Ch @ (U * lam.clamp_min(tiny).rsqrt().unsqueeze(-2))
    lam_f = floor_eigenvalues(lam, eps)
    f = eps * lam.amax(dim=-1, keepdim=True)
    H = E - Pr @ Pr.transpose(-1, -2)
    return Pr, lam_f, f, H


def stiefel_direction(G: torch.Tensor, Pr, lam, f, H, Q: torch.Tensor, lowp: torch.dtype | None = None):
    """F^-1 G on the wide side without the invisible block, projected off the current kernel, as factors
    X' = L @ Mt with L = [Q, R2] (s x 2k) and Mt = [Q^T X; Q^T] (2k x s). X itself is never formed: two products of
    order s^2 k (G P_r and (P_r^T G) H), everything else s k^2."""
    if lowp is None:
        GP = G @ Pr                                                        # s x k
    else:                                                                  # the two s^2 k products on tensor cores
        GP = (G.to(lowp) @ Pr.to(lowp)).float()
    Prt = Pr.transpose(-1, -2)
    K = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
    Y = (Prt @ GP) / K
    Y = 0.5 * (Y - Y.transpose(-1, -2))                                    # skew exactly (bf16 products are not)
    d = (2.0 / (lam + f)).unsqueeze(-1)                                    # k x 1
    if lowp is None:
        Z = (-GP.transpose(-1, -2)) @ H                                    # P_r^T G H, k x s
    else:
        Z = ((-GP.transpose(-1, -2)).to(lowp) @ H.to(lowp)).float()
    # X = P_r Y P_r^T + P_r D Z - Z^T D P_r^T  (D = diag d)
    a = Q.transpose(-1, -2) @ Pr                                           # Q^T P_r, k x k
    ZQ = Z @ Q                                                             # k x k
    QX = a @ Y @ Prt + (a * d.transpose(-1, -2)) @ Z - ZQ.transpose(-1, -2) @ (d * Prt)          # Q^T X, k x s
    # X skew gives X Q = -(Q^T X)^T; building R2 from QX itself makes L @ Mt skew by construction (the Cayley gate)
    S = QX @ Q
    S = 0.5 * (S - S.transpose(-1, -2))
    R2 = -QX.transpose(-1, -2) - Q @ S
    return torch.cat([Q, R2], dim=-1), torch.cat([QX, Q.transpose(-1, -2)], dim=-2)


def _factored_norm(L, Mt, v, iters):
    """||L Mt||_2 for skew L Mt by warm power iteration on the factors."""
    if v is None:
        v = torch.ones(*L.shape[:-2], L.shape[-2], 1, dtype=L.dtype, device=L.device)
    for _ in range(max(iters, 1)):
        u = L @ (Mt @ v)
        u = u / u.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        v = -(L @ (Mt @ u))
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    return (L @ (Mt @ v)).norm(dim=(-2, -1)), v


def _orth_basis(W: torch.Tensor, side: str) -> torch.Tensor:
    """Orthonormal basis of W's column space (out side) or row space (in side), by Cholesky of the small gram."""
    if side == "out":
        L = torch.linalg.cholesky_ex(W.transpose(-1, -2) @ W)[0]
        return torch.linalg.solve_triangular(L, W.transpose(-1, -2), upper=False).transpose(-1, -2)
    L = torch.linalg.cholesky_ex(W @ W.transpose(-1, -2))[0]
    return torch.linalg.solve_triangular(L, W, upper=False).transpose(-1, -2)


def default_of(cls: type, name: str):
    """The default a Natural Pion class gives keyword `name` (its own, or the nearest base class's)."""
    import inspect
    for k in cls.__mro__:
        p = inspect.signature(k.__init__).parameters.get(name) if "__init__" in vars(k) else None
        if p is not None and p.default is not inspect.Parameter.empty:
            return p.default
    raise KeyError(f"{cls.__name__} has no default for {name!r}")


class NaturalPionStiefel(NaturalPionTurbo):
    """Turbo with the wide side of every rectangular matrix in Stiefel form: its generator never forms the block that
    cannot move W, F^-1 costs O(s^2 k) instead of O(s^3) and the Cayley step is a Woodbury solve of size 2k. Square
    matrices and the narrow sides are exactly as in Turbo. The wide side's C is not floored (its rank-k structure is
    what makes the reduction exact). Aggressive tier: used in no comparison."""

    def __init__(self, params, lr: float = 0.05, **kwargs) -> None:
        super().__init__(params, lr, **kwargs)

    @staticmethod
    def _wide(shape) -> str | None:
        m, n = shape
        return "out" if m > n else "in" if n > m else None

    @staticmethod
    def _placeholder(B: torch.Tensor):
        """The wide side's slot in `bases`: never read (the Stiefel state replaces it), so nothing is computed."""
        z = B.new_zeros(B.shape[:-2] + (1, 1))
        return Basis(P=z, lam=z[..., 0], orthogonal=False)

    def _stiefel_members(self, members) -> bool:
        return all(len(self._units(p)) == 1 for p, *_ in members)

    def _basis_out(self, D, gram_out, eps, members):
        if self._wide(tuple(members[0][0].shape)) == "out" and self._stiefel_members(members):
            return self._placeholder(D)
        return super()._basis_out(D, gram_out, eps, members)

    def _basis_in(self, A, gram_in, eps, members):
        if self._wide(tuple(members[0][0].shape)) == "in" and self._stiefel_members(members):
            return self._placeholder(A)
        return super()._basis_in(A, gram_in, eps, members)

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        super()._refactor(params, group)                                   # narrow sides and square matrices
        dt, eps = group["compute_dtype"], group["eps"]
        by_shape: dict[tuple, list] = defaultdict(list)
        for p in params:
            if self._wide(p.shape) and len(self._units(p)) == 1:
                by_shape[tuple(p.shape)].append(p)
        for shape, ps in by_shape.items():
            side = self._wide(shape)
            W = torch.stack([p.detach().to(dt) for p in ps])
            A = torch.stack([self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p in ps])
            D = torch.stack([self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt) for p in ps])
            Q = _orth_basis(W, side)
            Qt = Q.transpose(-1, -2)
            if side == "out":                                              # C = W A W^T, Cq = (Q^T W) A (W^T Q)
                B, QW = D, Qt @ W
                Cq = QW @ A @ QW.transpose(-1, -2)
            else:                                                          # C = W^T D W, Cq = (W Q)^T D (W Q)
                B, WQ = A, W @ Q
                Cq = WQ.transpose(-1, -2) @ D @ WQ
            Pr, lam, f, H = stiefel_basis(B, Cq, Q, eps)
            for j, p in enumerate(ps):
                self.state[p]["stiefel"] = (Pr[j], lam[j], f[j], H[j])

    @torch.no_grad()
    def _rotate(self, active: list[torch.Tensor], group: dict) -> None:
        st_ps = [p for p in active if "stiefel" in self.state[p]]
        rest = [p for p in active if "stiefel" not in self.state[p]]
        if rest:
            super()._rotate(rest, group)
            self._tick(None)
        if not st_ps:
            return
        dt, beta1, lr = group["compute_dtype"], group["beta1"], group["lr"]
        by_shape: dict[tuple, list] = defaultdict(list)
        for p in st_ps:
            by_shape[tuple(p.shape)].append(p)
        for shape, ps in by_shape.items():
            self._tick("generators+momentum")
            side = self._wide(shape)
            W = torch.stack([p.detach().to(dt) for p in ps])
            Gw = torch.stack([p.grad.detach().to(dt) for p in ps])
            Wt = W.transpose(-1, -2)
            a, b = Wt @ Gw, Gw @ Wt
            g_in, g_out = 0.5 * (a - a.transpose(-1, -2)), 0.5 * (b - b.transpose(-1, -2))
            sts = [self.state[p] for p in ps]
            if beta1 > 0.0:
                for k, st in enumerate(sts):
                    st.setdefault("m_in", torch.zeros_like(g_in[k])); st.setdefault("m_out", torch.zeros_like(g_out[k]))
                ms = [st["m_in"] for st in sts] + [st["m_out"] for st in sts]
                torch._foreach_mul_(ms, beta1)
                torch._foreach_add_(ms, list(g_in.unbind(0)) + list(g_out.unbind(0)), alpha=1.0 - beta1)
                g_in = torch.stack([st["m_in"] for st in sts]); g_out = torch.stack([st["m_out"] for st in sts])
            # narrow side: the ordinary basis solve
            self._tick("F^-1 narrow")
            nb = [st["bases"][0 if side == "out" else 1] for st in sts]
            Gn = g_in if side == "out" else g_out
            P = torch.stack([x.P for x in nb]); den = torch.stack([x.denominator for x in nb]); Pt = P.transpose(-1, -2)
            if self.fisher_dtype == "bf16":                                # as Turbo: F^-1 products in bf16
                Pb, Ptb = P.to(torch.bfloat16), Pt.to(torch.bfloat16)
                Xn = skew((Pb @ ((Ptb @ Gn.to(torch.bfloat16) @ Pb).float() / den).to(torch.bfloat16) @ Ptb).float())
            else:
                Xn = skew(P @ ((Pt @ Gn @ P) / den) @ Pt)
            # wide side: Stiefel form
            self._tick("F^-1 wide (Stiefel)")
            Gwd = g_out if side == "out" else g_in
            Pr = torch.stack([st["stiefel"][0] for st in sts]); lam = torch.stack([st["stiefel"][1] for st in sts])
            f = torch.stack([st["stiefel"][2] for st in sts]); H = torch.stack([st["stiefel"][3] for st in sts])
            Q = _orth_basis(W, side)
            L, Mt = stiefel_direction(Gwd, Pr, lam, f, H, Q,
                                      lowp=torch.bfloat16 if self.fisher_dtype == "bf16" else None)
            # sigma = max over the two sides, warm power iteration
            self._tick("sigma")
            fresh = any("v_wide" not in st or st["since_refactor"] == 0 for st in sts)
            it = group["angle_warmup"] if fresh else group["angle_iters"]
            vn = None if any("v_narrow" not in st for st in sts) else torch.stack([st["v_narrow"] for st in sts])
            sn, vn = spectral_norm(Xn, it, vn)
            vw = None if any("v_wide" not in st for st in sts) else torch.stack([st["v_wide"] for st in sts])
            sw, vw = _factored_norm(L, Mt, vw, it)
            for k, st in enumerate(sts):
                st["v_narrow"], st["v_wide"] = vn[k], vw[k]
            c = (lr / torch.maximum(sn, sw).clamp_min(1e-30)).reshape(-1, 1, 1)
            # narrow Cayley by Newton-Schulz in Z - I (as Turbo), wide by Woodbury
            self._tick("Cayley + apply")
            A_n = 0.5 * c * Xn
            eye = torch.eye(A_n.shape[-1], dtype=dt, device=A_n.device)
            iters = ns_iterations(lr, group["ns_iters"]) if lr <= group["ns_guard"] else None
            if group["retraction"] == "ns_delta" and iters is not None:
                z = -A_n
                for _ in range(iters):
                    r = A_n + z + A_n @ z
                    z = z - r - z @ r
                Rn = eye + 2.0 * z
            else:
                E = eye.expand_as(A_n)
                Rn = torch.linalg.solve(E + A_n, E - A_n)
            k2 = L.shape[-1]
            Bk = 0.5 * c * (Mt @ L)                                         # K = I + Bk, 2k x 2k
            if group["retraction"] == "ns_delta" and iters is not None:
                # K^-1 by Newton-Schulz in the correction form (Section 4.8), on the 2k x 2k core: Bk has the
                # small eigenvalues of (c/2) X', so two iterations are exact to fp32
                z = -Bk
                for _ in range(iters):
                    r = Bk + z + Bk @ z
                    z = z - r - z @ r
                Kinv = torch.eye(k2, dtype=dt, device=L.device) + z
            else:
                prev = torch.backends.cuda.preferred_linalg_library()
                try:    # MAGMA's batched solve is meant for small matrices
                    if L.is_cuda:
                        torch.backends.cuda.preferred_linalg_library("cusolver")
                    Kinv = torch.linalg.inv(torch.eye(k2, dtype=dt, device=L.device) + Bk)
                finally:
                    if L.is_cuda:
                        torch.backends.cuda.preferred_linalg_library(prev)
            if side == "out":
                W1 = W @ Rn                                                   # in side (narrow)
                W2 = W1 - c * (L @ (Kinv @ (Mt @ W1)))                       # out side (wide) by Woodbury
            else:
                W1 = Rn @ W                                                   # out side (narrow)
                W2 = W1 - c * ((W1 @ L) @ (Kinv @ Mt))                       # in side (wide)
            for k, p in enumerate(ps):
                p.copy_(W2[k].to(p.dtype))
                self.state[p]["step"] += 1
                self.state[p]["since_refactor"] += 1
        self._tick(None)


# the optimizer names of the harness
BY_NAME = {"natural-pion": NaturalPion, "natural-pion-fast": NaturalPionFast,
           "natural-pion-turbo": NaturalPionTurbo, "natural-pion-stiefel": NaturalPionStiefel}
