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

from .covariance import CovarianceAccumulator, SketchAccumulator
from .direction import generators
from .factorization import (Basis, basis_congruence, basis_congruence_from_root, basis_identity_anchor,
                            floored_inverse_root)
from .linalg import floor_eigenvalues, safe_eigh
from .linalg import exact_fp32, is_identity, skew, spectral_norm

__all__ = ["NaturalPion", "NaturalPionFast", "NaturalPionTurbo"]


def drop_kernel(X: torch.Tensor, W: torch.Tensor, side: str) -> torch.Tensor:
    """Remove from the generator X the block that cannot move a rectangular W.

    Out side of W (m x n, m > n): rotations inside the complement of W's column space leave W unchanged. With Q an
    orthonormal basis of that column space and N = I - Q Q^T, return X - N X N, i.e. Q Q^T X + X Q Q^T - Q Q^T X Q Q^T.
    In side (m < n): the same with the row space. Skew in, skew out; X' W = X W (out) and W X' = W X (in) exactly."""
    if side == "out":
        L = torch.linalg.cholesky(W.transpose(-1, -2) @ W)                  # n x n, the small side
        Q = torch.linalg.solve_triangular(L, W.transpose(-1, -2), upper=False).transpose(-1, -2)   # W L^-T, m x n
    else:
        L = torch.linalg.cholesky(W @ W.transpose(-1, -2))                  # m x m
        Q = torch.linalg.solve_triangular(L, W, upper=False).transpose(-1, -2)                      # W^T L^-T, n x m
    QX = Q.transpose(-1, -2) @ X
    XQ = X @ Q
    return Q @ QX + XQ @ Q.transpose(-1, -2) - Q @ (QX @ Q) @ Q.transpose(-1, -2)


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
            `ns_iters` Newton-Schulz iterations (matrix products only), which is exact to fp32 round-off for
            angles well below 1 and is used only while the scheduled angle is at most `ns_guard`, but rounds
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
        with exact_fp32():
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

    def _correct(self, X, G, P, den, items, rows, group):
        """Hook after `X = F^-1 G` in the basis; the reference path uses the basis solution as it is."""
        return X

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
            X = skew(P @ ((Pt @ G @ P) / den) @ Pt)
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
            elif group["retraction"] == "ns_delta" and group["lr"] <= group["ns_guard"]:
                # The same Newton-Schulz iteration carried in the correction z = Z - I: residual r = M Z - I =
                # A + z + A z, and Z (I - r) gives z <- z - r - z r. Nothing is rounded next to 1, which in "ns"
                # biases R^T R - I positive by ~1e-6 per side per step in fp32 (the trained spectrum grew 1.1%
                # over 3000 steps). R = I + 2 z.
                A = half * X
                z = -A
                for _ in range(group["ns_iters"]):
                    r = A + z + A @ z
                    z = z - r - z @ r
                R = eye + 2.0 * z
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
                 retraction: str = "ns_delta", batch_sigma: bool = True, share_root: bool = True,
                 async_refresh: bool = False, cg_iters: int = 0, drop_kernel: bool = False, **kwargs) -> None:
        super().__init__(params, lr, t_fac=t_fac, stats_window=stats_window, beta_a=beta_a, beta_d=beta_d,
                         share_inputs=share_inputs, retraction=retraction, **kwargs)
        self.batch_sigma = batch_sigma
        self.share_root = share_root
        # A refresh is read only by the next step's rotation, so on a GPU it runs on a side stream, overlapping the
        # next forward and backward pass. Only with a statistics window: otherwise the hooks would write A and D
        # while the refresh reads them.
        self.async_refresh = async_refresh
        # cg_iters > 0: that many iterations of preconditioned conjugate gradient on F(X) = G with the *current* W
        # (S = W^T D W, V = W A W^T; A, D as last collected, every factor shifted by eps*lmax), started from the
        # basis solution and preconditioned by it. Corrects the basis for the rotation of W since the last refresh.
        if cg_iters < 0:
            raise ValueError("cg_iters must be >= 0")
        self.cg_iters = cg_iters
        # drop_kernel: on the wide side of a rectangular matrix, remove the block of X that cannot move W before sigma
        # and Cayley (it inflated sigma by a median 1.07, up to 1.39 on down projections; kernel_block.py)
        self.drop_kernel = drop_kernel
        self._side = None
        self._pending = None

    def _correct(self, X, G, P, den, items, rows, group):
        if self.cg_iters:
            X = self._cg(X, G, P, den, items, rows, group)
        if self.drop_kernel:
            out = []
            for k, (i, slot, _, _) in enumerate(items):
                W = rows[i][1]
                m, n = W.shape
                wide = (slot == 3 and m > n) or (slot == 2 and n > m)
                out.append(drop_kernel(X[k], W, "out" if slot == 3 else "in") if wide else X[k])
            X = torch.stack(out)
        return X

    def _cg(self, X, G, P, den, items, rows, group):
        eps = group["eps"]
        Bs, Cs = [], []
        for i, slot, _, _ in items:
            p, W, _, _, b, r, whole = rows[i]
            st = self.state[p]
            A = st["cov"].matrix.to(W.dtype)
            D = st["cov_backward"].matrix.to(W.dtype)
            if not whole:
                D = D[r, r]
            if slot == 2:                                   # in side: F(X) = (S X A + A X S)/2, S = W^T D W
                Bs.append(A); Cs.append(W.transpose(-1, -2) @ D @ W)
            else:                                           # out side: (V X D + D X V)/2, V = W A W^T
                Bs.append(D); Cs.append(W @ A @ W.transpose(-1, -2))
        B, C = torch.stack(Bs), torch.stack(Cs)
        eye = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
        B = B + eps * _lmax(B).unsqueeze(-1) * eye
        C = C + eps * _lmax(C).unsqueeze(-1) * eye
        Pt = P.transpose(-1, -2)
        F = lambda Y: 0.5 * (C @ Y @ B + B @ Y @ C)
        Minv = lambda R: skew(P @ ((Pt @ R @ P) / den) @ Pt)
        dot = lambda a, b: (a * b).sum(dim=(-2, -1), keepdim=True)
        r = G - F(X)
        z = Minv(r)
        d = z
        rz = dot(r, z)
        tiny = torch.finfo(X.dtype).tiny
        for k in range(self.cg_iters):
            Ad = F(d)
            alpha = rz / dot(d, Ad).clamp_min(tiny)
            X = X + alpha * d
            if k + 1 == self.cg_iters:
                break
            r = r - alpha * Ad
            z = Minv(r)
            rz_new = dot(r, z)
            d = z + (rz_new / rz.clamp_min(tiny)) * d
            rz = rz_new
        return skew(X)

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


class NaturalPionTurbo(NaturalPionFast):
    """Natural Pion, **aggressive** tier: whatever is faster, no guarantees; its loss cost is measured and
    recorded, and it is used in no comparison. Now: everything batched, including the parts Pion shares
    (generators, momentum, the two-sided product), sigma without host syncs, bf16 statistics. Same mathematics
    as `NaturalPionFast` with the same options, up to the round-off of batched products."""

    def __init__(self, params, lr: float = 0.05, *, stats_dtype: str = "bf16", fisher_dtype: str = "bf16",
                 stats_stride: int = 1, c_floor: str = "shift", inv_root: str = "ns", stats_window: int | None = 1,
                 warm_refresh: int = 0, stats_sketch: int = 0, **kwargs) -> None:
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
                               the refresh rebuilds the rank-k approximation, the floor fills the rest"""
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
            Bs = B + eps * self._lmax(B).unsqueeze(-1) * eye
            Cs = C + eps * self._lmax(C).unsqueeze(-1) * eye
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

    @staticmethod
    def _lmax(C: torch.Tensor, iters: int = 30) -> torch.Tensor:
        v = torch.ones(*C.shape[:-1], 1, dtype=C.dtype, device=C.device)
        for _ in range(iters):
            v = C @ v
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        return (v.transpose(-1, -2) @ C @ v).squeeze(-1)      # (..., 1)

    def _root(self, B: torch.Tensor, eps: float) -> torch.Tensor:
        if self.inv_root == "eigh":
            return floored_inverse_root(B, eps)
        eye = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
        Bd = B + eps * self._lmax(B).unsqueeze(-1) * eye
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
        C = C + eps * self._lmax(C).unsqueeze(-1) * eye
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
        use_ns = group["retraction"] == "ns_delta" and lr <= group["ns_guard"]
        for n, items in cay.items():
            X = torch.stack([x for _, _, x in items])
            idx = torch.tensor([k for _, k, _ in items], device=X.device)
            A = (0.5 * c[idx]).reshape(-1, 1, 1) * X
            eye = torch.eye(n, dtype=X.dtype, device=X.device)
            if use_ns:
                z = -A
                for _ in range(group["ns_iters"]):
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
