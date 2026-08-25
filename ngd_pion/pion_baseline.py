"""Vanilla Pion, as the baseline NGD-Pion is measured against.

A faithful transcription of `pion.py` from the reference implementation
(github.com/Sphere-AI-Lab/pion), reduced to what the comparison needs: the
same generators, the same RMS scaling, the same ambient momentum, and the same
degree-2 truncated exponential -- plus switches to turn each off, which is
what an ablated baseline requires.

The point of having this here is that the comparison must differ in exactly
one thing. NGD-Pion drops momentum and scaling and uses an exact retraction;
measuring it against published Pion would confound four changes at once, so
the baseline has to be runnable with those same three settings.

Two departures from their code, both forced:

* they offer no `momentum="none"` -- only `lie_lie` and
  `transported_ambient_ambient` -- so switching it off needs this flag rather
  than theirs;
* with `scaling="none"` their code returns unscaled generators, and where the
  learning rate then enters is not visible in the file. Here it multiplies the
  generators, which is the only reading consistent with the scaled branch.
"""

from __future__ import annotations

import math

import torch

from .direction import generators
from .linalg import cayley, exact_fp32, skew

__all__ = ["Pion"]


def _truncated_exp(A: torch.Tensor, degree: int) -> torch.Tensor:
    """`I + A + A^2/2! + ... + A^degree/degree!` -- their retraction.

    Orthogonal only to `O(A^(degree+1))`, so the singular values of `W` drift
    rather than hold: measured at `2.4e-4` relative over 2000 steps at an
    angle of 0.05, against `7e-15` for Cayley, and it inflates outright at
    large angles (spectral norm 12.5 at an angle of 5).
    """
    out = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A).clone()
    term = out.clone()
    for i in range(1, degree + 1):
        term = term @ A / float(i)
        out = out + term
    return out


class Pion(torch.optim.Optimizer):
    """Spectrum-preserving optimizer by orthogonal equivalence transformation.

    Args:
        params: 2-D weights.
        lr: learning rate.
        scaling: `"rms"` normalises the ambient update so its per-element RMS
            is exactly `lr * rms`, erasing the gradient's magnitude entirely;
            `"none"` leaves the generators alone.
        rms: the target RMS. Their 60M script uses `0.2`.
        momentum: `"ambient"` smooths the raw gradient before extracting the
            generators, which is what their 60M script sets; `"lie"` keeps
            separate buffers on the two generators instead, which is the
            variant their only published 60M numbers come from (final loss
            3.3575 bilateral, 3.3654 alternate); `"none"` is this file's
            addition, since they offer no way to switch momentum off.
        beta1, beta2: momentum rates. `beta2 = None` disables the second moment.
        retraction: `"trunc"` for their truncated exponential, `"cayley"` for
            the exact one.
        degree: truncation order, `2` in their script.
        alternate: one side per step, as their script does.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        scaling: str = "rms",
        rms: float = 0.2,
        momentum: str = "ambient",
        beta1: float = 0.9,
        beta2: float | None = 0.95,
        retraction: str = "trunc",
        degree: int = 2,
        alternate: bool = True,
        row_blocks: dict[int, int] | None = None,
    ) -> None:
        if scaling not in ("rms", "none"):
            raise ValueError(f"scaling must be 'rms' or 'none', got {scaling!r}")
        if momentum not in ("ambient", "lie", "none"):
            raise ValueError(f"momentum must be 'ambient', 'lie' or 'none', got {momentum!r}")
        if retraction not in ("trunc", "cayley"):
            raise ValueError(f"retraction must be 'trunc' or 'cayley', got {retraction!r}")
        params = list(params)
        # Their `pion_qkv_split_granularity`, which defaults to `"head"`: a
        # fused QKV parameter is sliced into per-head row-blocks and each block
        # is rotated on its own, with its own RMS scale. This harness keeps Q,
        # K and V as separate matrices, so what is left to mirror is the
        # granularity -- Q is rotated in `heads` blocks of `head_dim x hidden`,
        # not as one square matrix. Keyed by `id(p)` because a parameter is not
        # hashable by value.
        self.row_blocks = dict(row_blocks or {})
        for p in params:
            if p.dim() != 2:
                raise ValueError(f"Pion takes 2-D weights, got shape {tuple(p.shape)}")
        super().__init__(
            params,
            dict(
                lr=lr,
                scaling=scaling,
                rms=rms,
                momentum=momentum,
                beta1=beta1,
                beta2=beta2,
                retraction=retraction,
                degree=degree,
                alternate=alternate,
            ),
        )

    @staticmethod
    def _smooth_lie(g_in, g_out, state: dict, group: dict, prefix: str = ""):
        """Their `lie_lie`: separate buffers on the two generators.

        The buffers stay skew because the generators are, so the smoothed
        result is still a valid Lie-algebra element and needs no projection.
        """
        out = []
        for name, g in (("in", g_in), ("out", g_out)):
            key = f"{prefix}m_{name}"
            if key not in state:
                state[key] = torch.zeros_like(g)
            state[key].mul_(group["beta1"]).add_(g, alpha=1.0 - group["beta1"])
            m = state[key]
            if group["beta2"] is None:
                out.append(m)
                continue
            vkey = f"{prefix}v_{name}"
            if vkey not in state:
                state[vkey] = torch.zeros_like(g)
            state[vkey].mul_(group["beta2"]).add_(m.square(), alpha=1.0 - group["beta2"])
            out.append(m / (state[vkey].sqrt() + 1e-8))
        return out[0], out[1]

    @staticmethod
    def _smooth(g: torch.Tensor, state: dict, group: dict, prefix: str = "") -> torch.Tensor:
        """Their `transported_ambient_ambient`: smoothing happens in ambient space."""
        if group["momentum"] != "ambient":
            return g
        mkey, vkey = f"{prefix}m", f"{prefix}v"
        if mkey not in state:
            state[mkey] = torch.zeros_like(g)
        state[mkey].mul_(group["beta1"]).add_(g, alpha=1.0 - group["beta1"])
        m = state[mkey]
        if group["beta2"] is None:
            return m
        if vkey not in state:
            state[vkey] = torch.zeros_like(g)
        state[vkey].mul_(group["beta2"]).add_(m.square(), alpha=1.0 - group["beta2"])
        return m / (state[vkey].sqrt() + 1e-8)

    @staticmethod
    def _scale(W, g_in, g_out, group, side: str) -> float:
        """`alpha = lr * rms * sqrt(m n) / ||base||_F`, their `_scale_update_matrix_rms`.

        `base` is **the update actually applied**. Their function takes
        `update_side` and builds `p @ A_in`, `A_out @ p` or their sum
        accordingly, and `_effective_update_side` resolves `alternate` to one
        side *before* the scale is computed.

        This used to normalise against the two-sided update always. Under
        `alternate` that calibrates an RMS target of 0.2 against a step twice
        the size of the one taken, so the alternate arm is systematically
        under-stepped: it produced a bilateral-to-alternate gap of 0.0355
        against their published 0.0079, while both levels missed by amounts
        the data differences can account for.

        The learning rate is folded in here, so the caller applies no further
        factor. With `scaling="none"` the same call returns `lr` alone.
        """
        if group["scaling"] == "none":
            return group["lr"]
        m, n = W.shape
        if side == "in":
            base = W @ g_in
        elif side == "out":
            base = g_out @ W
        else:
            base = W @ g_in + g_out @ W
        fro = torch.linalg.matrix_norm(base, "fro")
        return float(group["lr"] * group["rms"] * math.sqrt(m * n) / (fro + 1e-12))

    def _update_block(self, W, G, state: dict, group: dict, prefix: str, side: str):
        """One rotated matrix, returned rather than written.

        A block of a larger parameter is a matrix like any other, so this is
        also what a per-head slice of Q goes through. The momentum buffers are
        keyed by `prefix` so blocks do not share them, while `state["step"]`
        stays per-parameter: their code increments the step once per parameter
        and every block of it therefore alternates in the same phase.
        """
        g = self._smooth(G, state, group, prefix)
        g_in, g_out = generators(W, g)
        if group["momentum"] == "lie":
            g_in, g_out = self._smooth_lie(g_in, g_out, state, group, prefix)
        g_in, g_out = skew(g_in), skew(g_out)
        c = self._scale(W, g_in, g_out, group, side)

        def retract(gen):
            if group["retraction"] == "cayley":
                return cayley(gen, c)
            return _truncated_exp(-c * gen, group["degree"])

        # Only the side being applied is retracted. Building both and using one
        # was harmless but wasteful, and it obscured that the scale and the
        # step have to agree about which side is happening.
        if side == "in":
            return W @ retract(g_in)
        if side == "out":
            return retract(g_out) @ W
        return retract(g_out) @ W @ retract(g_in)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # The baseline retracts too, and the comparison is only fair if both
        # arms get the same arithmetic. On a GPU that means saying so: fp32
        # matrix operations are TF32 by default, which moved the singular
        # values of a weight by a relative 1.0 over 200 steps in measurement.
        with exact_fp32():
            self._step()
        return loss

    @torch.no_grad()
    def _step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state.setdefault("step", 0)
                # Their `_effective_update_side`: `"in" if step % 2 == 1 else
                # "out"`, decided here rather than after the scale, because the
                # scale has to know which step it is normalising.
                if group["alternate"]:
                    side = "in" if state["step"] % 2 else "out"
                else:
                    side = "both"

                blocks = self.row_blocks.get(id(p), 1)
                if blocks > 1:
                    rows = p.shape[0] // blocks
                    if rows * blocks != p.shape[0]:
                        raise ValueError(
                            f"{tuple(p.shape)} does not divide into {blocks} row-blocks"
                        )
                    for b in range(blocks):
                        sl = slice(b * rows, (b + 1) * rows)
                        new = self._update_block(
                            p.detach()[sl], p.grad.detach()[sl], state, group, f"b{b}_", side
                        )
                        p.data[sl].copy_(new)
                else:
                    p.copy_(self._update_block(p.detach(), p.grad.detach(), state, group, "", side))
                state["step"] += 1
