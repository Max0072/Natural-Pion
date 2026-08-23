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
from .linalg import cayley, skew

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
    ) -> None:
        if scaling not in ("rms", "none"):
            raise ValueError(f"scaling must be 'rms' or 'none', got {scaling!r}")
        if momentum not in ("ambient", "lie", "none"):
            raise ValueError(f"momentum must be 'ambient', 'lie' or 'none', got {momentum!r}")
        if retraction not in ("trunc", "cayley"):
            raise ValueError(f"retraction must be 'trunc' or 'cayley', got {retraction!r}")
        params = list(params)
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
    def _smooth_lie(g_in, g_out, state: dict, group: dict):
        """Their `lie_lie`: separate buffers on the two generators.

        The buffers stay skew because the generators are, so the smoothed
        result is still a valid Lie-algebra element and needs no projection.
        """
        out = []
        for name, g in (("in", g_in), ("out", g_out)):
            key = f"m_{name}"
            if key not in state:
                state[key] = torch.zeros_like(g)
            state[key].mul_(group["beta1"]).add_(g, alpha=1.0 - group["beta1"])
            m = state[key]
            if group["beta2"] is None:
                out.append(m)
                continue
            vkey = f"v_{name}"
            if vkey not in state:
                state[vkey] = torch.zeros_like(g)
            state[vkey].mul_(group["beta2"]).add_(m.square(), alpha=1.0 - group["beta2"])
            out.append(m / (state[vkey].sqrt() + 1e-8))
        return out[0], out[1]

    @staticmethod
    def _smooth(g: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """Their `transported_ambient_ambient`: smoothing happens in ambient space."""
        if group["momentum"] != "ambient":
            return g
        if "m" not in state:
            state["m"] = torch.zeros_like(g)
        state["m"].mul_(group["beta1"]).add_(g, alpha=1.0 - group["beta1"])
        m = state["m"]
        if group["beta2"] is None:
            return m
        if "v" not in state:
            state["v"] = torch.zeros_like(g)
        state["v"].mul_(group["beta2"]).add_(m.square(), alpha=1.0 - group["beta2"])
        return m / (state["v"].sqrt() + 1e-8)

    @staticmethod
    def _scale(W, g_in, g_out, group) -> float:
        """`alpha = lr * rms * sqrt(m n) / ||base||_F`, their `_scale_update_matrix_rms`.

        The learning rate is folded in here, so the caller applies no further
        factor. With `scaling="none"` the same call returns `lr` alone.
        """
        if group["scaling"] == "none":
            return group["lr"]
        m, n = W.shape
        base = W @ g_in + g_out @ W
        fro = torch.linalg.matrix_norm(base, "fro")
        return float(group["lr"] * group["rms"] * math.sqrt(m * n) / (fro + 1e-12))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state.setdefault("step", 0)
                W = p.detach()
                g = self._smooth(p.grad.detach(), state, group)
                g_in, g_out = generators(W, g)
                if group["momentum"] == "lie":
                    g_in, g_out = self._smooth_lie(g_in, g_out, state, group)
                g_in, g_out = skew(g_in), skew(g_out)
                c = self._scale(W, g_in, g_out, group)

                if group["retraction"] == "cayley":
                    left = cayley(g_out, c)
                    right = cayley(g_in, c)
                else:
                    left = _truncated_exp(-c * g_out, group["degree"])
                    right = _truncated_exp(-c * g_in, group["degree"])

                if group["alternate"]:
                    W = W @ right if state["step"] % 2 else left @ W
                else:
                    W = left @ W @ right
                p.copy_(W)
                state["step"] += 1
        return loss
