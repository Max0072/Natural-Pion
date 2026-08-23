"""Convenience adapter: feed `nn.Linear` activations to the optimizer.

Deliberately thin and deliberately optional. Other frameworks name their
layers differently -- Megatron's parallel linears, fused QKV projections --
and each needs its own adapter of about this size. Keeping it out of
`optimizer` is what lets those exist without touching the optimizer at all.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch.nn as nn

__all__ = ["attach", "attached"]


def attach(modules, optimizer) -> list:
    """Register forward pre-hooks recording each layer's input covariance.

    Returns the handles; call `.remove()` on each when done, or use `attached`.
    """
    handles = []
    for module in modules:
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(module).__name__}")

        def hook(mod, inputs, _opt=optimizer):
            _opt.observe(mod.weight, inputs[0])
            return None

        handles.append(module.register_forward_pre_hook(hook))
    return handles


@contextmanager
def attached(modules, optimizer):
    """`attach` as a context manager, removing the hooks on exit."""
    handles = attach(modules, optimizer)
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()
