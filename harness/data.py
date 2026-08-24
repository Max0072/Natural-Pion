"""Token stream for pretraining.

The corpus is one flat array of token ids on disk, read through a memmap, and
batches are random windows into it. Nothing here knows about C4 specifically;
`scripts/prepare_data.py` is what turns a corpus into this format.

Megatron uses its own indexed binary format with document boundaries and a
sampling index. A flat array loses document boundaries -- a window can span
two documents -- which is what nanoGPT-style training does and is harmless at
this scale. It is worth remembering as a difference if the anchor run fails to
reproduce their number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["TokenCorpus"]


class TokenCorpus:
    """A memmapped array of token ids, sampled as random windows.

    Args:
        path: `.bin` file of little-endian uint16 token ids.
        seq_len: window length. Batches carry `seq_len + 1` tokens internally
            so inputs and targets are one shifted copy of the same window.
        seed: fixes the window sequence. Two runs with the same seed see the
            same batches in the same order, which is the point.
    """

    def __init__(self, path: str | Path, seq_len: int, seed: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run scripts/prepare_data.py to build it"
            )
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < seq_len + 2:
            raise ValueError(f"corpus has {len(self.tokens)} tokens, need at least {seq_len + 2}")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def epochs_for(self) -> float:
        """How many passes over the corpus a token budget implies."""
        return float(len(self.tokens))

    def batch(self, size: int, device: str | torch.device = "cpu"):
        """One batch of `(inputs, targets)`, each `(size, seq_len)`."""
        hi = len(self.tokens) - self.seq_len - 1
        starts = self.rng.integers(0, hi, size=size)
        window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        chunk = torch.from_numpy(window)
        inputs = chunk[:, :-1].to(device, non_blocking=True)
        targets = chunk[:, 1:].to(device, non_blocking=True)
        return inputs, targets

    @property
    def rng_state(self) -> dict:
        """The sampler's position, so a resumed run does not replay batches."""
        return self.rng.bit_generator.state

    @rng_state.setter
    def rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

    def fixed_batches(self, size: int, count: int, seed: int, device="cpu") -> list:
        """A held-out set that does not move between evaluations."""
        rng = np.random.default_rng(seed)
        hi = len(self.tokens) - self.seq_len - 1
        out = []
        for _ in range(count):
            starts = rng.integers(0, hi, size=size)
            window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
            chunk = torch.from_numpy(window)
            out.append((chunk[:, :-1].to(device), chunk[:, 1:].to(device)))
        return out
