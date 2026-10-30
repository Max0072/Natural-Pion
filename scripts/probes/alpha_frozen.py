"""Alpha switched off, its measured profile played back. (journal 2026-10-30.)

Runs with `--ngd-trust none` (live alpha = 1) and multiplies the rotational rate
by a table made from OTHER seeds (alpha_frozen_profile.py). Only the rotational
rate moves: `warmup_steps` is not used, because `lr_at` would warm AdamW too. The
parent's step is unchanged; it gets a copy of the group with `lr` scaled.

    ALPHA_FROZEN_MODE = ones | time | layer | both
    ALPHA_FROZEN_TABLE = path of the json
"""

from __future__ import annotations

import json
import os
import sys

import torch

import harness.train as train
from harness.run import main as run_main
from ngd_pion.unified import NGDPionUnified

MODE = os.environ["ALPHA_FROZEN_MODE"]
TABLE = json.load(open(os.environ["ALPHA_FROZEN_TABLE"]))


def multiplier(i: int, t: int) -> float:
    b = min(t // TABLE["bin"], len(TABLE["time"]) - 1)
    if MODE == "ones":
        return 1.0
    if MODE == "time":
        return TABLE["time"][str(b)]
    if MODE == "layer":
        return TABLE["layer"][str(i)]
    if MODE == "both":
        return TABLE["both"][f"{i}:{b}"]
    raise SystemExit(f"unknown mode {MODE}")


class AlphaFrozen(NGDPionUnified):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._index = {id(p): i for g in self.param_groups for i, p in enumerate(g["params"])}

    def _apply(self, p, group):
        m = multiplier(self._index[id(p)], self.state[p]["step"])
        super()._apply(p, {**group, "lr": group["lr"] * m})


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    train.NGD_IMPLEMENTATIONS["ngd-pion-s"] = AlphaFrozen
    run_main()
