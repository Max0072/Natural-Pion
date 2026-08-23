"""Command-line entry point: one process, one run.

    python -m harness.run --optimizer ngd --lr 1e-3 --seed 0

Every flag maps to a field of `RunConfig`, whose hash names the output
directory, so a sweep is a job array over flag combinations and the results
are keyed by what produced them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace

import torch

from .config import RunConfig
from .train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = RunConfig()
    for f in fields(RunConfig):
        if f.name == "model":
            continue
        kind = {bool: lambda s: s.lower() in ("1", "true", "yes")}.get(f.type, f.type)
        kind = kind if callable(kind) else str
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=getattr(base, f.name))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None, help="cut the run short, for smoke tests")
    args = ap.parse_args()

    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(RunConfig)
        if f.name != "model" and hasattr(args, f.name)
    }
    cfg = replace(base, **overrides)
    print(json.dumps(cfg.manifest()["config"] | {"name": cfg.name}, indent=2, default=str))
    out = train(cfg, device=args.device, max_steps=args.max_steps)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
