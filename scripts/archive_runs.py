"""Copy the small record of every run into the repository, so no number is lost.

`$DATA_p330/runs/` is not in git and holds 180 GB, of which almost all is
checkpoints and per-step diagnostics. What every figure in `docs/` was read from
is two small files per run: `manifest.json` (the full configuration, its hash,
the git commit and the machine) and `log.jsonl` (the loss curve). This copies
exactly those, to `runs-record/<investigation>/<run>/`, and leaves the
checkpoints and `diagnostics.jsonl` on disk (ADR 0012).

Idempotent: a file that is already identical is skipped, so it is safe to run
again after a job finishes -- which it has to be, since a run's `log.jsonl` is
still growing while the job is in flight. Runs whose log changed in the last ten
minutes are reported as in progress, and are copied anyway.

    python scripts/archive_runs.py            # sync
    python scripts/archive_runs.py --dry-run  # say what would change
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import time
from pathlib import Path

KEEP = ("manifest.json", "log.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=os.path.expandvars("$DATA_p330/runs"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs-record"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs, out = Path(args.runs), Path(args.out)
    if not runs.is_dir():
        raise SystemExit(f"{runs} is not a directory (is $DATA_p330 set?)")

    copied = same = missing = in_progress = 0
    total = 0
    now = time.time()
    for manifest in sorted(runs.glob("*/*/manifest.json")):
        run = manifest.parent
        rel = run.relative_to(runs)
        log = run / "log.jsonl"
        if log.exists() and now - log.stat().st_mtime < 600:
            in_progress += 1
            print(f"  in progress (copied anyway, re-run when it ends): {rel}")
        for name in KEEP:
            src = run / name
            if not src.exists():
                missing += 1
                print(f"  missing {name}: {rel}")
                continue
            dst = out / rel / name
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                same += 1
                continue
            copied += 1
            total += src.stat().st_size
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)

    readme = runs / "README.md"
    if readme.exists() and not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readme, out / "RUNS_README.md")

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} files ({total / 1e6:.1f} MB), {same} already identical, "
          f"{missing} missing, {in_progress} runs still in progress")


if __name__ == "__main__":
    main()
