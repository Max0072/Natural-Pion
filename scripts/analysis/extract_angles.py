"""Archive the per-matrix rotation angle of a run into runs-record, from the run's diagnostics.jsonl.

    python scripts/analysis/extract_angles.py <family/run-dir> ...      (relative to $DATA_p330/runs)

Writes runs-record/<family/run-dir>/angles.json: {step: [angle of each logged matrix, in log order]}. The
diagnostics file itself (5 MB) stays with the raw run; this is the part the paper uses.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for rel in sys.argv[1:]:
    src = Path(os.environ.get("DATA_p330", "/onyx/data/p330")) / "runs" / rel / "diagnostics.jsonl"
    by = defaultdict(list)
    for line in open(src):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "angle" in r and "step" in r:
            by[int(r["step"])].append(r["angle"])
    dst = ROOT / "runs-record" / rel / "angles.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump({str(k): v for k, v in sorted(by.items())}, open(dst, "w"))
    print(dst, len(by), "steps")
