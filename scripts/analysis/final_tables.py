"""Regenerate the tables of docs/paper/FINAL.md from runs-record/, so no number in the paper is typed by hand.

    python scripts/analysis/final_tables.py [runs-record] > /tmp/tables.md

Each cell reads the final `val_loss` (and `wall`) of one run's `log.jsonl`, taking the last attempt after the
last `start` event. A run still in progress is marked with `*` and its last val.
"""
from __future__ import annotations

import glob
import json
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "runs-record"


def run(pattern: str, seed: int | None = None) -> dict | None:
    """The single run matching `pattern` (and `seed`); None if absent."""
    hits = []
    for m in sorted(glob.glob(f"{R}/{pattern}/manifest.json")):
        cfg = json.load(open(m))["config"]
        if seed is None or cfg["seed"] == seed:
            hits.append(m[: -len("/manifest.json")])
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(f"{pattern} seed {seed}: {len(hits)} runs, expected one: {hits}")
    d = hits[0]
    cfg = json.load(open(f"{d}/manifest.json"))["config"]
    rows = []
    for line in open(f"{d}/log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        rows.append(r)
    vals = [r for r in rows if "val_loss" in r]
    walls = [r["wall"] for r in rows if "wall" in r]
    done = bool(vals) and vals[-1]["step"] >= cfg["train_steps"] - 1
    return {"val": vals[-1]["val_loss"] if vals else None, "step": vals[-1]["step"] if vals else None,
            "wall": walls[-1] if walls else None, "steps": cfg["train_steps"], "done": done, "dir": d}


def v(r: dict | None) -> str:
    if r is None or r["val"] is None:
        return "-"
    return f"{r['val']:.4f}" + ("" if r["done"] else f"* (step {r['step']})")


def gap(a, b) -> str:
    if not a or not b or a["val"] is None or b["val"] is None or not (a["done"] and b["done"]):
        return "-"
    return f"{a['val'] - b['val']:+.3f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


# ---- 1: Pion lr ----------------------------------------------------------------------------------
print("### 1. Pion lr\n")
L = "stepnormlong"
table(["Pion steps", "5e-4", "1e-3", "2e-3", "4e-3"], [
    ["3 840", v(run(f"{L}/pion3840-lr5e-4/*")), v(run("stepnormfinal/pion3840/*")),
     v(run(f"{L}/pion3840-lr2e-3/*")), v(run(f"{L}/pion3840-lr4e-3/*"))],
    ["19 200", "-", v(run("stepnormfinal/pion19200/*")), v(run(f"{L}/pion19200-lr2e-3/*")),
     v(run(f"{L}/pion19200-lr4e-3/*"))],
])

# ---- 5.1: full length ----------------------------------------------------------------------------
print("### 5.1 Full length, seed 0\n")
pion = run(f"{L}/pion73242-lr1e-3/*")
rows = []
for name, pat in [("Pion, lr 1e-3", f"{L}/pion73242-lr1e-3/*"),
                  ("ours, theta 0.05", f"{L}/ours56900-lr0.05/*"),
                  ("ours, theta 0.025", f"{L}/ours56900-lr0.025/*"),
                  ("ours, theta 0.05", f"{L}/ours73242-lr0.05/*")]:
    r = run(pat)
    rows.append([name, r["steps"] if r else "-", f"{r['wall']:.0f} s" if r and r["wall"] else "-", v(r),
                 gap(r, pion) if r and r["dir"] != pion["dir"] else "-"])
table(["run", "steps", "wall", "val", "vs Pion"], rows)

# ---- 5.2: same wall time, seeds ------------------------------------------------------------------
print("### 5.2 Same wall time, seeds 0-3 (gap = Pion - ours)\n")
budgets = [
    ("~1 800 s", lambda s: run("stepnormfinal/pion3840/*") if s == 0 else run(f"{L}/pion3840-lr1e-3/*", s),
                 lambda s: run("stepnormcombo/batch/*") if s == 0 else run("stepnormfinal/tf25/*", s)),
    ("~8 900 s", lambda s: run("stepnormfinal/pion19200/*") if s == 0 else run(f"{L}/pion19200-lr1e-3/*", s),
                 lambda s: run("stepnormfinal/tf25-15k-0.05/*") if s == 0 else run(f"{L}/ours15000-lr0.05/*", s)),
]
rows = []
for name, P, O in budgets:
    gaps = []
    cells = []
    for s in range(4):
        p, o = P(s), O(s)
        g = p["val"] - o["val"] if p and o and p["val"] and o["val"] else None
        gaps.append(g)
        cells.append(f"{g:.3f}" if g is not None else "-")
    m = mean(gaps)
    rows.append([name] + cells + [f"{m:.3f}" if m is not None else "-"])
table(["budget", "seed 0", "seed 1", "seed 2", "seed 3", "mean"], rows)
for name, P, O in budgets:
    print(f"* {name}: Pion " + " / ".join(v(P(s)) for s in range(4)) + "; ours " + " / ".join(v(O(s)) for s in range(4)))
print()

# ---- 5.3: same steps -----------------------------------------------------------------------------
print("### 5.3 Same number of steps\n")
p3000 = [run("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [run("seedshead/pion-lr0.001-*", s) for s in (1, 2, 3)]
o3000 = [run("stepnormcombo/batch/*")] + [run("stepnormfinal/tf25/*", s) for s in (1, 2, 3)]
mp, mo = mean([r["val"] for r in p3000 if r]), mean([r["val"] for r in o3000 if r])
p15 = run("ladder15k/pion-lr0.001-s0-f67fdad3e9")
o15 = run("stepnormfinal/tf25-15k-0.05/*")
o73 = run(f"{L}/ours73242-lr0.05/*")
table(["steps", "Pion", "ours", "gap"], [
    ["3 000 (seeds mean)", f"{mp:.4f}", f"{mo:.4f}", f"{mp - mo:.3f}"],
    ["15 000", v(p15), v(o15), gap(p15, o15)],
    ["73 242", v(pion), v(o73), gap(pion, o73)],
])

# ---- 6.5: initial spectrum -----------------------------------------------------------------------
print("### 6.5 Initial spectrum (3000 steps, theta 0.2, seed 0)\n")
I, IL = "stepnorminit", "stepnorminitlong"
base = run("stepnormcombo/batch/*")
rows = [["normal(0, 0.02)", v(base), "-"]]
for name, pat in [("tail alpha 2", f"{I}/pl2/*"), ("tail alpha 2.5", f"{IL}/ours3000-lr0.2-pl2.5/*"),
                  ("tail alpha 3", f"{I}/pl3/*"), ("tail alpha 4", f"{IL}/ours3000-lr0.2-pl4/*"),
                  ("tail alpha 5", f"{I}/pl5/*"), ("flat", f"{I}/flat/*"), ("orthogonal", f"{I}/orth/*")]:
    r = run(pat)
    rows.append([name, v(r), gap(r, base)])
table(["initial spectrum", "val", "d vs normal"], rows)
seeds = [(run("stepnorminit/pl3/*"), base)] + [(run(f"{IL}/ours3000-lr0.2-pl3/*", s), run("stepnormfinal/tf25/*", s)) for s in (1, 2, 3)]
table(["seed", "normal", "alpha 3", "gain"],
      [[s, v(b), v(a), f"{b['val'] - a['val']:.3f}" if a and b else "-"] for s, (a, b) in enumerate(seeds)])
table(["arm", "steps", "normal", "alpha 3", "gain"], [
    ["ours", 3000, v(base), v(run(f"{I}/pl3/*")), gap(base, run(f"{I}/pl3/*"))],
    ["Pion", 3000, v(p3000[0]), v(run(f"{IL}/pion3000-lr1e-3-pl3/*")), gap(p3000[0], run(f"{IL}/pion3000-lr1e-3-pl3/*"))],
    ["Pion", 3840, v(run("stepnormfinal/pion3840/*")), v(run(f"{IL}/pion3840-lr1e-3-pl3/*")),
     gap(run("stepnormfinal/pion3840/*"), run(f"{IL}/pion3840-lr1e-3-pl3/*"))],
    ["ours", 15000, v(o15), v(run(f"{IL}/ours15000-lr0.05-pl3/*")), gap(o15, run(f"{IL}/ours15000-lr0.05-pl3/*"))],
    ["Pion", 15000, v(p15), v(run(f"{IL}/pion15000-lr1e-3-pl3/*")), gap(p15, run(f"{IL}/pion15000-lr1e-3-pl3/*"))],
    ["Pion", 19200, v(run("stepnormfinal/pion19200/*")), v(run(f"{IL}/pion19200-lr1e-3-pl3/*")),
     gap(run("stepnormfinal/pion19200/*"), run(f"{IL}/pion19200-lr1e-3-pl3/*"))],
    ["ours", 56900, v(run(f"{L}/ours56900-lr0.05/*")), v(run(f"{IL}/ours56900-lr0.05-pl3/*")),
     gap(run(f"{L}/ours56900-lr0.05/*"), run(f"{IL}/ours56900-lr0.05-pl3/*"))],
    ["Pion", 73242, v(pion), v(run(f"{IL}/pion73242-lr1e-3-pl3/*")), gap(pion, run(f"{IL}/pion73242-lr1e-3-pl3/*"))],
])
