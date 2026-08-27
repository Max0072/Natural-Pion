"""Read the two floor-probe runs and answer the three open questions.

1. Does `floor_share` go to 1 exactly where `quad_over_curv` blows up?
   That is the direct test that the floor, not a missing Jacobian, is what
   makes `quad` and `curv` disagree.
2. What are `quad` and `curv` raw, so we can see which of the two degenerates
   and by how much, rather than only their ratio.
3. Does `angle` still fall to exactly zero now that `spectral_norm` recovers
   instead of latching? If it does not, the zeros were the absorbing state.
"""
import json, glob, os, statistics as st

def rows(pattern):
    out = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, "diagnostics.jsonl")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # The cancelled jobs 252299/252302 left step-1 rows in this same file --
    # it is opened in append mode -- so collapse duplicates by (layer, step),
    # keeping the last written.
    seen = {}
    for r in out:
        seen[(r.get("name"), r.get("step"))] = r
    return list(seen.values())

def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")

for tag, pat in (("eta 3e-3 (working)", "/onyx/data/p330/runs/floor2/ngd-pion-lr0.003-*"),
                 ("eta 1.0  (broken)",  "/onyx/data/p330/runs/floor2/ngd-pion-lr1-*")):
    rs = rows(pat)
    print(f"\n{'='*100}\n===== {tag} ===== {len(rs)} rows")
    if not rs:
        print("  no diagnostics found"); continue

    steps = sorted({r["step"] for r in rs})
    print(f"  steps {steps[0]}..{steps[-1]}")

    print(f"\n  by depth (median over the second half)")
    print(f"  {'depth':>5} {'quad':>11} {'curv':>11} {'q/c':>10} {'fs_in':>7} {'fs_out':>7}"
          f" {'null_frac':>9} {'lam_min':>10} {'alpha':>7} {'angle':>9}")
    half = steps[len(steps) // 2]
    for d in sorted({r["depth"] for r in rs}):
        sel = [r for r in rs if r["depth"] == d and r["step"] >= half]
        if not sel:
            continue
        print(f"  {d:>5} {med([r.get('quad') for r in sel]):11.4e}"
              f" {med([r.get('curv') for r in sel]):11.4e}"
              f" {med([r.get('quad_over_curv') for r in sel]):10.3e}"
              f" {med([r.get('floor_share_in') for r in sel]):7.4f}"
              f" {med([r.get('floor_share_out') for r in sel]):7.4f}"
              f" {med([r.get('null_frac') for r in sel]):9.4f}"
              f" {med([r.get('lam_min_A') for r in sel]):10.3e}"
              f" {med([r.get('alpha') for r in sel]):7.4f}"
              f" {med([r.get('angle') for r in sel]):9.3e}")

    print(f"\n  over time (across layers)")
    print(f"  {'step':>6} {'q/c med':>10} {'fs_in med':>10} {'fs_out med':>10}"
          f" {'angle max':>10} {'angle min':>10} {'#angle==0':>10} {'curv min':>11}")
    for s in steps:
        sel = [r for r in rs if r["step"] == s]
        zeros = sum(1 for r in sel if r.get("angle") == 0.0)
        print(f"  {s:>6} {med([r.get('quad_over_curv') for r in sel]):10.3e}"
              f" {med([r.get('floor_share_in') for r in sel]):10.4f}"
              f" {med([r.get('floor_share_out') for r in sel]):10.4f}"
              f" {max([r.get('angle', 0) or 0 for r in sel]):10.3e}"
              f" {min([r.get('angle', 0) or 0 for r in sel]):10.3e}"
              f" {zeros:>4}/{len(sel):<5}"
              f" {min([r.get('curv', float('inf')) for r in sel]):11.3e}")
