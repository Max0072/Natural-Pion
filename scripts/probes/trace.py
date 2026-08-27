import json, glob, os, sys
for label, pat in (("OLD 23edea9", "/onyx/data/p330/runs/ctl-old/ngd-pion-lr0.003-*"),
                   ("NEW HEAD  ", "/onyx/data/p330/runs/ctl-new/ngd-pion-lr0.003-*")):
    for run in sorted(glob.glob(pat)):
        f = os.path.join(run, "log.jsonl")
        if not os.path.exists(f):
            print(f"{label}: no log"); continue
        prev, out = None, []
        for line in open(f):
            r = json.loads(line)
            if "train_loss" not in r: continue
            w = r["wall"]
            sps = (w - prev[1]) / (r["step"] - prev[0]) if prev else float("nan")
            prev = (r["step"], w)
            out.append((r["step"], w, sps, r.get("tokens_per_sec_window"), r["train_loss"]))
        print(f"\n=== {label} ===")
        for s, w, sps, t, l in out:
            print(f"  step {s:>4}  wall {w:8.1f}  s/step {sps:7.3f}  tok/s {t}  loss {l:.4f}")
