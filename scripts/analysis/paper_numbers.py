"""Every measured number in docs/paper/preprint.md, computed from runs-record/ (or a runnable check).

    python scripts/analysis/paper_numbers.py            # writes docs/paper/numbers.json and docs/paper/numbers.md
    python scripts/analysis/render_paper.py             # fills preprint.md -> preprint.filled.md

The preprint refers to numbers only as {{key}}. Each key below says where its value comes from. A key
whose source is not yet in the repository is listed as MISSING with what is needed, and renders as a
visible gap rather than a number.

Conventions:
* `val` is the final validation loss of a run (last evaluation of the last attempt in its log.jsonl).
* "best of a grid" is the lowest final val over the runs matching a pattern (one seed, different angles).
* seconds per step = final wall time / steps, so it includes the run's periodic evaluations. Runs compared
  for speed in the paper share the same evaluation cadence.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs-record"
NUM, NOTE = {}, {}


# ---------------------------------------------------------------------------------------------- reading
def _rows(d: Path) -> list[dict]:
    rows = []
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        rows.append(r)
    return rows


def runs(pattern: str, seed: int | None = None, **cfg_eq) -> list[Path]:
    out = []
    for m in sorted(R.glob(f"{pattern}/manifest.json")):
        cfg = json.load(open(m))["config"]
        if seed is not None and cfg["seed"] != seed:
            continue
        if any(cfg.get(k) != v for k, v in cfg_eq.items()):
            continue
        out.append(m.parent)
    return out


def one(pattern: str, seed: int | None = None, **cfg_eq) -> Path:
    hits = runs(pattern, seed, **cfg_eq)
    if len(hits) != 1:
        raise SystemExit(f"{pattern} (seed {seed}, {cfg_eq}): {len(hits)} runs, expected one: {hits}")
    return hits[0]


def info(d: Path) -> dict:
    cfg = json.load(open(d / "manifest.json"))["config"]
    rows = _rows(d)
    vals = [r for r in rows if "val_loss" in r]
    walls = [r["wall"] for r in rows if "wall" in r]
    done = bool(vals) and vals[-1]["step"] >= cfg["train_steps"] - 1
    return {"val": vals[-1]["val_loss"] if vals else None, "steps": cfg["train_steps"], "lr": cfg["lr"],
            "wall": walls[-1] if walls else None, "done": done, "vals": {r["step"]: r["val_loss"] for r in vals},
            "rows": rows, "dir": d}


def val_at(d: Path, step: int) -> float:
    """Validation loss at `step`, from whichever attempt in the log recorded it (latest wins)."""
    found = None
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step") == step and "val_loss" in r:
            found = r["val_loss"]
    if found is None:
        raise SystemExit(f"{d}: no val at step {step}")
    return found


def val(pattern, seed=None, **kw) -> float:
    i = info(one(pattern, seed, **kw))
    if not i["done"]:
        raise SystemExit(f"{pattern}: not finished")
    return i["val"]


def sps(pattern, seed=None, **kw) -> float:
    """Wall seconds per step: last logged wall over last logged step (+1), so runs cut short count right."""
    rows = [r for r in _rows(one(pattern, seed, **kw)) if "wall" in r]
    return rows[-1]["wall"] / (rows[-1]["step"] + 1)


def sps_median(d: Path) -> float:
    """Median seconds per step between consecutive logged rows, over every attempt (robust to resumes)."""
    rates, prev = [], None
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            prev = None
            continue
        if "wall" in r:
            if prev and r["step"] > prev["step"] and r["wall"] > prev["wall"]:
                rates.append((r["wall"] - prev["wall"]) / (r["step"] - prev["step"]))
            prev = r
    rates.sort()
    return rates[len(rates) // 2]


def best(pattern, at: int | None = None, **kw) -> tuple[float, float, bool]:
    """(best final val, its lr, interior?) over a finished grid; with `at`, val at that step instead
    (for runs stopped early with --max-steps)."""
    infos = [info(d) for d in runs(pattern, **kw)]
    if at is not None:
        for i in infos:
            i["val"], i["done"] = val_at(i["dir"], at), True
    infos = [i for i in infos if i["done"]]
    if not infos:
        raise SystemExit(f"{pattern}: no finished runs")
    infos.sort(key=lambda i: i["lr"])
    k = min(range(len(infos)), key=lambda j: infos[j]["val"])
    return infos[k]["val"], infos[k]["lr"], 0 < k < len(infos) - 1


def put(key: str, value, source: str):
    NUM[key] = value
    NOTE[key] = source


def missing(key: str, need: str):
    NUM[key] = None
    NOTE[key] = "MISSING: " + need


# ---------------------------------------------------------------------------------------------- keys
L = "stepnormlong"

# headline, full length
pf, ncf, nsf = info(one(f"{L}/pion73242-lr1e-3/*")), info(one(f"{L}/ours56900-lr0.05/*")), info(one(f"{L}/ours73242-lr0.05/*"))
put("pion_full", pf["val"], f"{L}/pion73242-lr1e-3")
put("pion_full_wall", pf["wall"], f"{L}/pion73242-lr1e-3, wall")
put("np_full_clock", ncf["val"], f"{L}/ours56900-lr0.05")
put("np_full_clock_wall", ncf["wall"], f"{L}/ours56900-lr0.05, wall")
put("np_full_clock_0025", val(f"{L}/ours56900-lr0.025/*"), f"{L}/ours56900-lr0.025")
put("np_full_steps", nsf["val"], f"{L}/ours73242-lr0.05")
put("np_full_steps_wall", nsf["wall"], f"{L}/ours73242-lr0.05, wall")
put("gap_full_clock", pf["val"] - ncf["val"], "pion_full - np_full_clock")
put("gap_full_steps", pf["val"] - nsf["val"], "pion_full - np_full_steps")
put("pion_sps", pf["wall"] / pf["steps"], "pion_full wall / steps")
put("np_sps", ncf["wall"] / ncf["steps"], "np_full_clock wall / steps")
put("cost_ratio", NUM["np_sps"] / NUM["pion_sps"], "np_sps / pion_sps")

# Pion lr grid
for steps, lrs in ((3840, ("5e-4", "1e-3", "2e-3", "4e-3")), (19200, ("1e-3", "2e-3", "4e-3"))):
    for lr in lrs:
        pat = f"stepnormfinal/pion{steps}/*" if lr == "1e-3" else f"{L}/pion{steps}-lr{lr}/*"
        put(f"pion_lr_{steps}_{lr}", val(pat, seed=0), pat)

put("pion_lr_3840_better", NUM["pion_lr_3840_1e-3"] - NUM["pion_lr_3840_2e-3"], "Pion 3840: lr 1e-3 - lr 2e-3")

# equal wall time, seeds
P1800 = lambda s: ("stepnormfinal/pion3840/*", 0) if s == 0 else (f"{L}/pion3840-lr1e-3/*", s)
O1800 = lambda s: ("stepnormcombo/batch/*", 0) if s == 0 else ("stepnormfinal/tf25/*", s)
P8900 = lambda s: ("stepnormfinal/pion19200/*", 0) if s == 0 else (f"{L}/pion19200-lr1e-3/*", s)
O8900 = lambda s: ("stepnormfinal/tf25-15k-0.05/*", 0) if s == 0 else (f"{L}/ours15000-lr0.05/*", s)
for tag, P, O in (("1800", P1800, O1800), ("8900", P8900, O8900)):
    gaps = []
    for s in range(4):
        g = val(*P(s)) - val(*O(s))
        gaps.append(g)
        put(f"gap_{tag}_s{s}", g, f"{P(s)[0]} seed {s} - {O(s)[0]} seed {s}")
    put(f"gap_{tag}_mean", sum(gaps) / 4, f"mean of gap_{tag}_s0..3")
put("gap_1800_vs_pion2e3", val(f"{L}/pion3840-lr2e-3/*", 0) - val("stepnormcombo/batch/*", 0), "Pion 3840 lr 2e-3 - ours 3000, seed 0")

# equal steps
p3000 = [val("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [val("seedshead/pion-lr0.001-*", s) for s in (1, 2, 3)]
o3000 = [val(*O1800(s)) for s in range(4)]
put("pion3000_s0", p3000[0], "adamw2d/pion-lr0.001-s0-9e6e6d32f6 (Pion, 3000 steps, AdamW 8e-3)")
put("pion3000_mean", sum(p3000) / 4, "Pion 3000, seeds 0-3 (adamw2d, seedshead)")
put("np3000_mean", sum(o3000) / 4, "Natural Pion Fast 3000 @0.2, seeds 0-3")
put("gap3000", NUM["pion3000_mean"] - NUM["np3000_mean"], "pion3000_mean - np3000_mean")
put("pion15k", val("ladder15k/pion-lr0.001-s0-f67fdad3e9"), "ladder15k/pion-lr0.001-s0-f67fdad3e9")
put("np15k", val("stepnormfinal/tf25-15k-0.05/*"), "stepnormfinal/tf25-15k-0.05")
put("gap15k", NUM["pion15k"] - NUM["np15k"], "pion15k - np15k")

# steps to Pion's final val (first crossing inside one of our runs; seed 0)
ours_curves = [info(one(p)) for p in ("stepnormcombo/batch/*", "stepnormfinal/tf25-15k-0.05/*", f"{L}/ours56900-lr0.05/*", f"{L}/ours73242-lr0.05/*")]
ratios = []
for steps, pat in ((3840, "stepnormfinal/pion3840/*"), (4500, "stepnormfinal/pion4500/*"), (19200, "stepnormfinal/pion19200/*"), (73242, f"{L}/pion73242-lr1e-3/*")):
    target = val(pat, 0)
    cross = None
    for c in ours_curves:
        pts = sorted((s + 1, v) for s, v in c["vals"].items())
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v1 <= target < v0:
                s = s0 + (v0 - target) / (v0 - v1) * (s1 - s0)
                cross = s if cross is None else min(cross, s)
                break
    put(f"steps_pion_{steps}_val", target, pat)
    put(f"steps_cross_{steps}", cross, f"first step any of our seed-0 runs reaches val(Pion {steps}), linear between evaluations")
    put(f"steps_ratio_{steps}", steps / cross, f"{steps} / steps_cross_{steps}")
    ratios.append(steps / cross)
put("steps_ratio_min", min(ratios), "min of steps_ratio_*")

# ablations: direction
for key, pat in (("dir_fisher_nomom", "stepnorm/*"), ("dir_raw_nomom", "stepnormabl/none-cosine/*"),
                 ("dir_fisher_mom", "stepnormmom/fisher/*"), ("dir_raw_mom", "stepnormmom/none/*")):
    v, lr, interior = best(pat)
    put(key, v, f"best of {pat} (angle {lr}{'' if interior else ', EDGE'})")
put("gain_fisher_nomom", NUM["dir_raw_nomom"] - NUM["dir_fisher_nomom"], "dir_raw_nomom - dir_fisher_nomom")
put("gain_fisher_mom", NUM["dir_raw_mom"] - NUM["dir_fisher_mom"], "dir_raw_mom - dir_fisher_mom")
put("gain_angle_on_raw", NUM["pion3000_s0"] - NUM["dir_raw_mom"], "pion3000_s0 - dir_raw_mom")

# ablations: length rules (no momentum, t_fac 1)
for key, pat, kw in (("len_rms", "stepnormnorm/rms/*", {}), ("len_fishernorm", "stepnormnorm/fisher/*", {}),
                     ("len_alpha", "tfac1alpha/*", {}), ("len_fixed", "tfac1/*", {"ngd_t_fac": 1}),
                     ("len_before", "stepnormpre/*", {})):
    v, lr, interior = best(pat, **kw)
    put(key, v, f"best of {pat} {kw} (lr {lr}{'' if interior else ', EDGE'})")
    NUM[key + "_interior"] = interior
    NOTE[key + "_interior"] = f"is the best of {pat} inside its grid"
rms_seed = [val("stepnormnorm/rms/*", 0, lr=0.0024)] + [val("stepnormrmsseeds/*", s) for s in (1, 2, 3)]
spec_seed = [val("stepnorm/*", 0, lr=0.4)] + [val("stepnormseeds/*", s) for s in (1, 2, 3)]
for s in range(4):
    put(f"rms_gap_s{s}", rms_seed[s] - spec_seed[s], f"RMS (lr 2.4e-3) - spectral (angle 0.4), seed {s}")
for k in ("spectral", "rms", "alpha", "fishernorm", "fixed", "before"):
    missing(f"angle_spread_{k}", "p90/p10 of the realised angle across the 56 matrices after step 1000; needs the "
            "runs' diagnostics.jsonl archived and a script reading it")
NUM["angle_spread_fixed_c"] = None
NOTE["angle_spread_fixed_c"] = NOTE["angle_spread_fixed"]

# ablations: other choices
put("ref_tf25", val("stepnormcombo/cos0.2/*"), "stepnormcombo/cos0.2 (t_fac 25, momentum, A and D last 6)")
put("sides_best", min(val("stepnormcombo/sides0.1/*"), val("stepnormcombo/sides0.2/*")), "best of stepnormcombo/sides0.1, sides0.2")
for m in ("0.25", "0.5", "2", "4"):
    put(f"attn_{m}", val(f"stepnormattn/m{m}/*"), f"stepnormattn/m{m}")
for k in ("constant", "linear", "warmcos"):
    v, lr, interior = best(f"stepnormabl/fisher-{k}/*")
    put(f"sched_{k}", v, f"best of stepnormabl/fisher-{k} (angle {lr}{'' if interior else ', EDGE'})")
for d in ("0.2", "0.1"):
    put(f"wsd_{d}", val(f"stepnormwsd/decay{d}/*"), f"stepnormwsd/decay{d}")
put("polar_2499", val_at(one("stepnormpower/p0/*", lr=0.01), 2499), "stepnormpower/p0 angle 0.01, val at step 2499")
put("polar_ref_2499", val_at(one("stepnormmom/fisher/*", lr=0.1), 2499), "stepnormmom/fisher angle 0.1, val at step 2499")

# chapter 4: profile
for t in ("tfac1", "tfac25"):
    pr = json.load(open(R / "stepnormprof" / t / "profile.json"))
    n = pr["_step"][1]
    per = lambda k: pr[k][0] / n
    put(f"prof_{t}_refactor", per("_refactor"), f"stepnormprof/{t}/profile.json, _refactor / steps")
    put(f"prof_{t}_angle", per("_measure_angle"), f"stepnormprof/{t}/profile.json")
    put(f"prof_{t}_retract", per("_retract"), f"stepnormprof/{t}/profile.json")
    put(f"prof_{t}_rest", per("_apply") - per("_measure_angle") - per("_retract"), f"stepnormprof/{t}/profile.json, _apply - angle - retract")
    put(f"prof_{t}_stats", per("observe") + per("observe_backward"), f"stepnormprof/{t}/profile.json, observe + observe_backward")
    put(f"prof_{t}_step", sps(f"stepnormprof/{t}/*"), f"stepnormprof/{t} run, wall / steps (profiled run)")

# chapter 4: refresh cadence (3000 steps, seed 0; best angle of each grid)
put("tf1_nomom", NUM["dir_fisher_nomom"], "= dir_fisher_nomom")
put("tf1_nomom_sps", sps("stepnorm/*", lr=0.4), "stepnorm angle 0.4, wall / steps")
v, lr, _ = best("stepnormtfac/*"); put("tf25_nomom", v, "best of stepnormtfac"); put("tf25_nomom_sps", sps("stepnormtfac/*", lr=lr), "stepnormtfac best angle")
put("tf1_mom", NUM["dir_fisher_mom"], "= dir_fisher_mom")
put("tf1_mom_sps", sps("stepnormmom/fisher/*", lr=0.1), "stepnormmom/fisher angle 0.1")
put("tf10_mom", val("stepnormcombo/tf10-0.2/*"), "stepnormcombo/tf10-0.2")
put("tf10_mom_sps", sps("stepnormcombo/tf10-0.2/*"), "stepnormcombo/tf10-0.2")
v, lr, _ = best("stepnormtfacmom/*"); put("tf25_mom", v, "best of stepnormtfacmom"); put("tf25_mom_sps", sps("stepnormtfacmom/*", lr=lr), "stepnormtfacmom best angle")
tf1s = [val("stepnormmom/fisher/*", lr=0.1)] + [val("stepnormmomseeds/*", s) for s in (1, 2, 3)]
tf10s = [val("stepnormfinal/tf10/*", s) for s in (0, 1, 2, 3)]
put("tf1_mom_mean", sum(tf1s) / 4, "angle 0.1, seeds 0-3")
put("tf10_mom_mean", sum(tf10s) / 4, "stepnormfinal/tf10 (batched), angle 0.2, seeds 0-3")
put("tf10_batched_sps", sps("stepnormfinal/tf10/*", 0), "stepnormfinal/tf10 seed 0")
put("tf1_mom_15k", val("stepnormmom15k/*"), "stepnormmom15k")
put("exact_vs_fast_3000", NUM["np3000_mean"] - NUM["tf1_mom_mean"], "np3000_mean - tf1_mom_mean")

# chapter 4: statistics windows
put("win_ref", NUM["tf25_mom"], "= tf25_mom"); put("win_ref_sps", NUM["tf25_mom_sps"], "= tf25_mom_sps")
for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095"):
    put(f"win_{k}", val(f"stepnormfactors/{k}/*"), f"stepnormfactors/{k}")
    put(f"win_{k}_sps", sps(f"stepnormfactors/{k}/*"), f"stepnormfactors/{k}")

# chapter 4: FAST and batching
ref, fast = _rows(one("stepnormcombo/idref/*")), _rows(one("stepnormcombo/idfast/*"))
same = [r.get("train_loss") for r in ref if "train_loss" in r] == [r.get("train_loss") for r in fast if "train_loss" in r]
same = same and [r.get("val_loss") for r in ref if "val_loss" in r] == [r.get("val_loss") for r in fast if "val_loss" in r]
put("fast_bit_identical", same, "stepnormcombo/idref vs idfast: every logged train and val loss equal")
put("fast_sps", sps("stepnormcombo/cos0.2/*"), "stepnormcombo/cos0.2 (FAST, unbatched)")
put("batch_val", val("stepnormcombo/batch/*"), "stepnormcombo/batch")
put("batch_sps", sps("stepnormcombo/batch/*"), "stepnormcombo/batch")
put("speedup_vs_exact", NUM["tf1_mom_sps"] / NUM["batch_sps"], "tf1_mom_sps / batch_sps")

# 2.2: empirical vs true Fisher (earlier variant, 3000 steps)
put("fisher_empirical", best("fisher/*", at=2999, ngd_fisher_mc_every=0)[0], "best of runs-record/fisher at step 2999 (runs cut by --max-steps), empirical D")
put("fisher_true", best("fisher/*", at=2999, ngd_fisher_mc_every=1)[0], "best of runs-record/fisher at step 2999, D from sampled targets")

# appendix D: initial spectrum
I, IL = "stepnorminit", "stepnorminitlong"
put("init_normal", NUM["batch_val"], "= batch_val")
for k, pat in (("2", f"{I}/pl2/*"), ("2.5", f"{IL}/ours3000-lr0.2-pl2.5/*"), ("3", f"{I}/pl3/*"),
               ("4", f"{IL}/ours3000-lr0.2-pl4/*"), ("5", f"{I}/pl5/*"), ("flat", f"{I}/flat/*"), ("orth", f"{I}/orth/*")):
    put(f"init_{k}", val(pat), pat)
put("init_best_d", NUM["init_3"] - NUM["init_normal"], "init_3 - init_normal")
put("init_flat_d", NUM["init_flat"] - NUM["init_normal"], "init_flat - init_normal")
put("init_orth_d", NUM["init_orth"] - NUM["init_normal"], "init_orth - init_normal")
g = [NUM["init_normal"] - NUM["init_3"]] + [val("stepnormfinal/tf25/*", s_) - val(f"{IL}/ours3000-lr0.2-pl3/*", s_) for s_ in (1, 2, 3)]
put("init_gain_3000_mean", sum(g) / 4, "Natural Pion, alpha 3 vs normal, 3000 steps, seeds 0-3")
put("init_gain_15k", NUM["np15k"] - val(f"{IL}/ours15000-lr0.05-pl3/*"), "Natural Pion, 15000 steps")
put("init_pion_gain_3000", NUM["pion3000_s0"] - val(f"{IL}/pion3000-lr1e-3-pl3/*"), "Pion, 3000 steps")
put("init_pion_gain_15k", NUM["pion15k"] - val(f"{IL}/pion15000-lr1e-3-pl3/*"), "Pion, 15000 steps")

# small derived values used in the text
put("tf25_nomom_d", NUM["tf25_nomom"] - NUM["tf1_nomom"], "tf25_nomom - tf1_nomom")
put("tf10_mom_d", NUM["tf10_mom"] - NUM["tf1_mom"], "tf10_mom - tf1_mom")
put("tf25_mom_d", NUM["tf25_mom"] - NUM["tf1_mom"], "tf25_mom - tf1_mom")
put("win_maxdev", max(abs(NUM[f"win_{k}"] - NUM["win_ref"]) for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095")),
    "largest |win_* - win_ref|")
put("pion_vs_paper", NUM["pion_full"] - 3.3575, "pion_full - 3.3575 (the Pion paper's own 60M number, cited)")

# probes and checks that are not in runs-record
try:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/check_fisher.py")], capture_output=True, text=True, check=True).stdout
    vals = dict(line.split(None, 2)[:2] if False else (line.rsplit(" ", 1)[0].strip(), line.rsplit(" ", 1)[1]) for line in out.splitlines() if line)
    put("mc_err_in", float(out.split("in  rel err")[1].split()[0]), "scripts/analysis/check_fisher.py")
    put("mc_err_out", float(out.split("out rel err")[1].split()[0]), "scripts/analysis/check_fisher.py")
    put("pullback_err", float(out.split("pullback")[1].split()[0]), "scripts/analysis/check_fisher.py")
except Exception as e:  # numpy missing, etc.
    for k in ("mc_err_in", "mc_err_out", "pullback_err"):
        missing(k, f"scripts/analysis/check_fisher.py failed: {e}")
for k, need in (("tf32_drift", "scripts/probes/tf32.py output, recorded"),
                ("fp32_drift_3000", "scripts/probes/spectrum_drift.py on archived checkpoints, recorded"),
                ("svd_share_of_step", "scripts/probes/power_acc.py or step-cost probe output, recorded"),
                ("power_speedup", "power iteration vs matrix_norm timing, recorded"),
                ("power_cold_1", "scripts/probes/power_acc.py, recorded"), ("power_cold_5", "same"), ("power_cold_20", "same"),
                ("power_warm_1", "same"), ("ns_cost", "scripts/probes/ns_cayley.py timing, recorded"),
                ("ns_err_02", "scripts/probes/ns_law.py, recorded"), ("ns_err_04", "same"), ("ns_err_08", "same"),
                ("fp32_onestep_err", "same"), ("ns_drift_toy", "same"), ("cayley_drift_toy", "same"),
                ):
    missing(k, need)
p0 = [sps_median(d) for d in runs("stepnormpower/p0/*")]
put("svd_sps_lo", min(p0), "stepnormpower/p0 runs, median s/step between logged rows, min over angles")
put("svd_sps_hi", max(p0), "same, max over angles")


def main():
    out = ROOT / "docs/paper"
    json.dump({k: v for k, v in NUM.items()}, open(out / "numbers.json", "w"), indent=1, default=str)
    with open(out / "numbers.md", "w") as f:
        f.write("# Numbers of the preprint\n\nGenerated by `scripts/analysis/paper_numbers.py`; do not edit.\n\n")
        f.write("| key | value | source |\n|---|---|---|\n")
        for k in NUM:
            v = NUM[k]
            f.write(f"| `{k}` | {'**MISSING**' if v is None else (f'{v:.6g}' if isinstance(v, float) else v)} | {NOTE[k]} |\n")
    n_miss = sum(v is None for v in NUM.values())
    print(f"{len(NUM)} keys, {n_miss} missing")


if __name__ == "__main__":
    main()
