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
    cross, bracket = None, None
    for c in ours_curves:
        pts = sorted((s + 1, v) for s, v in c["vals"].items())
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v1 <= target < v0:
                s = s0 + (v0 - target) / (v0 - v1) * (s1 - s0)
                if cross is None or s < cross:
                    cross, bracket = s, (s0, s1)
                break
    put(f"steps_pion_{steps}_val", target, pat)
    put(f"steps_cross_{steps}", cross, f"first step any of our seed-0 runs reaches val(Pion {steps}), linear between evaluations")
    put(f"steps_ratio_{steps}", steps / cross, f"{steps} / steps_cross_{steps}")
    ratios.append(steps / cross)
    put(f"steps_ratio_{steps}_lo", steps / bracket[1], f"{steps} / the evaluation step after the crossing")
    put(f"steps_ratio_{steps}_hi", steps / bracket[0], f"{steps} / the evaluation step before the crossing")
put("steps_ratio_min", min(ratios), "min of steps_ratio_*")
put("steps_ratio_min_lo", min(NUM[f"steps_ratio_{n}_lo"] for n in (3840, 4500, 19200, 73242)), "min of steps_ratio_*_lo: the conservative bound")

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
def spread(rel: str) -> float:
    """Median over logged steps >= 1000 of p90/p10 of the realised angle across the 56 matrices."""
    ang = json.load(open(R / rel / "angles.json"))
    ratios = []
    for step, xs in ang.items():
        if int(step) < 1000:
            continue
        xs = sorted(abs(x) for x in xs)
        lo, hi = xs[int(0.1 * (len(xs) - 1))], xs[int(0.9 * (len(xs) - 1))]
        if lo > 0:
            ratios.append(hi / lo)
    ratios.sort()
    return ratios[len(ratios) // 2]


for k, rel in (("spectral", "stepnorm/ngd-pion-s-lr0.4-s0-abc48c0d29"),
               ("rms", "stepnormnorm/rms/ngd-pion-s-lr0.0024-s0-0e56770015"),
               ("fishernorm", "stepnormnorm/fisher/ngd-pion-s-lr0.0048-s0-39ae0e1733"),
               ("alpha", "tfac1alpha/ngd-pion-s-lr0.004-s0-98de9ab11a"),
               ("fixed", "tfac1/ngd-pion-s-lr0.002-s0-0abdc0de2e"),
               ("before", "stepnormpre/ngd-pion-s-lr3e-05-s0-72fbe69d19")):
    put(f"angle_spread_{k}", spread(rel), f"{rel}/angles.json (scripts/analysis/extract_angles.py): median over steps >= 1000 of p90/p10 across matrices")
put("angle_spread_fixed_c", NUM["angle_spread_fixed"], "= angle_spread_fixed")

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

# chapter 4: Pion at the same evaluation cadence as the 3000-step runs; memory; unprofiled remainder
put("pion3000_sps", sps("stepnormfinal/pion3840/*", 0), "Pion 3840 steps, seed 0, eval every 500 like the 3000-step runs (the 3000-step Pion run, adamw2d, ran on older code and is not used for timing)")
put("cost_ratio_3000", NUM["batch_sps"] / NUM["pion3000_sps"], "batch_sps / pion3000_sps")
for t in ("tfac1", "tfac25"):
    parts = sum(NUM[f"prof_{t}_{k}"] for k in ("refactor", "angle", "retract", "rest", "stats"))
    put(f"prof_{t}_other", NUM[f"prof_{t}_step"] - parts, f"prof_{t}_step - sum of the profiled parts (forward, backward, AdamW, data)")

def peak(pattern):
    g = [r.get("peak_gb") for r in _rows(one(pattern)) if r.get("peak_gb")]
    return max(g)
put("peak_pion", peak("adamw2d/pion-lr0.001-s0-9e6e6d32f6"), "max peak_gb in the Pion 3000-step log (micro-batch 256)")
put("peak_np", peak("stepnormcombo/batch/*"), "max peak_gb in the Natural Pion Fast 3000-step log (micro-batch 256)")
put("peak_d", NUM["peak_np"] - NUM["peak_pion"], "peak_np - peak_pion")

# open checks: per-head query split; regularisation grid
put("qhead_val", val("npqhead/*"), "npqhead: natural-pion-fast, query rotated per head, 3000 steps, seed 0")
put("qhead_d", NUM["qhead_val"] - NUM["batch_val"], "qhead_val - batch_val (query rotated whole)")
for e in ("1e-6", "1e-5", "1e-3", "1e-2"):
    put(f"reg_eps_{e}", val(f"npreg/eps{e}-d0/*"), f"npreg/eps{e}-d0: floor eps = {e}, no damping")
for d in ("1e-4", "1e-3", "1e-2", "1e-1"):
    put(f"reg_damp_{d}", val(f"npreg/eps1e-6-d{d}/*"), f"npreg/eps1e-6-d{d}: floor 1e-6 plus damping {d} x max eigenvalue")
put("reg_best_other", min([NUM[f"reg_eps_{e}"] for e in ("1e-6", "1e-5", "1e-3", "1e-2")] + [NUM[f"reg_damp_{d}"] for d in ("1e-4", "1e-3", "1e-2", "1e-1")]),
    "best of the eight alternatives")
put("reg_best_other_d", NUM["reg_best_other"] - NUM["batch_val"], "reg_best_other - batch_val (eps 1e-4)")
put("npcheck2_identical", True if all(a == b for a, b in zip(
    [r["train_loss"] for r in _rows(one("npcheck2/*")) if "train_loss" in r],
    [r["train_loss"] for r in _rows(one("stepnormcombo/batch/*")) if "train_loss" in r])) else False,
    "npcheck2 (natural-pion-fast, operator 1/2) vs stepnormcombo/batch: every logged train loss of the first 500 steps")

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
put("init_gain_full", NUM["np_full_clock"] - val(f"{IL}/ours56900-lr0.05-pl3/*"), "Natural Pion, 56900 steps")
put("init_pion_gain_full", NUM["pion_full"] - val(f"{IL}/pion73242-lr1e-3-pl3/*"), "Pion, 73242 steps")

# small derived values used in the text
put("tf25_nomom_d", NUM["tf25_nomom"] - NUM["tf1_nomom"], "tf25_nomom - tf1_nomom")
put("tf10_mom_d", NUM["tf10_mom"] - NUM["tf1_mom"], "tf10_mom - tf1_mom")
put("tf25_mom_d", NUM["tf25_mom"] - NUM["tf1_mom"], "tf25_mom - tf1_mom")
put("win_maxdev", max(abs(NUM[f"win_{k}"] - NUM["win_ref"]) for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095")),
    "largest |win_* - win_ref|")
put("pion_vs_paper", NUM["pion_full"] - 3.3575, "pion_full - 3.3575 (the Pion paper's own 60M number, cited)")

# analytic: the truncated exponential's departure from orthogonality, ||T^T T - I||_2 = (angle)^4 / 4
for th in (0.05, 0.2):
    put(f"trunc_err_{th}", th ** 4 / 4, f"theta^4/4 at theta = {th} (Proposition 4)")
rc = json.load(open(R / "probes" / "rl_refactor_cost_b200.json"))
put("rl_refactor_b200", rc["refactor_total_s"], "runs-record/probes/rl_refactor_cost_b200.json (scripts/probes/rl_refactor_cost.py, job 397885)")
PA = R / "probes" / "pion_angles.json"
if PA.exists():
    pa = json.load(open(PA))          # {step: [angle of every rotated block]}, from scripts/probes/pion_angles.py
    per_step = [max(v) for k, v in pa.items() if int(k) > 0]
    put("pion_angle_max", max(per_step), "runs-record/probes/pion_angles.json: largest block angle, first 1058 steps of the 3840-step run (the probe hit its time limit there; the learning rate is still within 15% of its peak)")
    put("pion_angle_steps", max(int(k) for k in pa), "last logged step of the Pion angle probe")
    meds, sprs = [], []
    for k, v in pa.items():
        if int(k) == 0:
            continue
        xs = sorted(v)
        meds.append(xs[len(xs) // 2])
        sprs.append(xs[int(0.9 * (len(xs) - 1))] / xs[int(0.1 * (len(xs) - 1))])
    meds.sort(); sprs.sort()
    put("trunc_err_pion", meds[len(meds) // 2] ** 4 / 4, "(pion_angle_block_median)^4 / 4")
    put("pion_angle_block_median", meds[len(meds) // 2], "pion_angles.json: median over steps of the median block angle")
    put("pion_angle_spread", sprs[len(sprs) // 2], "pion_angles.json: median over steps of p90/p10 of the block angles")
    per_step.sort()
    put("pion_angle_median", per_step[len(per_step) // 2], "same, median over logged steps of the largest block angle")
else:
    missing("pion_angle_max", "run scripts/probes/pion_angles.sbatch and archive its output as runs-record/probes/pion_angles.json")

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
for ret in ("ns", "trunc"):
    for st in ("3000", "15000"):
        hits = runs(f"np{ret}/s{st}/*")
        if hits and info(hits[0])["done"]:
            put(f"ret_{ret}_{st}", info(hits[0])["val"], f"np{ret}/s{st}: natural-pion-fast with the {ret} retraction")
        else:
            missing(f"ret_{ret}_{st}", f"scripts/paper/natural_pion_ns.sbatch with RET={ret}, {st} steps")
    f = R / "probes" / f"paper_checks_{ret}.json"
    if f.exists():
        d = json.load(open(f))
        put(f"ret_{ret}_drift", d["fp32_drift_3000_median"], f"{f.name}: spectrum of the 3000-step {ret} checkpoint vs init, median over matrices")
        put(f"ret_{ret}_drift_max", d["fp32_drift_3000_max"], f"{f.name}: worst matrix")
    else:
        missing(f"ret_{ret}_drift", f"scripts/probes/paper_checks.py <np{ret}/s3000 checkpoint> {ret}")
        missing(f"ret_{ret}_drift_max", "same")
for ret in ("ns", "trunc"):
    put(f"ret_{ret}_sps", sps(f"np{ret}/s3000/*"), f"np{ret}/s3000 (RTX)")
put("ret_ns_saving", NUM["batch_sps"] - NUM["ret_ns_sps"], "batch_sps - ret_ns_sps (RTX)")
put("ret_hw_noise", max(abs(val(f"np{r}/s3000/*") - val(f"np{r}_b200/s3000/*")) for r in ("ns", "trunc")),
    "same configuration on RTX and B200 (np{ns,trunc}/s3000 vs _b200): max |val difference|")
_m = json.load(open(one("npstats_fp32_b200/s3000/*") / "manifest.json"))
put("tokens_per_step", _m["tokens_per_step"], "manifest tokens_per_step")
# ns_delta: Newton-Schulz in the correction form, B200 only so far
hits = runs("npnsdelta_b200/s3000/*")
if hits and info(hits[0])["done"]:
    put("ret_nsd_3000", info(hits[0])["val"], "npnsdelta_b200/s3000")
    put("ret_nsd_sps_b200", sps("npnsdelta_b200/s3000/*"), "npnsdelta_b200/s3000 (B200)")
else:
    missing("ret_nsd_3000", "RET=ns_delta OUTROOT=npnsdelta_b200 sbatch -p b200 --array=0 scripts/paper/natural_pion_ns.sbatch")
    missing("ret_nsd_sps_b200", "same")
f = R / "probes" / "paper_checks_ns_delta.json"
if f.exists():
    d = json.load(open(f))
    put("ret_nsd_drift", d["fp32_drift_3000_median"], f.name)
    put("ret_nsd_drift_max", d["fp32_drift_3000_max"], f.name)
else:
    missing("ret_nsd_drift", "scripts/probes/paper_checks.py <npnsdelta_b200/s3000 checkpoint> ns_delta")
    missing("ret_nsd_drift_max", "same")
for ret in ("ns", "trunc"):
    put(f"ret_{ret}_sps_b200", sps(f"np{ret}_b200/s3000/*"), f"np{ret}_b200/s3000 (B200)")
# per-step change of ||W||_F^2 in training, first 200 steps (scripts/probes/ns_diag.py)
for ret in ("cayley", "ns", "trunc", "ns_delta"):
    f = R / "probes" / f"ns_diag_{ret}.json"
    if f.exists():
        d = json.load(open(f))
        ms = sorted(x["mean_signed"] for x in d["steps"])
        put(f"diag_{ret}_step", ms[len(ms) // 2], f"{f.name}: median over steps of the mean over matrices")
        put(f"diag_{ret}_neg", sum(m < 0 for m in ms), f"{f.name}: steps where ||W||_F^2 fell")
        put(f"diag_{ret}_sum", sum(ms), f"{f.name}: summed over the 200 steps")
        if ret == "ns":
            put("diag_sigma_ratio", sorted(r[1] for r in d["sigma_ratio"])[len(ms) // 2],
                f"{f.name}: median sigma_est / ||X||_2")
            put("diag_precision", d["summary"]["precision_seen"] == ['[false, false, "highest"]'],
                f"{f.name}: autocast off, tf32 off, highest fp32 precision at every step")
    else:
        for k in ("step", "neg", "sum"):
            missing(f"diag_{ret}_{k}", f"RETS={ret} sbatch scripts/probes/ns_diag.sbatch")
# rounding of the retractions on bare matrices (scripts/probes/ns_rounding.py)
f = R / "probes" / "ns_rounding.json"
if f.exists():
    d = json.load(open(f))
    for ret in ("cayley", "ns", "ns_delta"):
        for th in ("0.2", "0.05"):
            put(f"rnd_{ret}_{th}", d[f"n1376_th{th}"][ret], f"{f.name}: 1376 x 1376, 200 one-sided steps")
    put("rnd_delta_vs_cayley", max(r["delta_vs_cayley"] for r in d.values()), f"{f.name}: max over sizes")
else:
    missing("rnd_ns_0.2", "python scripts/probes/ns_rounding.py")
# statistics: one A per input tensor, fp32 or bf16 inputs (scripts/paper/natural_pion_stats.sbatch, B200)
for st in ("fp32", "bf16"):
    hits = runs(f"npstats_{st}_b200/s3000/*")
    if hits and info(hits[0])["done"]:
        put(f"stats_{st}_val", info(hits[0])["val"], f"npstats_{st}_b200/s3000")
        put(f"stats_{st}_sps", sps(f"npstats_{st}_b200/s3000/*"), f"npstats_{st}_b200/s3000 (B200)")
    else:
        missing(f"stats_{st}_val", f"STATS={st} OUTROOT=npstats_{st}_b200 sbatch -p b200 scripts/paper/natural_pion_stats.sbatch")
if "stats_bf16_val" in NUM and "stats_fp32_val" in NUM:
    put("stats_bf16_d", NUM["stats_bf16_val"] - NUM["stats_fp32_val"], "stats_bf16_val - stats_fp32_val")
    put("stats_bf16_saving", NUM["stats_fp32_sps"] - NUM["stats_bf16_sps"], "stats_fp32_sps - stats_bf16_sps")
hits = runs("npstats_bf16_b200/s15000/*")
if hits and info(hits[0])["done"]:
    put("stats_bf16_15k", info(hits[0])["val"], "npstats_bf16_b200/s15000")
else:
    missing("stats_bf16_15k", "STATS=bf16 OUTROOT=npstats_bf16_b200 sbatch -p b200 --array=1 scripts/paper/natural_pion_stats.sbatch")

PC = R / "probes" / "paper_checks.json"
pc = json.load(open(PC)) if PC.exists() else {}
for k in ("trunc_drift_toy", "tf32_drift", "fp32_drift_200", "fp32_drift_3000", "fp32_drift_3000_max", "fp32_drift_3000_signed_mean", "power_speedup",
          "power_cold_1", "power_cold_5", "power_cold_20", "power_warm_1", "ns_cost", "ns_err_02", "ns_err_04",
          "ns_err_08", "fp32_onestep_cayley", "fp32_onestep_ns", "ns_drift_toy", "cayley_drift_toy", "kfac_vs_exact",
          "svd_exact_s", "svd_power_s"):
    if k in pc:
        put(k, pc[k], "runs-record/probes/paper_checks.json (scripts/probes/paper_checks.py)")
    else:
        missing(k, "run scripts/probes/paper_checks.sbatch (writes runs-record/probes/paper_checks.json)")
if "svd_exact_s" in pc:
    rest = NUM["prof_tfac25_refactor"] + NUM["prof_tfac25_retract"] + NUM["prof_tfac25_rest"]
    put("svd_share_of_step", pc["svd_exact_s"] / (pc["svd_exact_s"] + rest),
        "exact spectral norms / (that + refresh, retraction and rest of the T_fac 25 optimizer step, profiled)")
else:
    missing("svd_share_of_step", "needs paper_checks.json")
p0 = [sps_median(d) for d in runs("stepnormpower/p0/*")]
put("sched_linear_fast", val("stepnormcombo/lin0.2/*"), "stepnormcombo/lin0.2 (Natural Pion Fast, linear decay, peak 0.2)")
put("clock_pion_sps", sps("stepnormclock/pion-*"), "stepnormclock Pion 5300-step run: the step time measured before the clock runs were planned")
put("clock_steps_planned", 73242 * NUM["clock_pion_sps"] / NUM["batch_sps"], "73242 x clock_pion_sps / batch_sps; rounded to 56 900 for the run")
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
