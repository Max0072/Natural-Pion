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

L = "stepnormlong"

# ------------------------------------------------------------------------------------------- Pion (the baseline)
pf = info(one(f"{L}/pion73242-lr1e-3/*"))
put("pion_full", pf["val"], f"{L}/pion73242-lr1e-3")
put("pion_full_wall", pf["wall"], f"{L}/pion73242-lr1e-3, wall")
put("pion_sps", pf["wall"] / pf["steps"], "pion_full wall / steps")

# Pion lr grid
for steps, lrs in ((3840, ("5e-4", "1e-3", "2e-3", "4e-3")), (19200, ("1e-3", "2e-3", "4e-3"))):
    for lr in lrs:
        pat = f"stepnormfinal/pion{steps}/*" if lr == "1e-3" else f"{L}/pion{steps}-lr{lr}/*"
        put(f"pion_lr_{steps}_{lr}", val(pat, seed=0), pat)

put("pion_lr_3840_better", NUM["pion_lr_3840_1e-3"] - NUM["pion_lr_3840_2e-3"], "Pion 3840: lr 1e-3 - lr 2e-3")

p3000 = [val("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [val("seedshead/pion-lr0.001-*", s) for s in (1, 2, 3)]
put("pion3000_s0", p3000[0], "adamw2d/pion-lr0.001-s0-9e6e6d32f6 (Pion, 3000 steps, AdamW 8e-3)")
put("pion3000_mean", sum(p3000) / 4, "Pion 3000, seeds 0-3 (adamw2d, seedshead)")
put("pion15k", val("ladder15k/pion-lr0.001-s0-f67fdad3e9"), "ladder15k/pion-lr0.001-s0-f67fdad3e9")

put("pion3000_sps", sps("stepnormfinal/pion3840/*", 0), "Pion 3840 steps, seed 0, eval every 500 like the 3000-step runs (the 3000-step Pion run, adamw2d, ran on older code and is not used for timing)")

def peak(d: Path) -> float:
    return max(r.get("peak_gb") or 0.0 for r in _rows(d))
put("peak_pion", peak(one("adamw2d/pion-lr0.001-s0-9e6e6d32f6")), "max peak_gb in the Pion 3000-step log (micro-batch 256)")

_tr = [r for r in _rows(one(f"{L}/pion73242-lr1e-3/*")) if "train_loss" in r]
_tr = [r["train_loss"] for r in _tr if r["step"] > _tr[-1]["step"] - 2000]
put("pion_full_train", sum(_tr) / len(_tr), f"{L}/pion73242-lr1e-3: mean logged training loss over the last 2000 steps")
put("pion_vs_paper", NUM["pion_full_train"] - 3.3575,
    "pion_full_train - 3.3575 (the Pion paper's final training loss, Fig. 5, cited)")

# analytic: T^T T = I + Omega^4/4 (Proposition 4), so one step moves a singular value by about (angle)^4 / 8
for th in (0.05, 0.2):
    put(f"trunc_err_{th}", th ** 4 / 8, f"theta^4/8 at theta = {th} (Proposition 4: sqrt(1 + theta^4/4) - 1)")
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
    put("trunc_err_pion", meds[len(meds) // 2] ** 4 / 8, "(pion_angle_block_median)^4 / 8")
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

# per-step change of ||W||_F^2 in training, first 200 steps (scripts/probes/ns_diag.py)
for ret in ("cayley", "ns", "trunc", "ns_delta"):
    f = R / "probes" / f"ns_diag_{ret}.json"
    if f.exists():
        d = json.load(open(f))
        ms = sorted(x["mean_signed"] for x in d["steps"])
        put(f"diag_{ret}_step", ms[len(ms) // 2], f"{f.name}: median over steps of the mean over matrices")
        put(f"diag_{ret}_neg", sum(m < 0 for m in ms), f"{f.name}: steps where ||W||_F^2 fell")
        put(f"diag_{ret}_sum", sum(ms), f"{f.name}: summed over the 200 steps")
        if ret == "ns" and d["sigma_ratio"]:
            put("diag_sigma_ratio", sorted(r[1] for r in d["sigma_ratio"])[len(ms) // 2],
                f"{f.name}: median sigma_est / ||X||_2")
            put("diag_precision", d["summary"]["precision_seen"] == ['[false, false, "highest"]'],
                f"{f.name}: autocast off, tf32 off, highest fp32 precision at every step")
    else:
        for k in ("step", "neg", "sum"):
            missing(f"diag_{ret}_{k}", f"RETS={ret} sbatch scripts/probes/ns_diag.sbatch")

f = R / "probes" / "retraction_bias.json"
if f.exists():
    d = json.load(open(f))["n1376_th0.05"]
    for tag in ("solve", "ns", "ns_delta"):
        put(f"rb_{tag}_mean", d[tag]["mean"], f"{f.name}: 1376 x 1376, angle 0.05, mean tr(R^T R - I)/n")
        put(f"rb_{tag}_z", d[tag]["z"], f"{f.name}: same, mean over its standard error")
else:
    missing("rb_solve_mean", "python scripts/probes/retraction_bias.py")

# optimizer step alone, by part (scripts/probes/optimizer_step_bench.py, synchronised timers)
f = R / "probes" / "optimizer_step_bench.json"
if f.exists():
    d = json.load(open(f))
    put("bench_pion", d["pion"], f.name + ": Pion, whole optimizer step")
    for tag, key in (("old", "fast_cayley_loop_parts"), ("new", "fast_nsdelta_batchsigma_bf16_parts")):
        for part, v in d[key].items():
            put(f"bench_{tag}_" + part.replace("^-1", "inv").replace("+", "_").replace(" ", "_").replace("(", "").replace(")", ""),
                v, f"{f.name}: {key} {part}")
    for tag in ("old", "new"):
        put(f"bench_{tag}_shared", sum(NUM[f"bench_{tag}_{k}"] for k in ("generators_momentum", "apply_W")),
            "generators+momentum + apply W")
else:
    missing("bench_pion", "scripts/probes/optimizer_step_bench.sbatch")

# FLOP per step (scripts/analysis/flops.py)
f = R / "probes" / "flops.json"
if f.exists():
    d = json.load(open(f))
    for size in ("60M", "1.3B"):
        tag = size.replace(".", "_")
        put(f"flops_{tag}_pion", d[size]["pion_overhead"], f"{f.name}: Pion optimizer / model FLOP, {size}")
        put(f"flops_{tag}_np", d[size]["np_overhead"], f"{f.name}: Natural Pion Fast optimizer / model FLOP, {size}")
        put(f"flops_{tag}_ratio", d[size]["np_over_pion"], f"{f.name}: step FLOP, Natural Pion Fast / Pion, {size}")
        put(f"flops_{tag}_shared", (d[size]["natural_pion_fast"]["generators"] + d[size]["natural_pion_fast"]["rotation"])
            / d[size]["model"], f"{f.name}: generators + rotation / model FLOP, {size}")
        for part, v in d[size]["natural_pion_fast"].items():
            put(f"flops_{tag}_part_" + part.split(" ")[0].replace("^-1", "inv"), v / d[size]["model"],
                f"{f.name}: {part} / model FLOP, {size}")
else:
    missing("flops_60M_np", "python scripts/analysis/flops.py")

# ---------------------------------------------------------------- Section 5 (v2.sbatch rows 99-110)

# Section 5 with the query projection per head (scripts/paper/v2.sbatch, rows 99-110)
def _v(pattern, seed=None):
    try:
        return val(pattern, seed)
    except SystemExit:
        return None
V2 = {}
for st in (3000, 15000):
    vs = [_v(f"npv2/s{st}/*", sd) for sd in range(4)]
    for sd, v in enumerate(vs):
        (put(f"v2_{st}_s{sd}", v, f"npv2/s{st} seed {sd}") if v is not None
         else missing(f"v2_{st}_s{sd}", f"v2.sbatch, {st} steps, seed {sd}"))
    if all(v is not None for v in vs):
        put(f"v2_{st}_mean", sum(vs) / 4, f"npv2/s{st}, seeds 0-3")
    else:
        missing(f"v2_{st}_mean", "all four seeds")
p3 = [val("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [val("seedshead/pion-lr0.001-*", sd) for sd in (1, 2, 3)]
if NUM.get("v2_3000_mean") is not None:
    put("v2_gap3000", NUM["pion3000_mean"] - NUM["v2_3000_mean"], "pion3000_mean - v2_3000_mean")
    for sd in range(4):
        put(f"v2_gap3000_s{sd}", p3[sd] - NUM[f"v2_3000_s{sd}"], f"Pion 3000 seed {sd} - v2 seed {sd}")
    put("v2_gap3000_min", min(NUM[f"v2_gap3000_s{sd}"] for sd in range(4)), "smallest per-seed gap at 3000")
else:
    missing("v2_gap3000", "v2_3000_mean"); missing("v2_gap3000_min", "v2_3000_mean")
if NUM.get("v2_15000_s0") is not None:
    put("v2_gap15k", NUM["pion15k"] - NUM["v2_15000_s0"], "pion15k - v2_15000_s0 (seed 0)")
else:
    missing("v2_gap15k", "v2_15000_s0")
vf = runs("npv2/s73242/*")
if vf and info(vf[0])["done"]:
    i = info(vf[0])
    put("v2_full", i["val"], "npv2/s73242")
    put("v2_full_wall", i["wall"], "npv2/s73242, wall")
    put("v2_gap_full", NUM["pion_full"] - i["val"], "pion_full - v2_full")
    put("v2_full_sps", i["wall"] / 73242, "npv2/s73242 wall / steps")
    put("v2_full_ratio", NUM["v2_full_sps"] / NUM["pion_sps"], "v2_full_sps / pion_sps")
else:
    for k in ("v2_full", "v2_full_wall", "v2_gap_full", "v2_full_sps", "v2_full_ratio"):
        missing(k, "v2.sbatch row 110 (73 242 steps)")
try:
    put("v2_sps", sps("npv2/s3000/*", 0), "npv2/s3000 seed 0 (RTX)")
    put("v2_ratio", NUM["v2_sps"] / NUM["pion3000_sps"], "v2_sps / pion3000_sps")
except SystemExit:
    missing("v2_sps", "npv2/s3000 seed 0")

# steps saving from runs with their own full schedules (scripts/analysis/budget_fit.py)
f = R / "probes" / "budget_fit.json"
if f.exists():
    d = json.load(open(f))
if f.exists() and d.get("complete"):
    inside = {B: t for B, t in d["targets"].items() if t["interpolated"]}
    for B, t in inside.items():
        put(f"budget_{B}_L", t["L"], f"{f.name}: Pion val at {B} steps")
        put(f"budget_{B}_pow", t["ratio_powerlaw"], f"{f.name}: B_Pion / B_NP(L), power law")
        put(f"budget_{B}_int", t["ratio_interp"], f"{f.name}: same, interpolation in log B")
    allr = [x for t in inside.values() for x in (t["ratio_powerlaw"], t["ratio_interp"])]
    put("budget_lo", min(allr), f"{f.name}: smallest saving over both models and every interpolated target")
    put("budget_hi", max(allr), f"{f.name}: largest")
    put("budget_resid", max(d["residual_max_pion"], d["residual_max_np"]), f"{f.name}: worst fit residual")
    put("budget_n_pion", len(d["pion"]["B"]), f"{f.name}: Pion budgets")
    put("budget_n_np", len(d["natural_pion"]["B"]), f"{f.name}: Natural Pion budgets")
    for B, e in d["equal_time"].items():
        put(f"eqt_{B}_steps", e["B_np"], f"{f.name}: Natural Pion steps in Pion's time at {B}")
        put(f"eqt_{B}_L", e["L_np_powerlaw"], f"{f.name}: Natural Pion loss there, from the fit")
        put(f"eqt_{B}_gap", e["gap_powerlaw"], f"{f.name}: Pion - Natural Pion at equal time, fit")
        put(f"eqt_{B}_gap_int", e["gap_interp"], f"{f.name}: same, interpolation")
        put(f"eqt_{B}_Lp", e["L_pion"], f"{f.name}: Pion at {B}")
else:
    for k in ["budget_lo", "budget_hi", "budget_resid", "budget_n_pion", "budget_n_np"] + \
             [f"budget_{B}_{x}" for B in (5300, 15000, 19200, 30000, 45000, 73242) for x in ("L", "pow", "int")] + \
             [f"eqt_{B}_{x}" for B in (3840, 19200, 73242) for x in ("steps", "L", "gap", "gap_int", "Lp")]:
        missing(k, "python scripts/analysis/budget_fit.py (needs every npv2 run)")

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

# ------------------------------------------------------------------- Sections 3-4, 5.4, Appendix D: scripts/paper/v2.sbatch
# Every run is a row of v2.sbatch, found by its tag under runs-record/npv2/<tag>/ (collect_npv2.sh copies them there).
V = "npv2"


def vrun(tag: str, seed: int = 0) -> Path | None:
    hits = runs(f"{V}/{tag}/*", seed)
    return max(hits, key=lambda d: (d / "log.jsonl").stat().st_mtime) if hits else None


def vinfo(tag: str, seed: int = 0) -> dict | None:
    d = vrun(tag, seed)
    if d is None:
        return None
    i = info(d)
    return i if i["done"] else None


def vval(key: str, tag: str, seed: int = 0):
    i = vinfo(tag, seed)
    if i is None:
        missing(key, f"v2.sbatch row {tag}, seed {seed}")
        return None
    put(key, i["val"], f"{V}/{tag} seed {seed}")
    return i["val"]


def vsps(key: str, tag: str, seed: int = 0, need_done: bool = True):
    """Wall seconds per step of a v2 run (its last logged wall over its last logged step + 1)."""
    d = vrun(tag, seed)
    if d is None or (need_done and not info(d)["done"]):
        missing(key, f"v2.sbatch row {tag}, seed {seed}")
        return None
    rows = [r for r in _rows(d) if "wall" in r]
    v = rows[-1]["wall"] / (rows[-1]["step"] + 1)
    put(key, v, f"{V}/{tag} seed {seed}, wall / steps (RTX)")
    return v


def vbest(key: str, tags: list[str], seed: int = 0):
    """Best final val over an angle grid (one tag per angle). Puts key, key_lr, key_tag and key_interior; missing
    until every point of the grid is done."""
    infos = [(t, vinfo(t, seed)) for t in tags]
    if any(i is None for _, i in infos):
        todo = [t for t, i in infos if i is None]
        for k in (key, key + "_lr", key + "_interior"):
            missing(k, "v2.sbatch rows " + ", ".join(todo))
        return None
    infos.sort(key=lambda ti: ti[1]["lr"])
    k = min(range(len(infos)), key=lambda j: infos[j][1]["val"])
    interior = 0 < k < len(infos) - 1
    t, i = infos[k]
    put(key, i["val"], f"best of {', '.join(tags)}: {t} (angle {i['lr']}{'' if interior else ', EDGE: extend the grid'})")
    put(key + "_lr", i["lr"], f"angle of {key}")
    put(key + "_interior", interior, f"is {key} inside its grid")
    NOTE[key + "_tag"], NUM[key + "_tag"] = "run of the best angle", t
    return i["val"]


def derive(key: str, fn, deps: tuple, note: str):
    vals = [NUM.get(d) for d in deps]
    if any(v is None for v in vals):
        missing(key, "needs " + ", ".join(d for d, v in zip(deps, vals) if v is None))
        return None
    put(key, fn(*vals), note)
    return NUM[key]


def grid(prefix: str, angles) -> list[str]:
    return [f"{prefix}-{a}" for a in angles]


# the method's own runs, read by many rows below
REF = vval("m_3000", "s3000")                              # Natural Pion Fast, angle 0.2, momentum 0.9, seed 0
vval("m_15000", "s15000")
vsps("m_sps", "s3000")
_m = vrun("s3000")
if _m is not None:
    _man = json.load(open(_m / "manifest.json"))
    put("tokens_per_step", _man["tokens_per_step"], "manifest tokens_per_step")
    put("stats_tokens", _man["config"]["micro_batch"] * _man["config"]["model"]["seq_len"],
        "micro_batch x seq_len: the hooks record the first micro-batch of a step (harness/train.py)")
    put("peak_np", peak(_m), "max peak_gb in the Natural Pion Fast 3000-step log (micro-batch 256)")
else:
    for k in ("tokens_per_step", "stats_tokens", "peak_np"):
        missing(k, "v2.sbatch row s3000")
derive("peak_d", lambda a, b: a - b, ("peak_np", "peak_pion"), "peak_np - peak_pion")

# 3.1: the true Fisher (D from targets drawn from the model) against the empirical one, best of three angles each
vbest("fisher_true", grid("fisher-mc", ("0.1", "0.2", "0.4")))
# the empirical Fisher is the method itself: its angle grid is the direction grid with momentum
vbest("fisher_empirical", ["dir-nat-mom-0.1", "s3000", "dir-nat-mom-0.4", "dir-nat-mom-0.8"])

# 3.4 / 4.8: retractions (3000 steps angle 0.2; 15000 steps angle 0.05)
vval("ret_cayley_3000", "ret-cayley")
vval("ret_ns_3000", "ret-ns")
vval("ret_trunc_3000", "ret-trunc")
vval("ret_trunc_15000", "ret-trunc-15k")
vsps("ret_cayley_sps", "ret-cayley")
vsps("ret_ns_sps", "ret-ns")
vsps("ret_trunc_sps", "ret-trunc")

# 4.1: the step by part (400 profiled steps; ngd_pion.natural_pion's own GPU-synchronised timers)
PARTS = {"refresh": "refresh", "sigma": "sigma", "cayley": "cayley", "Finv": "F^-1",
         "shared": ("generators+momentum", "apply W")}
for t, tag in (("tfac1", "prof-alg1"), ("tfac25", "prof-tf25")):
    d = vrun(tag)
    pf_ = (d / "profile.json") if d is not None else None
    if pf_ is None or not pf_.exists():
        for k in list(PARTS) + ["step", "other"]:
            missing(f"prof_{t}_{k}", f"v2.sbatch row {tag} (writes profile.json)")
        continue
    pr = json.load(open(pf_))
    n, sec = pr["steps"], pr["seconds"]
    for k, names in PARTS.items():
        names = names if isinstance(names, tuple) else (names,)
        put(f"prof_{t}_{k}", sum(sec.get(x, 0.0) for x in names) / n, f"{V}/{tag}/profile.json: {' + '.join(names)} / steps")
    vsps(f"prof_{t}_step", tag, need_done=False)
    derive(f"prof_{t}_other", lambda s, *p: s - sum(p), (f"prof_{t}_step",) + tuple(f"prof_{t}_{k}" for k in PARTS),
           f"prof_{t}_step - the optimizer's parts: forward and backward with the statistics hooks, AdamW, data")

# 4.2: refresh cadence (Algorithm 1 = natural-pion; every row's angle tuned on its own grid)
vbest("tf1_nomom", grid("alg1-nomom", ("0.2", "0.4", "0.8")))
vbest("tf25_nomom", grid("tf25-nomom", ("0.4", "0.8", "1.6")))
vbest("tf1_mom", grid("alg1-mom", ("0.05", "0.1", "0.2")))
vbest("tf25_mom", grid("tf25-mom", ("0.1", "0.2", "0.4", "0.8")))
vval("tf10_mom", "tf10-s0")
for k in ("tf1_nomom", "tf25_nomom", "tf1_mom", "tf25_mom"):
    if NUM.get(k + "_tag"):
        vsps(k + "_sps", NUM[k + "_tag"])
    else:
        missing(k + "_sps", f"{k} grid")
vsps("tf10_mom_sps", "tf10-s0")
derive("tf25_nomom_d", lambda a, b: a - b, ("tf25_nomom", "tf1_nomom"), "tf25_nomom - tf1_nomom")
derive("tf10_mom_d", lambda a, b: a - b, ("tf10_mom", "tf1_mom"), "tf10_mom - tf1_mom")
derive("tf25_mom_d", lambda a, b: a - b, ("tf25_mom", "tf1_mom"), "tf25_mom - tf1_mom")
_tf10 = [vinfo("tf10-s0" if s == 0 else f"tf10-s{s}", s) for s in range(4)]
if all(_tf10):
    put("tf10_mom_mean", sum(i["val"] for i in _tf10) / 4, "v2 rows tf10-s0..s3, angle 0.2")
else:
    missing("tf10_mom_mean", "v2.sbatch rows tf10-s0..s3")
# Algorithm 1 over seeds at its best angle: seed 0 from the grid, seeds 1-3 from the rows "alg1-mom-seeds"
_s0 = NUM.get("tf1_mom")
_s = [vinfo("alg1-mom-seeds", s) for s in (1, 2, 3)]
if _s0 is not None and all(_s):
    put("tf1_mom_mean", (_s0 + sum(i["val"] for i in _s)) / 4, "Algorithm 1, best angle, seeds 0-3")
else:
    missing("tf1_mom_mean", "v2.sbatch rows alg1-mom-seeds, seeds 1-3, at the angle of tf1_mom (add once the grid is done)")
vval("tf1_mom_15k", "alg1-mom-15k")
derive("exact_vs_fast_3000", lambda a, b: a - b, ("v2_3000_mean", "tf1_mom_mean"), "v2_3000_mean - tf1_mom_mean")

# 4.3: statistics windows on the T_fac = 25 form, angle 0.2 (reference: tf25-mom-0.2)
vval("win_ref", "tf25-mom-0.2")
vsps("win_ref_sps", "tf25-mom-0.2")
for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095"):
    vval(f"win_{k}", f"win-{k}")
    vsps(f"win_{k}_sps", f"win-{k}")
_w = [NUM.get(f"win_{k}") for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095")]
if NUM.get("win_ref") is not None and all(v is not None for v in _w):
    put("win_maxdev", max(abs(v - NUM["win_ref"]) for v in _w), "largest |win_* - win_ref|")
else:
    missing("win_maxdev", "every win-* row")

# 4.5: batching by shape (Fast with exact Cayley, fp32 statistics, unbatched sigma, one A per matrix)
vval("batch_val", "form46")
vsps("batch_sps", "form46")

# 4.7: statistics inputs, fp32 against bf16 (the method)
vval("stats_fp32_val", "stats-fp32")
vsps("stats_fp32_sps", "stats-fp32")
vval("stats_fp32_15k", "stats-fp32-15k")
derive("stats_bf16_d", lambda a, b: a - b, ("m_3000", "stats_fp32_val"), "m_3000 - stats_fp32_val")
derive("stats_bf16_saving", lambda a, b: a - b, ("stats_fp32_sps", "m_sps"), "stats_fp32_sps - m_sps")

# cost of the method against Pion and Algorithm 1
derive("speedup_vs_exact", lambda a, b: a / b, ("tf1_mom_sps", "m_sps"), "tf1_mom_sps / m_sps")

# 5.4: direction, X = F^-1 G against X = G (best angle of each grid; 0.2 with momentum is the method's own run)
vbest("dir_fisher_nomom", grid("dir-nat-nomom", ("0.1", "0.2", "0.4", "0.8")))
vbest("dir_fisher_mom", ["dir-nat-mom-0.1", "s3000", "dir-nat-mom-0.4", "dir-nat-mom-0.8"])
vbest("dir_raw_nomom", grid("dir-raw-nomom", ("0.025", "0.05", "0.1", "0.2", "0.4")))
vbest("dir_raw_mom", grid("dir-raw-mom", ("0.1", "0.2", "0.4")))
derive("gain_fisher_nomom", lambda a, b: a - b, ("dir_raw_nomom", "dir_fisher_nomom"), "dir_raw_nomom - dir_fisher_nomom")
derive("gain_fisher_mom", lambda a, b: a - b, ("dir_raw_mom", "dir_fisher_mom"), "dir_raw_mom - dir_fisher_mom")

# 5.4: length rules, Fast with momentum; angle spread = median over logged steps >= 1000 of p90/p10 of the realised angle
GRIDS = {"before": ("3e-6", "1e-5", "3e-5", "1e-4", "3e-4"), "rms": ("1.5e-4", "3e-4", "6e-4", "1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3"),
         "fisher": ("2.4e-3", "4.8e-3", "9.6e-3", "1.9e-2", "3.8e-2"), "quad": ("1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"),
         "fixed": ("5e-4", "1e-3", "2e-3", "4e-3", "8e-3")}


def angle_spread(d: Path) -> float | None:
    rs = sorted(r["angle_p90"] / r["angle_p10"] for r in _rows(d)
                if r.get("step", 0) >= 1000 and r.get("angle_p10", 0) > 0 and "angle_p90" in r)
    return rs[len(rs) // 2] if rs else None


put("angle_spread_spectral", 1.0, "by construction: c = theta / sigma turns every matrix by theta")
for rule, g in GRIDS.items():
    vbest(f"len_{rule}", grid(f"len-{rule}", g))
    t = NUM.get(f"len_{rule}_tag")
    s_ = angle_spread(vrun(t)) if t else None
    (put(f"angle_spread_{rule}", s_, f"{V}/{t}: median over steps >= 1000 of angle_p90 / angle_p10") if s_ is not None
     else missing(f"angle_spread_{rule}", f"len-{rule} grid"))
# the RMS rule against the spectral one over seeds (RMS at its best angle: rows "len-rms-seeds", seeds 1-3)
for s in range(4):
    rms = NUM.get("len_rms") if s == 0 else (vinfo("len-rms-seeds", s) or {}).get("val")
    spec = NUM.get("m_3000") if s == 0 else NUM.get(f"v2_3000_s{s}")
    if rms is not None and spec is not None:
        put(f"rms_gap_s{s}", rms - spec, f"RMS rule at its best angle - the method, seed {s}")
    else:
        missing(f"rms_gap_s{s}", "v2.sbatch rows len-rms-seeds at the angle of len_rms (add once the grid is done)")

# 5.4: regularisation of the Fisher (reference: the method, floor 1e-4)
for e in ("1e-6", "1e-5", "1e-3", "1e-2"):
    vval(f"reg_eps_{e}", f"reg-eps{e}")
for dmp in ("1e-4", "1e-3", "1e-2", "1e-1"):
    vval(f"reg_damp_{dmp}", f"reg-damp{dmp}")
_r = [NUM.get(f"reg_eps_{e}") for e in ("1e-6", "1e-5", "1e-3", "1e-2")] + \
     [NUM.get(f"reg_damp_{d}") for d in ("1e-4", "1e-3", "1e-2", "1e-1")]
if REF is not None and all(v is not None for v in _r):
    put("reg_best_other_d", min(_r) - REF, "best of the eight alternatives - m_3000")
else:
    missing("reg_best_other_d", "every reg-* row and s3000")

# 5.4: other choices (reference: the method, angle 0.2, cosine)
vbest("sides_best", grid("sides-each", ("0.1", "0.2")))
for m in ("0.25", "0.5", "2", "4"):
    vval(f"attn_{m}", f"attn-x{m}")
vbest("sched_constant", grid("sched-constant", ("0.05", "0.1", "0.2")))
vbest("sched_linear", grid("sched-linear", ("0.1", "0.2", "0.4")))
vbest("sched_warmcos", grid("sched-warmcos", ("0.1", "0.2", "0.4")))
for dd in ("0.2", "0.1"):
    vval(f"wsd_{dd}", f"sched-wsd-{dd}")
vbest("polar", grid("planes-equal", ("0.1", "0.2")))
_p = [vsps(f"_polar_sps_{a}", f"planes-equal-{a}") for a in ("0.1", "0.2")]
if all(v is not None for v in _p):
    put("svd_sps_lo", min(_p), "planes-equal runs, wall / steps, min over angles")
    put("svd_sps_hi", max(_p), "same, max")
else:
    missing("svd_sps_lo", "planes-equal rows"); missing("svd_sps_hi", "planes-equal rows")

# Appendix D: the initial spectrum (Natural Pion Fast; Pion's runs as before)
put("init_normal", REF, "= m_3000") if REF is not None else missing("init_normal", "s3000")
for k in ("2", "2.5", "3", "4", "5", "flat", "orth"):
    vval(f"init_{k}", f"init-pl{k}" if k[0].isdigit() else f"init-{k}")
derive("init_best_d", lambda *v: min(v[1:]) - v[0], ("init_normal", "init_2", "init_2.5", "init_3", "init_4", "init_5"),
       "best power-law tail - init_normal")
derive("init_flat_d", lambda a, b: a - b, ("init_flat", "init_normal"), "init_flat - init_normal")
derive("init_orth_d", lambda a, b: a - b, ("init_orth", "init_normal"), "init_orth - init_normal")
_g = [(NUM.get(f"v2_3000_s{s}"), (vinfo("init-pl3", s) or {}).get("val")) for s in range(4)]
if all(a is not None and b is not None for a, b in _g):
    put("init_gain_3000_mean", sum(a - b for a, b in _g) / 4, "Natural Pion, alpha 3 vs normal, 3000 steps, seeds 0-3")
else:
    missing("init_gain_3000_mean", "init-pl3 seeds 0-3 and s3000 seeds 0-3")
_i15 = vinfo("init-pl3-15k")
derive("init_gain_15k", lambda a: a - _i15["val"], ("m_15000",), "m_15000 - init-pl3-15k") if _i15 else missing("init_gain_15k", "v2.sbatch row init-pl3-15k")
_if = vinfo("init-pl3-73242")
derive("init_gain_full", lambda a: a - _if["val"], ("v2_full",), "v2_full - init-pl3-73242") if _if else missing("init_gain_full", "v2.sbatch row init-pl3-73242")
I_, IL = "stepnorminit", "stepnorminitlong"
put("init_pion_gain_3000", NUM["pion3000_s0"] - val(f"{IL}/pion3000-lr1e-3-pl3/*"), "Pion, 3000 steps")
put("init_pion_gain_15k", NUM["pion15k"] - val(f"{IL}/pion15000-lr1e-3-pl3/*"), "Pion, 15000 steps")
put("init_pion_gain_full", NUM["pion_full"] - val(f"{IL}/pion73242-lr1e-3-pl3/*"), "Pion, 73242 steps")

# trained spectrum drift per retraction, in units of each matrix's largest singular value (retraction_drift.py,
# on the v2 checkpoints: ret-cayley, ret-ns, ret-trunc and the method's s3000)
f = R / "probes" / "retraction_drift.json"
d = json.load(open(f)) if f.exists() else {}
for tag in ("cayley", "ns", "trunc", "ns_delta"):
    if tag in d and d[tag].get("setting") == "v2":
        put(f"rd_{tag}_med", d[tag]["scaled_median"], f"{f.name}: median over the rotated matrices of max|s1-s0|/max s0")
        put(f"rd_{tag}_max", d[tag]["scaled_max"], f"{f.name}: worst matrix, same measure")
        put(f"rd_{tag}_signed", d[tag]["signed_mean"], f"{f.name}: mean signed relative change")
    else:
        for k in ("med", "max", "signed"):
            missing(f"rd_{tag}_{k}", "python scripts/probes/retraction_drift.py (run by collect_npv2.sh once the checkpoints exist)")


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

if "svd_exact_s" in pc and all(NUM.get(f"prof_tfac25_{k}") is not None for k in ("refresh", "cayley", "Finv", "shared")):
    rest = sum(NUM[f"prof_tfac25_{k}"] for k in ("refresh", "cayley", "Finv", "shared"))
    put("svd_share_of_step", pc["svd_exact_s"] / (pc["svd_exact_s"] + rest),
        "exact spectral norms / (that + the rest of the T_fac 25 optimizer step: refresh, Cayley, F^-1, generators, rotation)")
else:
    missing("svd_share_of_step", "paper_checks.json and the prof-tf25 profile")

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
