#!/bin/bash
#SBATCH -A p330
#SBATCH -J shampoo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# First GPU reading of `shampoo-pion`: `eta` swept, bracketed on both sides.
#
# The preconditioner is built from the rotation generators themselves rather
# than from network statistics -- see ALGORITHM.md §8 and the journal entry of
# 2026-08-27 evening, which carries the pre-registration written before this
# was submitted.
#
# WHY THIS BRACKET. At step 1 the accumulator holds exactly one gradient, so
# `P = G G^T` and the sandwich orthogonalises the generator: every rotation
# plane turns by the same angle and `||X||_2 = 1` exactly. **`eta` is therefore
# the rotation angle in radians on the first step**, which is the first time in
# this project that the learning rate has had a direct physical reading. The
# grid spans 3e-4 to 1e-1 rad.
#
# Expect the effective step to decay: with `shampoo_beta = 0` the accumulator
# is a plain sum, so `P ~ t` and `||X|| ~ t^-1/2`. That is Adagrad's decay and
# it compounds with the cosine schedule. It is a property to read off `angle`,
# not a bug.
#
# PRE-REGISTERED, and all three are falsifiable here:
#   * The cross-layer spread of `angle` falls from the Fisher variant's
#     4658x-36496x to order 1e1 or below. **This is readable at step 1**, from
#     the first diagnostics row of any arm, without waiting for the run.
#   * `eta` shows a broad plateau rather than the sharp optimum `ngd-pion-s`
#     has at 1e-2.
#   * If an arm blows up, the first hypothesis is step scale, not the idea:
#     raw Shampoo without grafting is known unreliable there, and grafting the
#     norm -- the analogue of Pion's `rms` -- is the named fallback.
#
# WHAT IT IS COMPARED AGAINST, and not at a shared `eta`. `pion_ablated` at its
# own swept optimum is 6.1143 at 150 steps; `ngd-pion` at its own is 5.9113;
# `ngd-pion-s` at its own is 5.5068. Those are the three numbers on the board.
#
# `--shampoo-plane-every 30` turns on the per-layer plane-angle spread, which is
# off by default because it costs an `svdvals` per side per layer. It is
# identical across arms, so it cannot bias the comparison.
#
#   sbatch -p rtx scripts/sbatch/shampoo.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/shampoo"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
COMMON="--optimizer shampoo-pion --ngd-t-fac 25 --adamw-lr 1e-3 \
        --precision bf16 --seed 0 --shampoo-plane-every 30 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin --out-dir $OUT"

echo "node $(hostname)  job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name --format=csv,noheader

run () { echo; echo "=== eta=$1 ==="
         $EXEC python -m harness.run $COMMON --lr "$1" \
           || echo "    FAILED at eta=$1 -- continuing"; }

# Space-separated `eta` list. A subset can be run first and the rest appended
# later: each arm writes its own directory under `$OUT` and the summary reads
# all of them.
ARMS="${ARMS:-3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
for ETA in $ARMS; do run "$ETA"; done

echo
echo "=== the grid ==="
$EXEC python -c '
import glob, json, os, sys
rows = []
for m in glob.glob(os.path.join(sys.argv[1], "*", "manifest.json")):
    d = os.path.dirname(m); cfg = json.load(open(m))["config"]
    val = float("nan"); steps = -1; a_lo = a_hi = float("nan")
    p = os.path.join(d, "log.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try: r = json.loads(line)
            except ValueError: continue
            if "val_loss" in r: val = r["val_loss"]
            if "train_loss" in r: steps = max(steps, r["step"])
            if "angle_min" in r and r["angle_min"] == r["angle_min"]:
                a_lo, a_hi = r["angle_min"], r["angle_max"]
    rows.append((cfg["lr"], steps, val, a_lo, a_hi))
print("%9s%7s%10s%12s%12s%10s" % ("eta","steps","val@150","angle_min","angle_max","spread"))
for lr, steps, val, lo, hi in sorted(rows):
    sp = hi / lo if lo == lo and lo > 0 else float("nan")
    print(f"{lr:>9g}{steps:>7}{val:>10.4f}{lo:>12.3e}{hi:>12.3e}{sp:>10.1f}x")
ok = [r for r in rows if r[2] == r[2]]
if ok:
    best = min(ok, key=lambda r: r[2])
    print(f"\n  best: eta = {best[0]:g} at {best[2]:.4f}")
print("  on the board at 150 steps, each at its OWN optimum:")
print("    pion_ablated 6.1143   ngd-pion 5.9113   ngd-pion-s 5.5068")
print("  pre-registered: angle spread should be order 1e1 or below,")
print("  against 4658x-36496x for the Fisher variant")
' "$OUT"
