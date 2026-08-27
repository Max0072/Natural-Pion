#!/bin/bash
#SBATCH -A p330
#SBATCH -J eta-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# `eta` for the measured `S`, which it has never been given.
#
# `with_s.py` says in its own docstring that `eta` does not carry over from the
# `S = I` arm -- `F` scales with `||D||`, which is 1e-18 to 1e-12 here -- and
# that it has to be swept from scratch. It never was: every non-MC
# `ngd-pion-s` run on disk sits at `eta = 1`, and the sweep launched for it
# (`with-s`, 1e-8 to 1e-14) died with `steps = 0` in all four directories. So
# "the measured `S` is worse" currently rests on one point at the one `eta`
# the module predicts will be wrong.
#
# One variable. `ngd-pion` (FastNGDPion) and `ngd-pion-s` (FastNGDPionS) share
# the floor on the pencil, `power = 1`, the diagnostics and everything else;
# the difference is `S = I` against `S = W^T E[dd^T] W`. `ngd-pion`'s own `eta`
# curve already exists, densely, from 0.003 to 300, with its optimum at 5.74 to
# 5.83. This supplies the other half of the pair.
#
# It also separates `S` from MC sampling, which `mcfisher2` did not: that arm
# is the best 150-step number in the project (5.5517 at `eta = 0.01`) and
# carries both changes at once, so the 0.28 it wins by is unattributed.
#
# `--ngd-power 1.0` is passed to make the manifest honest rather than to change
# anything: `build_optimizers` passes `power` only for `ngd-pion-pow`, so
# `FastNGDPionS` runs at `Basis.power = 1.0` whatever the config says, and the
# default of 0.5 would be recorded as a controlled variable that was not one.
#
# The grid is centred on `mcfisher2`'s optimum and brackets it on both sides,
# which that grid did not -- 0.01 was its lowest point.
#
#   sbatch -p rtx scripts/sbatch/eta_s.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/eta-s"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"

# Matched to the `ngd-pion` arm this is compared against: T_fac 25, AdamW
# pinned at 1e-3 so the sweep varies the rotational rate alone, bf16, seed 0,
# log every 30 steps so the per-layer diagnostics line up with mcfisher2.
COMMON="--optimizer ngd-pion-s --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 \
        --ngd-fisher-mc-every 0 --ngd-power 1.0 --seed 0 --max-steps 150 \
        --log-every 30 --no-resume \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin \
        --out-dir $OUT"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}  job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for LR in 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
  echo
  echo "=== ngd-pion-s  eta = $LR ==="
  if ! $EXEC python -m harness.run $COMMON --lr "$LR"; then
    echo "    FAILED at eta = $LR -- continuing with the rest"
  fi
done

echo
echo "=== the curve ==="
$EXEC python -c '
import glob, json, os, sys
rows = []
for m in glob.glob(os.path.join(sys.argv[1], "*", "manifest.json")):
    d = os.path.dirname(m)
    cfg = json.load(open(m))["config"]
    val = float("nan"); steps = -1
    p = os.path.join(d, "log.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try: r = json.loads(line)
            except ValueError: continue
            if "val_loss" in r: val = r["val_loss"]
            if "train_loss" in r: steps = max(steps, r["step"])
    rows.append((cfg["lr"], steps, val, os.path.basename(d)))
print("%9s%7s%10s   run" % ("eta", "steps", "val@150"))
for lr, steps, val, name in sorted(rows):
    print(f"{lr:>9g}{steps:>7}{val:>10.4f}   {name}")
print()
ok = [r for r in rows if r[2] == r[2]]
if ok:
    best = min(ok, key=lambda r: r[2])
    print(f"  best: eta = {best[0]:g} at {best[2]:.4f}")
    print(f"  against ngd-pion (S = I), same settings, its own swept optimum: 5.74 - 5.83")
    print(f"  against ngd-pion-s + MC sampling at eta = 0.01: 5.5517")
    if best[0] in (3e-4, 1e-1):
        print(f"  WARNING: the optimum is at the edge of the grid -- not bracketed")
' "$OUT"
