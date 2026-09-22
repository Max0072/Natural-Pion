#!/bin/bash
#SBATCH -A p330
#SBATCH -J ladder15k2
#SBATCH --array=0-5%6
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Wave 2 of the H4 horizon ladder (docs/PLAN.md). Wave 1 (job 332064) put both
# gradients for ngd-pion-s pointing down and left off the cross it was given:
#
#     rot 3e-3  adamw 8e-3   3.5302   <- best of wave 1, LOWER EDGE of the rot arm
#     rot 6e-3  adamw 4e-3   3.5425   <- second best, LOWER EDGE of the adamw arm
#     rot 6e-3  adamw 8e-3   3.5448   (the 3000-step optimum)
#
# A cross tests one axis at a time and cannot see a corner; both single-axis
# trends point at (rot ~1.5-3e-3, adamw ~2-4e-3), a cell nobody has run. This
# fills a 3x3 grid, rot x {1.5e-3, 3e-3, 6e-3}, adamw x {2e-3, 4e-3, 8e-3}, minus
# the three wave-1 cells that are corners of it already (3e-3/8e-3, 6e-3/4e-3,
# 6e-3/8e-3) -- six new cells here.
#
# pion is NOT extended: its wave-1 best (rot 1e-3, adamw 8e-3, val 3.5254) beat
# both neighbours on each axis (rot 5e-4: 3.5818, rot 2e-3: 3.5369; adamw 4e-3:
# 3.5316, adamw 1.6e-2: 3.5360), so it is already interior.
#
# THE RULE, unchanged from wave 1 (docs/PLAN.md H4, sbatch header of
# ladder15k.sbatch): gap15 = val(best pion, 3.5254, fixed) - val(best ngd-pion-s
# over wave 1 + wave 2). >= +0.020 persists; +0.010 to +0.020 decaying but
# present; < +0.010 inside noise or gone. If the new best again sits on an edge
# of this 3x3 (rot 1.5e-3 or adamw 2e-3), extend again rather than quote.
#
# COST. rtx6002 is idle (checked before submitting): all six run at once, one GPU
# each. ~3.5 h per run from wave 1's elapsed time (15000 steps, B=512), so about
# 3.5-4 h wall for the array against the original 6 h limit.
#
# RESUBMITTED 2026-09-22, 19:37, after all six original tasks wedged. They were
# not slow from a bad step -- they were I/O-bound from nearly step 0 (23 steps
# took 80 minutes) and got worse once two more concurrent readers (the
# rho_held_trust job) landed on the same node around 18:46, fully stalling by
# 19:03 (0% GPU, load 8.00, 14 GB of page cache against a ~20 GB corpus with
# 968 GB of RAM sitting free -- the corpus was not staying resident). A
# single-task resubmission (array index 1 alone) recovered to a healthy,
# stable ~3.3-4 s/step -- slower than the usual ~0.9 s/step solo figure but a
# real, computing rate, not a wedge. At 3.5 s/step a 15000-step run is ~14.6 h,
# hence the limit raised to 20 h. Confirmed working as designed: the stale
# 248-minute-old run-directory lock left by the dead process was detected and
# taken over automatically (`harness/train.py`'s per-directory lock), so the
# resubmission needed no manual cleanup of the run directories.
#
#   sbatch -p rtx scripts/sbatch/ladder15k_wave2.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/ladder15k"

ROT=( 1.5e-3 1.5e-3 1.5e-3 3e-3   3e-3   6e-3   )
ADW=( 2e-3   4e-3   8e-3   2e-3   4e-3   2e-3   )

T=$SLURM_ARRAY_TASK_ID
L="${ROT[$T]}"; A="${ADW[$T]}"

echo "node $(hostname)  task $T  ngd-pion-s  rot=$L  adamw=$A  batch=512  steps=15000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr "$A" \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 15000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 1000 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
